import argparse
import dgl
import dgl.nn as dglnn
from dgl.data.utils import load_ccg_for, load_feat
from dgl.data import DGLDataset
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.data import CoraGraphDataset
import time
import numpy as np
import os
from torch.profiler import profile, record_function, ProfilerActivity
from dgl.utils.log import pflogger
import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from dgl.dataloading import (
    CCGDataLoader,
    CCGNeighborSampler
)
from dgl.multiprocessing import shared_tensor
from torch.nn.parallel import DistributedDataParallel

PROF_FLAG = False

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        # self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = CCGDataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size
                    if l != len(self.layers) - 1
                    else self.out_size,
                )
            )
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y


def evaluate(model, g, num_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(
    proc_id, device, g, num_classes, nid, model, use_uva, batch_size=2**16
):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        print("Test Accuracy {:.4f}".format(acc.item()))


def train(
    proc_id, nprocs, device, g, num_classes, train_idx, val_idx, model, use_uva, epochs
):
    sampler = CCGNeighborSampler(
        [20, 5], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = CCGDataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    val_dataloader = CCGDataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(epochs):
        if dist.get_rank() == 0:
            pflogger.info('bg epoch %f', time.time())
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            if dist.get_rank() == 0:
                for bid, block in enumerate(blocks):
                    pflogger.info(f'stat block{bid}.src_nodes {block.number_of_src_nodes()}')
                    pflogger.info(f'stat block{bid}.dst_nodes {block.number_of_dst_nodes()}')
                    pflogger.info(f'stat block{bid}.edges {block.number_of_edges()}')
                pflogger.info('bg model_compution %f', time.time())
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
            if dist.get_rank() == 0:
                pflogger.info('ed model_compution %f', time.time())
        acc = evaluate(model, g, num_classes, val_dataloader).to(device) / nprocs
        dist.reduce(acc, 0)

        if dist.get_rank() == 0:
            if epoch % 5 == 0 or epoch == epochs - 1:
                pflogger.info("stat epoch{:d}_loss {:.4f}".format(epoch, total_loss / (it + 1)))
                pflogger.info('stat epoch{:d}_acc {:.4f}'.format(epoch, acc.item()))
            pflogger.info('ed epoch %f', time.time())
    if dist.get_rank() == 0:
        pflogger.info('ed end2end %f', time.time())

class CCGDataset(DGLDataset):
    def __init__(self, _graph_name, _ccg_name, _graph_path, _feat_path=None,  _num_classes=50, _feat_dim=10, _feat_lr=-10.0, _feat_rr=10.0):
        self.graph_name = _graph_name
        self.file_path = _graph_path
        self.num_classes = _num_classes
        self.feat_path = _feat_path
        self.feat_dim = _feat_dim
        self.feat_lr = _feat_lr
        self.feat_rr = _feat_rr
        super().__init__(name=_ccg_name)
    
    def read_dgl_graph(self):
        return dgl.graph(([],[]))

    def process(self):
        self.graph = self.read_dgl_graph()
        self.graph = load_ccg_for(self.graph, self.file_path, self.feat_path)
        
        if self.graph.ndata == {}:
            label = [random.randint(0, self.num_classes - 1) for i in range(self.graph.ccg.v_num)]
            feat = []
            for i in range(self.graph.ccg.v_num):
                node_feat = [random.uniform(self.feat_lr, self.feat_rr) for j in range(self.feat_dim)]
                feat.append(node_feat)
            feat = torch.tensor(feat).to(torch.float32)
            label = torch.tensor(label).to(int)
            self.graph._node_frames = [dgl.frame.Frame({"feat" : feat, "label": label})]
            self.graph._edge_frames = [dgl.frame.Frame()]
        else:
            self.num_classes = self.graph.ndata['label'].max() + 1
    
    def __getitem__(self, i):
        assert i == 0, 'This dataset has only one graph'
        return self.graph
    
    def __len__(self):
        return 1

    def get_split(self, train_pc=0.1, test_pc=0.8, valid_pc=0.1):
        idx = [i for i in range(self.graph.ccg.v_num)]
        random.shuffle(idx)
        if self.graph_name == 'cora':
            # 140, 1000, 500
            train_idx = torch.tensor(idx[:140])
            test_idx = torch.tensor(idx[140:1140])
            valid_idx = torch.tensor(idx[1140:1640])
        elif self.graph_name == 'Reddit':
            type_path = '/mnt/data2/chijj/data/Reddit/node_types.npy'
            node_types = np.load(type_path)
            train_idx = torch.tensor(np.nonzero(node_types==1)[0])
            valid_idx = torch.tensor(np.nonzero(node_types==2)[0])
            test_idx = torch.tensor(np.nonzero(node_types==3)[0])
        else:
            train_idx = torch.tensor(idx[:int(self.graph.ccg.v_num * train_pc)])
            test_idx = torch.tensor(idx[int(self.graph.ccg.v_num * train_pc) : int(self.graph.ccg.v_num * (train_pc + test_pc))])
            valid_idx = torch.tensor(idx[int(self.graph.ccg.v_num * (train_pc + test_pc)) : self.graph.ccg.v_num])
        return train_idx, test_idx, valid_idx

def run(proc_id, nprocs, devices, args):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )

    ccg_name = "tc1_k32_h2"
    file_path = os.path.join(os.path.join("/mnt/data2/chijj/data", args.graph_name), "tc1_k32_h2")
    ck_feat_path = os.path.join(os.path.join("/mnt/data2/chijj/data", args.graph_name), "feat")
    feat_path = ck_feat_path if os.path.isfile(ck_feat_path) else None

    if not dist.is_initialized() or dist.get_rank() == 0:
        pflogger.info('start %f', time.time())
    
    # load and preprocess dataset
    if dist.get_rank() == 0:
        pflogger.info('bg load_graph %f', time.time())
    dataset = CCGDataset(args.graph_name, ccg_name, file_path, feat_path)
    train_idx, test_idx, valid_idx = dataset.get_split()
    if dist.get_rank() == 0:
        pflogger.info('ed load_graph %f', time.time())
    graph = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    # graph.create_formats_()
    # thread limiting to avoid resource competition
    
    if isinstance(dataset.num_classes, torch.Tensor):
        dataset.num_classes = dataset.num_classes.item()

    if dist.get_rank() == 0:
        pflogger.info('bg end2end %f', time.time())
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    if dist.get_rank() == 0:
        pflogger.info('bg to_dvs %f', time.time())
    graph = graph.to(device if args.mode == "puregpu" else "cpu")
    if dist.get_rank() == 0:
        pflogger.info('ed to_dvs %f', time.time())
    # create GraphSAGE model (distributed)
    in_size = graph.ndata["feat"].shape[1]
    model = SAGE(in_size, 256, dataset.num_classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    use_uva = args.mode == "mixed"
    train(
        proc_id,
        nprocs,
        device,
        graph,
        dataset.num_classes,
        train_idx,
        valid_idx,
        model,
        use_uva,
        args.n_epoch
    )
    # layerwise_infer(proc_id, device, graph, num_classes, test_idx, model, use_uva)
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # orkut patents lj1
    # cora Reddit ogbn_products
    parser.add_argument("-g","--graph_name", type=str, default='toy', help="dataset name")
    parser.add_argument("--use_uva", action='store_true', default=False, help="use unified virtual space")
    parser.add_argument("--pin", action='store_true', default=False, help="pin graph features")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("-e", "--n_epoch", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("-f", "--n_feat", type=int, default=10,
                        help="number of features")
    parser.add_argument("-s", "--sampler",type=str, default="sage", choices=["sage", "labor"], help="graph sampler")
    parser.add_argument("--cnt", type=int, default=-1, help="number of experiments")
    parser.add_argument(
        "--mode",
        default="puregpu",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    args = parser.parse_args()
    print(args)
    torch.cuda.reset_peak_memory_stats()

    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)

    mp.spawn(run, args=(nprocs, devices, args), nprocs=nprocs)
