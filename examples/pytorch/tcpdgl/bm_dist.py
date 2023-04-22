import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import random
import dgl.nn as dglnn
import pandas
import numpy as np
import os
import time
import tqdm
from dgl.data import CoraGraphDataset, RedditDataset
from dgl.data.utils import save_graphs, load_graphs
from ogb.nodeproppred import DglNodePropPredDataset

import torch.distributed as dist
import torch.multiprocessing as mp
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from torch.nn.parallel import DistributedDataParallel

def LoadDataset(graph_name, n_feat):
    data_path = '/mnt/data2/chijj/data'
    dgl_data_path = os.path.join(data_path, 'dgl')
    cache_path = os.path.join(data_path, graph_name, 'dglgraph.bin')
    print(data_path)
    if graph_name == 'cora':
        if os.path.exists(cache_path):
            graph, label_dict = load_graphs(cache_path)
            graph = graph[0]
            num_classes = torch.max(graph.ndata['label']) + 1
        else:
            dataset = CoraGraphDataset(dgl_data_path)
            graph = dataset[0]
            num_classes = dataset.num_classes
        train_idx = torch.nonzero(graph.ndata['train_mask']).squeeze()
        valid_idx = torch.nonzero(graph.ndata['val_mask']).squeeze()
        test_idx = torch.nonzero(graph.ndata['test_mask']).squeeze()
    elif graph_name == 'Reddit':
        if os.path.exists(cache_path):
            graph, label_dict = load_graphs(cache_path)
            graph = graph[0]
            num_classes = torch.max(graph.ndata['label']) + 1
        else:
            dataset = RedditDataset(raw_dir=dgl_data_path, verbose=True)
            graph = dataset[0]
            num_classes = dataset.num_classes    
        train_idx = torch.nonzero(graph.ndata['train_mask']).squeeze()
        valid_idx = torch.nonzero(graph.ndata['val_mask']).squeeze()
        test_idx = torch.nonzero(graph.ndata['test_mask']).squeeze()
    elif graph_name == 'ogbn_products':
        if os.path.exists(cache_path):
            graph, label_dict = load_graphs(cache_path)
            graph = graph[0]
            num_classes = torch.max(graph.ndata['label']) + 1
        else:
            dataset = DglNodePropPredDataset('ogbn-products', data_path)
            graph, label = dataset[0]
            num_classes = dataset.num_classes
            graph.ndata['label'] = label.squeeze()
        train_pc=0.1
        test_pc=0.8
        print('[PF] bg split',time.time())
        idx = [i for i in range(graph.num_nodes())]
        random.shuffle(idx)
        train_idx = torch.tensor(idx[:int(graph.num_nodes() * train_pc)])
        test_idx = torch.tensor(idx[int(graph.num_nodes() * train_pc) : int(graph.num_nodes() * (train_pc + test_pc))])
        valid_idx = torch.tensor(idx[int(graph.num_nodes() * (train_pc + test_pc)) : graph.num_nodes()])
        print('[PF] ed split',time.time())
        # split_idx = dataset.get_idx_split()
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    else:
        if os.path.exists(cache_path):
            graph, label_dict = load_graphs(cache_path)

            graph = graph[0]
            num_classes = torch.max(graph.ndata['label'])+1
        else:
            file_path = os.path.join(data_path, graph_name, 'un_sort')
            num_classes = 50
            feat_dim = n_feat
            feat_lr=-10.0
            feat_rr=10.0

            
            df = pandas.read_csv(file_path , header=None, names=["Src", "Dst"], sep=' ')
            src = df["Src"].to_numpy()
            dst = df["Dst"].to_numpy()
            graph = dgl.graph((src, dst), idtype=torch.int64)
            
            label = [random.randint(0, num_classes-1) for i in range(graph.num_nodes())]
            feat = []
            for i in range(graph.num_nodes()):
                node_feat = [random.uniform(feat_lr, feat_rr) for j in range(feat_dim)]
                feat.append(node_feat)
            feat = torch.tensor(feat).to(torch.float32)
            label = torch.tensor(label).to(int)
            graph._node_frames = [dgl.frame.Frame({"feat" : feat, "label": label})]
            graph._edge_frames = [dgl.frame.Frame()]

        idx = [i for i in range(graph.num_nodes())]
        train_pc = 0.1
        test_pc = 0.8
        random.shuffle(idx)
        train_idx = torch.tensor(idx[:int(graph.num_nodes() * train_pc)])
        test_idx = torch.tensor(idx[int(graph.num_nodes() * train_pc) : int(graph.num_nodes() * (train_pc + test_pc))])
        valid_idx = torch.tensor(idx[int(graph.num_nodes() * (train_pc + test_pc)) : graph.num_nodes()])

    print('n_nodes: {}, n_edges: {}, feat: {}, label: {}, num_class: {}'.format(
        graph.num_nodes(), graph.num_edges(), graph.ndata['feat'].size(), graph.ndata['label'].size(), num_classes))
    return graph, train_idx, valid_idx, test_idx, num_classes

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
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
            dataloader = DataLoader(
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
    proc_id, nprocs, device, g, num_classes, train_idx, val_idx, model, use_uva
):
    sampler = NeighborSampler(
        [10, 10, 10], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = DataLoader(
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
    val_dataloader = DataLoader(
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
    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        acc = (
            evaluate(model, g, num_classes, val_dataloader).to(device) / nprocs
        )
        dist.reduce(acc, 0)
        if proc_id == 0:
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, total_loss / (it + 1), acc.item()
                )
            )


def run(proc_id, nprocs, devices, g, data, mode):
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
    num_classes, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    g = g.to(device if mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 256, num_classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    use_uva = mode == "mixed"
    train(
        proc_id,
        nprocs,
        device,
        g,
        num_classes,
        train_idx,
        val_idx,
        model,
        use_uva,
    )
    # layerwise_infer(proc_id, device, g, num_classes, test_idx, model, use_uva)
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ogbn-products Reddit
    parser.add_argument("-g","--graph_name", type=str, default='cora', help="dataset name")
    parser.add_argument("--use_uva", action='store_true', default=False, help="use unified virtual space")
    parser.add_argument("--pin", action='store_true', default=False, help="pin graph features")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-e", "--n_epoch", type=int, default=10, help="number of training epochs")
    parser.add_argument("-f", "--n_feat", type=int, default=10, help="number of features")
    parser.add_argument("-s", "--sampler",type=str, default="sage", choices=["sage", "labor"], help="graph sampler")
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

    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # load and preprocess dataset
    print("Loading data")
    graph, train_idx, valid_idx, test_idx, num_classes = LoadDataset(args.graph_name, args.n_feat)
    # avoid creating certain graph formats in each sub-process to save momory
    graph.create_formats_()
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        num_classes.item(),
        train_idx,
        valid_idx,
        test_idx,
    )
    mp.spawn(run, args=(nprocs, devices, graph, data, args.mode), nprocs=nprocs)
