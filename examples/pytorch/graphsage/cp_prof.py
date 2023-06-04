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

PROF_FLAG = False

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        feat = g.ndata['feat']
        print(feat)
        sampler = dgl.dataloading.CCGMultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.CCGDataLoader(
                g, torch.arange(g.ccg.v_num).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                g.ccg.v_num, self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device)
            print(y, y.shape)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in dataloader:
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
                print(y, y.shape)
            #     y = torch.zeros(
            #     g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            #     device=buffer_device)
            # print('y', y, sep='\n')
            # for input_nodes, output_nodes, blocks in dataloader:
            #     x = blocks[0].srcdata['h']
            #     h = layer(blocks[0], x)
            #     if l != len(self.layers) - 1:
            #         h = F.relu(h)
            #         h = self.dropout(h)
            #     y[output_nodes] = h.to(buffer_device)
            #     print(y)
            # # print('y: ')
            # # print(y)
            # g._node_frames[0]['h'] = y
            feat = y
        return y

class CCGDataset(DGLDataset):
    def __init__(self, _graph_name, _ccg_name, _graph_path, _feat_path=None,  _num_classes=50, _feat_dim=10, _feat_lr=-10.0, _feat_rr=10.0):
        print("Graph Dataset : ", _graph_name)
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
            type_path = '/home/chijj/data/Reddit/node_types.npy'
            node_types = np.load(type_path)
            train_idx = torch.tensor(np.nonzero(node_types==1)[0])
            valid_idx = torch.tensor(np.nonzero(node_types==2)[0])
            test_idx = torch.tensor(np.nonzero(node_types==3)[0])
        else:
            train_idx = torch.tensor(idx[:int(self.graph.ccg.v_num * train_pc)])
            test_idx = torch.tensor(idx[int(self.graph.ccg.v_num * train_pc) : int(self.graph.ccg.v_num * (train_pc + test_pc))])
            valid_idx = torch.tensor(idx[int(self.graph.ccg.v_num * (train_pc + test_pc)) : self.graph.ccg.v_num])
        return train_idx, test_idx, valid_idx

# @profile
def main():
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # orkut patents lj1
    # cora Reddit ogbn_products
    parser.add_argument("-g","--graph_name", type=str, default='toy', help="dataset name")
    parser.add_argument("--use_uva", action='store_true', default=False, help="use unified virtual space")
    parser.add_argument("--pin", action='store_true', default=False, help="pin graph features")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("-e", "--n_epoch", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("-f", "--n_feat", type=int, default=10,
                        help="number of features")
    parser.add_argument("-s", "--sampler",type=str, default="sage", choices=["sage", "labor"], help="graph sampler")
    parser.add_argument_group()
    args = parser.parse_args()
    print(args)
    torch.cuda.reset_peak_memory_stats()

    device = 'cuda'
    ccg_name = "tc1_k32_h2"
    file_path = os.path.join(os.path.join("/mnt/data2/chijj/data", args.graph_name), "tc1_k32_h2")
    ck_feat_path = os.path.join(os.path.join("/mnt/data2/chijj/data", args.graph_name), "feat")
    feat_path = ck_feat_path if os.path.isfile(ck_feat_path) else None

    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    #     with record_function("prepare"):
    print('[PF] start', time.time())
    print('[PF] bg load_graph', time.time())
    dataset = CCGDataset(args.graph_name, ccg_name, file_path, feat_path)
    print('[PF] bg split',time.time())
    train_idx, test_idx, valid_idx = dataset.get_split()
    print('[PF] ed split',time.time())
    graph = dataset[0]
    print('[PF] ed load_graph', time.time())
    
    print('[PF] bg end2end', time.time())
    print('[PF] bg to_dvs', time.time())
    if args.use_uva == False:
        graph = graph.to(device, move_feat=not args.pin)
    print('[PF] ed to_dvs', time.time())
    # input()

    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)

    model = SAGE(graph.ndata['feat'].shape[1], 256, dataset.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    fanout = [20, 5]
    print('[PF] bg pre_sampler', time.time())

    if args.sampler=='sage':
        sampler = dgl.dataloading.CCGNeighborSampler(
                fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    elif args.sampler=='labor':
        sampler = dgl.dataloading.CCGLaborSampler(
            fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'], importance_sampling=-2)
    train_dataloader = dgl.dataloading.CCGDataLoader(
            graph, train_idx, sampler, device=device, batch_size=204800, shuffle=True,
            drop_last=False, num_workers=0, pin_prefetcher=args.pin, use_uva=args.use_uva)
    print('[PF] ed pre_sampler', time.time())
    # valid_dataloader = dgl.dataloading.CCGDataLoader(
    #         graph, valid_idx, sampler, device=device, batch_size = 204800, shuffle=True,
    #         drop_last=False, num_workers=0, use_uva=args.use_uva, pin_prefetcher=True)
    # print('[PF] stat mem_used', torch.cuda.memory_allocated() /1024/1024)
    print('[PF] stop', time.time())
    print("warmup")
    model.train()
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print('training')
        print('[PF] start', time.time())
    # model.train()
    for epoch in range(args.n_epoch):
        print('[PF] bg epoch', time.time())
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            for bid, block in enumerate(blocks):
                # print('[PF] stat batch_{}_block_{}.src'.format(it, bid), block.number_of_src_nodes())
                # print('[PF] stat batch_{}_block_{}.dst'.format(it, bid), block.number_of_dst_nodes())
                # print('[PF] stat batch_{}_block_{}.edges'.format(it, bid), block.number_of_edges())
                print(f'[PF] stat block{bid}.src_nodes', block.number_of_src_nodes()) 
                print(f'[PF] stat block{bid}.dst_nodes', block.number_of_dst_nodes()) 
                print(f'[PF] stat block{bid}.edges', block.number_of_edges())
                # input()
            print('[PF] bg model_compution', time.time())
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x) # MC
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward() # MC
            opt.step()
            print('[PF] ed model_compution', time.time())
            
        print('[PF] ed epoch', time.time())
    print('[PF] ed end2end', time.time())
    print(f'[PF] stat max_mem_used', torch.cuda.max_memory_allocated()/1024/1024)
    print('[PF] stop', time.time())
if __name__ == '__main__':
    main()
