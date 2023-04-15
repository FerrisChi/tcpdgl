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
import sys
import time
from dgl.data import CoraGraphDataset, RedditDataset
from dgl.data.utils import save_graphs, load_graphs
from ogb.nodeproppred import DglNodePropPredDataset

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

        # all property should be assigned before pin
        g.ndata['h'] = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device)
            for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = blocks[0].srcdata['h']
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to(buffer_device)
            g.ndata['h'] = y
        return y

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

# @profile
def main():
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # ogbn-products Reddit
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
    args = parser.parse_args()
    print(args)

    torch.cuda.reset_peak_memory_stats()

    device = 'cuda'
    print('[PF] start', time.time())
    print('[PF] bg load_graph', time.time())
    graph, train_idx, valid_idx, test_idx, num_classes = LoadDataset(args.graph_name, args.n_feat)
    # save_graphs(f'/home/chijj/data/{args.graph_name}/dglgraph.bin', [graph])
    # print('done')
    # input()
    print('[PF] ed load_graph', time.time())
    
    print('[PF] bg end2end', time.time())
    print('[PF] bg to_dvs', time.time())
    if args.use_uva == False:
        graph = graph.to(device, move_feat=not args.pin)
    print('[PF] ed to_dvs', time.time())
    
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)
    

    model = SAGE(graph.ndata['feat'].shape[1], 256, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    fanout = [20, 5]
    print('[PF] bg pre_sampler', time.time())
    if args.sampler=='sage':
        sampler = dgl.dataloading.NeighborSampler(
                fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'], replace=True)
    elif args.sampler=='labor':
        sampler = dgl.dataloading.LaborSampler(
            fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'], importance_sampling=-2)
    train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler, device=device, batch_size=204800, shuffle=True,
            drop_last=False, num_workers=0, pin_prefetcher=args.pin, use_uva=args.use_uva)
    print('[PF] ed pre_sampler', time.time())
    # print('torch.cuda.memory_allocated', torch.cuda.memory_allocated() /1024/1024)
    # input()
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
    for epoch in range(args.n_epoch):
        model.train()
        print('[PF] bg epoch', time.time())
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            for bid, block in enumerate(blocks):
                # print('[PF] stat batch_{}_block_{}.src'.format(it, bid), block.number_of_src_nodes())
                # print('[PF] stat batch_{}_block_{}.dst'.format(it, bid), block.number_of_dst_nodes())
                # print('[PF] stat batch_{}_block_{}.edges'.format(it, bid), block.number_of_edges())
                print(f'[PF] stat block_{bid}.srcnodes {block.number_of_src_nodes()}')
                print(f'[PF] stat block_{bid}.dstnodes {block.number_of_dst_nodes()}')
                print(f'[PF] stat block_{bid}.edges {block.number_of_edges()}')
                # input()
            print('[PF] bg model_compution', time.time())
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print('[PF] ed model_compution', time.time())
            
        print('[PF] ed epoch', time.time())
    print('[PF] ed end2end', time.time())
    print(f'[PF] stat max_mem_used', torch.cuda.max_memory_allocated()/1024/1024)
    print('[PF] stop', time.time())
    print('end')

if __name__ == '__main__':
    main()