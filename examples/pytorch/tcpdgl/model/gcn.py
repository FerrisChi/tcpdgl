import dgl
import dgl.nn as dglnn
from dgl.multiprocessing import shared_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchmetrics.functional as MF
import tqdm

__all__ = ['SAGE', 'GCN']

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

    def ccg_inference(self, g, device, batch_size, num_workers, buffer_device=None):
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

    def dgl_inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.

        # all property should be assigned before pin
        g.ndata['h'] = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.DataLoader(
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
    
    def dgl_dist_inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = dgl.dataloading.DataLoader(
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
    
class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=F.relu))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


