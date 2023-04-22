import dgl
from dgl.data.utils import load_ccg_for, load_feat
from dgl.data import DGLDataset
from dgl.data import CoraGraphDataset, RedditDataset
from dgl.data.utils import save_graphs, load_graphs
from dgl.utils import pflogger
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset

import torch
import os
import numpy as np
import random
import pandas
import time
import argparse

__all__ = ['CCGDataset', 'LoadDataset']

class CCGDataset(DGLDataset):
    def __init__(self, _graph_name, _ccg_name, _graph_path, _feat_path=None, _data_path=None, _num_classes=50, _feat_dim=10, _feat_lr=-10.0, _feat_rr=10.0):
        self.graph_name = _graph_name
        self.file_path = _graph_path
        self.num_classes = _num_classes
        self.feat_path = _feat_path
        self.data_path = _data_path
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
            type_path = os.path.join(self.data_path, 'Reddit', 'node_types.npy')
            node_types = np.load(type_path)
            train_idx = torch.tensor(np.nonzero(node_types==1)[0])
            valid_idx = torch.tensor(np.nonzero(node_types==2)[0])
            test_idx = torch.tensor(np.nonzero(node_types==3)[0])
        else:
            train_idx = torch.tensor(idx[:int(self.graph.ccg.v_num * train_pc)])
            test_idx = torch.tensor(idx[int(self.graph.ccg.v_num * train_pc) : int(self.graph.ccg.v_num * (train_pc + test_pc))])
            valid_idx = torch.tensor(idx[int(self.graph.ccg.v_num * (train_pc + test_pc)) : self.graph.ccg.v_num])
        return train_idx, test_idx, valid_idx

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
        pflogger.info('bg split %f', time.time())
        idx = [i for i in range(graph.num_nodes())]
        random.shuffle(idx)
        train_idx = torch.tensor(idx[:int(graph.num_nodes() * train_pc)])
        test_idx = torch.tensor(idx[int(graph.num_nodes() * train_pc) : int(graph.num_nodes() * (train_pc + test_pc))])
        valid_idx = torch.tensor(idx[int(graph.num_nodes() * (train_pc + test_pc)) : graph.num_nodes()])
        pflogger.info('ed split %f', time.time())
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

def load_from_ogbl_with_name(name, load_ccg=False, data_path = '/mnt/data2/chijj/data', save_dgl = False):
    choices = ["ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation"]
    assert name in choices, "name must be selected from " + str(choices)

    if load_ccg:
        graph = dgl.graph(([],[]))
        file_path = os.path.join(data_path, name.replace('-','_'), "tc1_k32_h2")
        ck_feat_path = os.path.join(data_path, name.replace('-','_'), "feat")
        feat_path = ck_feat_path if os.path.isfile(ck_feat_path) else None
        print(file_path, feat_path)
        graph = load_ccg_for(graph, file_path, feat_path)
        # graph = graph.to('cuda:0')
        # print(graph.ccg.num_nodes(), graph.ccg.num_edges())
    else:
        cache_path = os.path.join(data_path, name.replace('-', '_'), 'dglgraph.bin')
        if os.path.exists(cache_path):
                print('Load from Cached file ', cache_path)
                graph, label_dict = load_graphs(cache_path)
                graph = graph[0]
        else:
            dataset = DglLinkPropPredDataset(name, data_path)
            graph = dataset[0]
        print(graph.num_nodes(), graph.num_edges())

        if save_dgl:
            print(f'Saving {name} into {cache_path}')
            save_graphs(cache_path, [graph])
            print('Saved.')
    return graph



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        choices=["ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation"],
        default="ogbl-ddi",
        help="name of datasets by ogb",
    )
    parser.add_argument(
        '--save_txt',
        action='store_true',
        default=False,
        help='Whether to save graph into .txt'
    )
    parser.add_argument(
        '--save_dgl',
        action='store_true',
        default=False,
        help='Whether to save graph into dglgraph.bin'
    )
    parser.add_argument(
        '--load_ccg',
        action='store_true',
        default=False,
        help='Whether to load ccg graph'
    )
    args = parser.parse_args()
    print(args)

    data_path = '/mnt/data2/chijj/data'

    print("loading graph... it might take some time")
    name = args.name
    g = load_from_ogbl_with_name(name, args.load_ccg, data_path, args.save_dgl)

    try:
        w = g.edata["edge_weight"]
        weighted = True
    except:
        weighted = False
        
    print("writing...")
    if weighted:
        input('weight')

    if args.save_txt:
        start_time = time.time()
        edge_num = g.edges()[0].shape[0]
        src = list(g.edges()[0])
        tgt = list(g.edges()[1])
        if weighted:
            weight = list(g.edata["edge_weight"])
        with open(os.path.join(data_path, name.replace('-', '_'), name.replace('-', '_') + ".txt"), "w") as f:
            for i in range(edge_num):
                if weighted:
                    f.write(
                        str(src[i].item())
                        + " "
                        + str(tgt[i].item())
                        + " "
                        + str(weight[i].item())
                        + "\n"
                    )
                else:
                    # " 1\n"
                    f.write(
                        str(src[i].item()) + " " + str(tgt[i].item()) + "\n" 
                    )
        print("writing used time: %d s" % int(time.time() - start_time))
