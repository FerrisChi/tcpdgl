import numpy as np
import scipy.sparse as sp
import argparse
import functools
import os
import subprocess
import dgl
import torch
from dgl.data.utils import save_feat

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def preprocess_feat(feats: sp.csr_matrix):
    rowsum = np.array(feats.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feats = r_mat_inv.dot(feats)
    # feats.data = np.round(feats.data, 4)
    return np.asarray(feats.todense())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, choices=['ogbn-products', 'ogbl-ddi', 'ogbl-ppa'])
    parser.add_argument('--data_path', type=str, default='/mnt/data2/chijj/data')
    argc = parser.parse_args()

    raw_path = os.path.join(argc.data_path, argc.graph_name.replace('-', '_'), 'raw', 'edge.csv')
    feat_path = os.path.join(argc.data_path, 'raw', 'node-feat.csv')
    label_path = os.path.join(argc.data_path, 'raw', 'node-label.csv')
    out_path = os.path.join(argc.data_path, argc.graph_name.replace('-', '_'))

    # sort graph and make it undirected    
    o2n = {}
    n2o = {}
    tot = 0
    mxid = 0
    edges = set()

    for line in open(raw_path):
        line = line.split(',')
        u, v = int(line[1]), int(line[0])
        if u==v:
            continue
        if o2n.get(u, -1) == -1:
            o2n[u] = tot
            n2o[tot] = u
            tot += 1
        if o2n.get(v, -1) == -1:
            o2n[v] = tot
            n2o[tot] = v
            tot += 1
        u, v = o2n[u], o2n[v]
        edges.add((u,v))
        edges.add((v,u))
    print(f'Graph has {tot} vertices, {len(edges)} edges.')

    def cmp(a, b):
        if a[0] == b[0]:
            if a[1] == b[1]:
                return 0
            if a[1] < b[1]:
                return -1
            return 1
        if a[0] < b[0]:
            return -1
        return 1
    edges = sorted(list(edges), key=functools.cmp_to_key(cmp))
    print(f'Graph sorted.')

    # save into un_sort
    out_graph_path = os.path.join(out_path, 'un_sort')
    with open(out_graph_path , 'w+') as f:
        for edge in edges:
            line = str(edge[0]) + ' ' + str(edge[1]) + '\n'
            f.write(line)
    print('Graph txt writed into {}.'.format(out_graph_path))

    # build dgl graph
    edges = list(map(list, zip(*edges)))
    graph = dgl.graph((edges[0], edges[1]))
    print(f'According DGL graph has {graph.num_nodes()} noddes and {graph.num_edges()} edges')

    # save feat for tcpdgl
    idx = [n2o[x] for x in range(tot)]
    frame_dict = {}
    if os.path.exists(feat_path):
        feat = np.loadtxt(argc.feat_path, dtype=np.float32, delimiter=',')[idx]
        feat = torch.tensor(feat).to(torch.float32)
        print(f'Load feat: {feat.shape} from {argc.feat_path}')
        frame_dict['feat'] = feat
    else:
        feat = None

    if os.path.exists(label_path):
        label = np.loadtxt(argc.label_path, dtype=np.int32, delimiter=',')[idx]
        label = torch.tensor(label).to(int)
        print(f'Load label {label.shape} from {argc.label_path}')
        frame_dict['label'] = label
    else:
        label = None

    if frame_dict:
        graph._node_frames = [dgl.frame.Frame(frame_dict)]
        graph._edge_frames = [dgl.frame.Frame()]
        out_feat_path = os.path.join(out_path, 'feat')
        save_feat(graph, out_feat_path)

    # execute compressor
    PROJECT_PATH = '/home/chijj/tcpdgl'
    PROCESSER_PATH=os.path.join(PROJECT_PATH, 'tools', 'block_compress_32_a')
    CP_NAME = 'tc1_k32_h2'
    TIMEOUT = 1200

    cp_cmd = f'{PROCESSER_PATH}/compressor {out_graph_path} {out_path}/{CP_NAME} 32 2'
    print(f'compress command: {cp_cmd}')

    p = subprocess.Popen(cp_cmd.split())
    try:
        print('Running in process', p.pid)
        p.wait(TIMEOUT)
    except subprocess.TimeoutExpired:
        print('Timed out - killing', p.pid)
        p.kill()
    print("Done")