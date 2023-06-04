import numpy as np
import scipy.sparse as sp
import argparse
import functools
import os


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
    parser.add_argument('--graph_path', type=str, default='/home/chijj/data/cora/cora.cites')
    parser.add_argument('--feat_path', type=str, default='/home/chijj/data/cora/cora.content')
    parser.add_argument('--num_feat', type=int, default=1433)
    parser.add_argument('--output_dir_path', type=str, default='/home/chijj/data/cora')
    argc = parser.parse_args()
    
    o2n = {}
    n2o = {}
    tot = 0
    edges = set()

    for line in open(argc.graph_path):
        line = line.split('\t')
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

    graph_name = argc.graph_path.split('/')[-1]
    with open(os.path.join(argc.output_dir_path, graph_name + '_sorted') , 'w+') as f:
        for edge in edges:
            line = str(edge[0]) + ' ' + str(edge[1]) + '\n'
            f.write(line)

    print(f'Graph has {tot} vertices, {len(edges)} edges.')

    
    idx_feats_labels = np.genfromtxt(argc.feat_path, dtype=str)
    oidx = [o2n[int(id)] for id in idx_feats_labels[:, 0]]
    oidx = np.argsort(oidx)
    labels = idx_feats_labels[:, -1][oidx]
    onehot_labels = encode_onehot(labels)
    labels = np.argmax(onehot_labels, 1)
    feats = sp.csr_matrix(idx_feats_labels[:, 1:-1][oidx], dtype=np.float32)
    feats = preprocess_feat(feats)

    # labels = np.zeros(tot)
    # cate_label = {}
    # for line in open(argc.feat_path):
    #     line = line.split('\t')
    #     id, label = int(line[0]), line[-1]
    #     feat = list(map(int, line[1:-1]))
    #     cate_label.setdefault(label, len(cate_label)+1)
    #     labels[o2n[id]] = cate_label[label]
    #     feats[o2n[id]] = feat
    
    np.save(os.path.join(argc.output_dir_path, 'feat'), feats)
    np.save(os.path.join(argc.output_dir_path, 'label'), labels)
    print(f'feat: {feats.shape}, label: {labels.shape} writed into {argc.output_dir_path}')
