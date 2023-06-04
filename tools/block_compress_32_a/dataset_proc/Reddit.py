import numpy as np
import scipy.sparse as sp
import argparse
import functools
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, default='Reddit')
    parser.add_argument('--graph_path', type=str, default='/home/chijj/data/Reddit/raw/reddit_graph.npz')
    parser.add_argument('--data_path', type=str, default='/home/chijj/data/Reddit/raw/reddit_data.npz')
    parser.add_argument('--num_feat', type=int, default=1433)
    parser.add_argument('--output_dir_path', type=str, default='/home/chijj/data/Reddit/')
    parser.add_argument('--directed', type=bool, default=False)
    argc = parser.parse_args()
    
    o2n = {}
    tot = 0
    edges = set()
    graph = np.load(argc.graph_path)

    for (u,v) in zip(graph['row'], graph['col']):
        if u==v:
            continue
        # if o2n.get(u, -1) == -1:
        #     o2n[u] = tot
        #     tot += 1
        # if o2n.get(v, -1) == -1:
        #     o2n[v] = tot
        #     tot += 1
        # u, v = o2n[u], o2n[v]
        edges.add((u,v))
        edges.add((v,u))

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

    with open(os.path.join(argc.output_dir_path, argc.graph_name + '_sorted') , 'w+') as f:
        for edge in edges:
            line = str(edge[0]) + ' ' + str(edge[1]) + '\n'
            f.write(line)

    print(f'Graph has {tot} vertices, {len(edges)} edges.')
