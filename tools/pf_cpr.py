import numpy as np
import scipy.sparse as sp
import argparse
import os
import subprocess

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, choices=['ogbn-products', 'ogbl-ddi', 'ogbl-ppa'])
    parser.add_argument('--data_path', type=str, default='/mnt/data2/chijj/data')
    argc = parser.parse_args()

    raw_path = os.path.join(argc.data_path, argc.graph_name.replace('-', '_'), 'raw', 'edge.csv')
    feat_path = os.path.join(argc.data_path, 'raw', 'node-feat.csv')
    label_path = os.path.join(argc.data_path, 'raw', 'node-label.csv')
    out_path = os.path.join(argc.data_path, argc.graph_name.replace('-', '_'))

    # execute compressor
    PROJECT_PATH = '/home/chijj/tcpdgl'
    PROCESSER_PATH=os.path.join(PROJECT_PATH, 'tools')
    PROCESSER = ['block_compress_32_a', 'GraSS-GT', 'GraSS-LE']
    CP_NAME = 'tc1_k32_h2'
    DATASET_PATH = '/mnt/data2/chijj/data'
    DATASET = ['cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1']
    TIMEOUT = 1200

    for pross in PROCESSER:
        for d in DATASET:

            pross_path = os.join(PROCESSER_PATH, pross, 'compressor')
            input_path = os.join(DATASET_PATH, d, 'un_sort')
            
            cp_cmd = f'{pross_path} {out_graph_path} {out_path}/{CP_NAME} 32 2'
        print(f'compress command: {cp_cmd}')

        p = subprocess.Popen(cp_cmd.split())
        try:
            print('Running in process', p.pid)
            p.wait(TIMEOUT)
        except subprocess.TimeoutExpired:
            print('Timed out - killing', p.pid)
            p.kill()
        print("Done")