import argparse
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
from dgl.utils import pflogger
import time

from model.gcn import SAGE
from load_dataset import LoadDataset

# @profile
def main():
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # ogbn-products Reddit
    parser.add_argument("-g","--graph_name", type=str, default='cora', help="dataset name")
    parser.add_argument("--use_uva", action='store_true', default=False, help="use unified virtual space")
    parser.add_argument("--pin", action='store_true', default=False, help="pin graph features")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("-e", "--n_epoch", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("-f", "--n_feat", type=int, default=10,
                        help="number of features")
    parser.add_argument("-s", "--sampler",type=str, default="sage", choices=["sage", "labor"], help="graph sampler")
    parser.add_argument("--cnt", type=int, default=-1, help="number of experiments")
    args = parser.parse_args()
    print(args)

    torch.cuda.reset_peak_memory_stats()

    device = 'cuda'
    pflogger.info('start %f', time.time())
    pflogger.info('bg load_graph %f', time.time())
    graph, train_idx, valid_idx, test_idx, num_classes = LoadDataset(args.graph_name, args.n_feat)
    # save_graphs(f'/home/chijj/data/{args.graph_name}/dglgraph.bin', [graph])
    # print('done')
    # input()
    pflogger.info('ed load_graph %f', time.time())
    
    pflogger.info('bg end2end %f', time.time())
    pflogger.info('bg to_dvs %f', time.time())
    if args.use_uva == False:
        graph = graph.to(device, move_feat=not args.pin)
    pflogger.info('ed to_dvs %f', time.time())

    
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)
    

    model = SAGE(graph.ndata['feat'].shape[1], 256, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    fanout = [20, 5]
    if args.sampler=='sage':
        sampler = dgl.dataloading.NeighborSampler(
                fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'], replace=True)
    elif args.sampler=='labor':
        sampler = dgl.dataloading.LaborSampler(
            fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'], importance_sampling=-2)
    train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler, device=device, batch_size=1024, shuffle=True, # 204800
            drop_last=False, num_workers=0, pin_prefetcher=args.pin, use_uva=args.use_uva)
    # print('torch.cuda.memory_allocated', torch.cuda.memory_allocated() /1024/1024)
    pflogger.info('stop %f', time.time())
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
    pflogger.info('start %f', time.time())
    for epoch in range(args.n_epoch):
        model.train()
        pflogger.info('bg epoch %f', time.time())
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            for bid, block in enumerate(blocks):
                # print('[PF] stat batch_{}_block_{}.src'.format(it, bid), block.number_of_src_nodes())
                # print('[PF] stat batch_{}_block_{}.dst'.format(it, bid), block.number_of_dst_nodes())
                # print('[PF] stat batch_{}_block_{}.edges'.format(it, bid), block.number_of_edges())
                pflogger.info(f'stat block{bid}.src_nodes {block.number_of_src_nodes()}')
                pflogger.info(f'stat block{bid}.dst_nodes {block.number_of_dst_nodes()}')
                pflogger.info(f'stat block{bid}.edges {block.number_of_edges()}')
                # input()
            pflogger.info('bg model_compution %f', time.time())
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pflogger.info('ed model_compution %f', time.time())
            
        pflogger.info('ed epoch %f', time.time())
    pflogger.info('ed end2end %f', time.time())
    pflogger.info('[PF] stat max_mem_used %d', torch.cuda.max_memory_allocated()/1024/1024)
    pflogger.info('stop %f', time.time())
    print('end')

if __name__ == '__main__':
    main()