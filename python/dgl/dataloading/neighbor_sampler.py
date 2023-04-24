"""Data loading components for neighbor sampling"""
from ..base import EID, NID
from ..transforms import to_block
from .base import BlockSampler
from dgl.sampling.neighbor import ccg_sample_neighbors, ccg_sample_full_neighbors
from dgl.sampling.randomwalks import ccg_random_walk, random_walk

import time
import torch
import torch.distributed as dist
from ..utils import pflogger
from .. import backend as F, utils
from .._ffi.function import _init_api

_init_api("dgl.dataloading.neighbor_sampler")

class NeighborSampler(BlockSampler):
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    edge_dir : str, default ``'in'``
        Can be either ``'in' `` where the neighbors will be sampled according to
        incoming edges, or ``'out'`` otherwise, same as :func:`dgl.sampling.sample_neighbors`.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``.  The feature must be
        a scalar on each edge.

        This argument is mutually exclusive with :attr:`mask`.  If you want to
        specify both a mask and a probability, consider multiplying the probability
        with the mask instead.
    mask : str, optional
        If given, a neighbor could be picked only if the edge mask with the given
        name in ``g.edata`` is True.  The data must be boolean on each edge.

        This argument is mutually exclusive with :attr:`prob`.  If you want to
        specify both a mask and a probability, consider multiplying the probability
        with the mask instead.
    replace : bool, default False
        Whether to sample with replacement
    prefetch_node_feats : list[str] or dict[ntype, list[str]], optional
        The source node data to prefetch for the first MFG, corresponding to the
        input node features necessary for the first GNN layer.
    prefetch_labels : list[str] or dict[ntype, list[str]], optional
        The destination node data to prefetch for the last MFG, corresponding to
        the node labels of the minibatch.
    prefetch_edge_feats : list[str] or dict[etype, list[str]], optional
        The edge data names to prefetch for all the MFGs, corresponding to the
        edge features necessary for all GNN layers.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the
        minibatch of seed nodes.

    Examples
    --------
    **Node classification**

    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15])
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.NeighborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)

    If you would like non-uniform neighbor sampling:

    >>> g.edata['p'] = torch.rand(g.num_edges())   # any non-negative 1D vector works
    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15], prob='p')

    Or sampling on edge masks:

    >>> g.edata['mask'] = torch.rand(g.num_edges()) < 0.2   # any 1D boolean mask works
    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15], prob='mask')

    **Edge classification and link prediction**

    This class can also work for edge classification and link prediction together
    with :func:`as_edge_prediction_sampler`.

    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15])
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_eid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)

    See the documentation :func:`as_edge_prediction_sampler` for more details.

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks


MultiLayerNeighborSampler = NeighborSampler


class MultiLayerFullNeighborSampler(NeighborSampler):
    """Sampler that builds computational dependency of node representations by taking messages
    from all neighbors for multilayer GNN.

    This sampler will make every node gather messages from every single neighbor per edge type.

    Parameters
    ----------
    num_layers : int
        The number of GNN layers to sample.
    kwargs :
        Passed to :class:`dgl.dataloading.NeighborSampler`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors for the first,
    second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(self, num_layers, **kwargs):
        super().__init__([-1] * num_layers, **kwargs)

class CCGNeighborSampler(BlockSampler):
    def __init__(self, fanouts,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts

    def init_nextdoor(self, ccg, batch_size, fanout, ctx):
        if isinstance(ctx, torch.device):
            ctx = utils.to_dgl_context(ctx)
        _CAPI_AllocNextDoorData(ccg.ccg_data, batch_size, fanout, ctx.device_type, ctx.device_id)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # g is ccg graph here
        if not dist.is_initialized() or dist.get_rank() == 0:
            pflogger.info('bg sampler.sample_blocks %f', time.time())
        seed_nodes, output_nodes, sample_blocks = g.ccg_sample_neighbors(seed_nodes, self.fanouts, copy_ndata=False, copy_edata=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            pflogger.info('ed sampler.sample_blocks %f', time.time())
        return seed_nodes, output_nodes, sample_blocks

class CCGMultiLayerFullNeighborSampler(BlockSampler):
    def __init__(self, num_layers, fanouts=[-1],
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.num_layers = num_layers
        self.fanouts = [-1] * num_layers

    def init_nextdoor(self, ccg, batch_size, fanout, ctx):
        if isinstance(ctx, torch.device):
            ctx = utils.to_dgl_context(ctx)
        _CAPI_AllocNextDoorData(ccg.ccg_data, batch_size, fanout, ctx.device_type, ctx.device_id)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # g is ccg graph here
        if not dist.is_initialized() or dist.get_rank() == 0:
            pflogger.info('bg sampler.sample_blocks %f', time.time())
        seed_nodes, output_nodes, sample_blocks = g.ccg_sample_full_neighbors(seed_nodes, self.num_layers, copy_ndata=False, copy_edata=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            pflogger.info('ed sampler.sample_blocks %f', time.time())
        return seed_nodes, output_nodes, sample_blocks
    
class DeepwalkSampler(object):
    def __init__(self, G, seeds, batch_size, walk_length, ccg_sample, device=None):
        """random walk sampler

        Parameter
        ---------
        G dgl.Graph : the input graph
        seeds torch.LongTensor : starting nodes
        walk_length int : walk length
        """
        self.G = G
        self.seeds = seeds
        self.batch_size = batch_size
        self.walk_length = walk_length
        self.ccg_sample = ccg_sample
        if self.ccg_sample:
            self.seeds = self.seeds.to('cuda:0')
            self.init_nextdoor(batch_size, F.to_dgl_nd(torch.tensor([walk_length])), F.context(self.seeds))
        elif device:
            self.seeds = self.seeds.to(device)
    
    def init_nextdoor(self, batch_size, fanout, ctx):
        if isinstance(ctx, torch.device):
            ctx = utils.to_dgl_context(ctx)
        _CAPI_AllocNextDoorData(self.G.ccg.ccg_data, batch_size, fanout, ctx.device_type, ctx.device_id)

    def sample(self, seeds):
        if not dist.is_initialized() or dist.get_rank() == 0:
            pflogger.info('bg sampler.sample_blocks %f', time.time())
        if self.ccg_sample:
            walks = ccg_random_walk(self.G, seeds, length=self.walk_length)[0]
        else:
            walks = random_walk(self.G, seeds, length=self.walk_length - 1)[0]
        if not dist.is_initialized() or dist.get_rank() == 0:
            pflogger.info('ed sampler.sample_blocks %f', time.time())
        return walks
