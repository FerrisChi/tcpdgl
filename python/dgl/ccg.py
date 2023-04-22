"""Classes for CCG."""
#pylint: disable= too-many-lines

import copy
import numbers
import torch

from ._ffi.function import _init_api
from ._ffi.object import ObjectBase
import traceback
from .base import ALL, is_all, DGLError
from . import backend as F, utils

__all__ = ["CCG"]

class CCG(object):
    def __init__(self, _v_num, _ccg_data):
        self.v_num = _v_num
        self.ccg_data = _ccg_data
        self.ctx = None
        return

    def __getstate__(self):
        # BUG
        input("__getstate__ CCG")
        print(type(self.v_num), type(self.ccg_data))
        return self.v_num, self.ccg_data
    
    def __setstate__(self, state):
        # BUG
        self.v_num, self.ccg_data = state
        self.ctx = None

    def to(self, ctx, **kwargs):  # pylint: disable=invalid-name
        # thisctx = utils.to_dgl_context(device)
        self.ccg_data = _CAPI_TransferCCGTo_(self.ccg_data, ctx.device_type, ctx.device_id)
        self.ctx = ctx
        return self
        # TODO(chijj): whether to copy ccg_data?
        ret = copy.copy(self)

        # 1. Copy graph structure
        # ret._graph = self._graph.copy_to(utils.to_dgl_context(device))

        # 2. Copy features
        # TODO(minjie): handle initializer
        new_nframes = []
        for nframe in self._node_frames:
            new_nframes.append(nframe.to(device, **kwargs))
        ret._node_frames = new_nframes

        new_eframes = []
        for eframe in self._edge_frames:
            new_eframes.append(eframe.to(device, **kwargs))
        ret._edge_frames = new_eframes

        # 2. Copy misc info
        ret._batch_num_nodes = None
        ret._batch_num_edges = None

        return ret
    
    def num_nodes(self):
        return self.v_num

    def num_edges(self):
        return _CAPI_CCGNumEdges(self.ccg_data)

    def out_degrees(self, u=ALL, etype=None):
        # srctype = self.to_canonical_etype(etype)[0]
        # etid = self.get_etype_id(etype)
        if is_all(u):
            u = torch.arange(self.v_num, device=self.device)
        u_tensor = utils.prepare_tensor(self, u, "u")
        # if F.as_scalar(
        #     F.sum(self.has_nodes(u_tensor, ntype=srctype), dim=0)
        # ) != len(u_tensor):
        #     raise DGLError("u contains invalid node IDs")
        # deg = self._graph.out_degrees(etid, utils.prepare_tensor(self, u, "u"))
        v_array = u_tensor.todgltensor()
        deg = utils.toindex(_CAPI_CCGOutDegrees(self.ccg_data, v_array))
        if isinstance(u, numbers.Integral):
            return F.as_scalar(deg)
        else:
            return deg

    def pin_memory_(self):
        return _CAPI_CCGPinMemory_(self.ccg_data)

    def unpin_memory_(self):
        return _CAPI_CCGUnpinMemory_(self.ccg_data)

_init_api("dgl.ccg")