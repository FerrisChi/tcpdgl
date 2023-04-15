"""Classes for CCG."""
#pylint: disable= too-many-lines

import copy
from ._ffi.function import _init_api
from . import utils

__all__ = ["CCG"]

class CCG(object):
    def __init__(self, _v_num, _ccg_data):
        self.v_num = _v_num
        self.ccg_data = _ccg_data
        self.ctx = None
        return

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

    def pin_memory_(self):
        return _CAPI_CCGPinMemory_(self.ccg_data)

    def unpin_memory_(self):
        return _CAPI_CCGUnpinMemory_(self.ccg_data)

_init_api("dgl.ccg")