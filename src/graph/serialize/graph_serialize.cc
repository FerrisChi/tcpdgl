/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/serialize/graph_serialize.cc
 * @brief Graph serialization implementation
 *
 * The storage structure is
 * {
 *   // MetaData Section
 *   uint64_t kDGLSerializeMagic
 *   uint64_t kVersion
 *   uint64_t GraphType
 *   ** Reserved Area till 4kB **
 *
 *   dgl_id_t num_graphs
 *   vector<dgl_id_t> graph_indices (start address of each graph)
 *   vector<dgl_id_t> nodes_num_list (list of number of nodes for each graph)
 *   vector<dgl_id_t> edges_num_list (list of number of edges for each graph)
 *
 *   vector<GraphData> graph_datas;
 *
 * }
 *
 * Storage of GraphData is
 * {
 *   // Everything uses in csr
 *   NDArray indptr
 *   NDArray indices
 *   NDArray edge_ids
 *   vector<pair<string, NDArray>> node_tensors;
 *   vector<pair<string, NDArray>> edge_tensors;
 * }
 *
 */
#include "graph_serialize.h"
#include "./dglstream.h"
#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../sampling/ccgsample/ccg_sample.h"

using namespace dgl::runtime;

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
using dgl::runtime::NDArray;
using dgl::serialize::GraphData;
using dgl::serialize::GraphDataObject;
using dmlc::SeekStream;
using dmlc::Stream;
using std::vector;

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, GraphDataObject, true);
}

namespace dgl {
namespace serialize {

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_MakeGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      GraphRef gptr = args[0];
      ImmutableGraphPtr imGPtr = ToImmutableGraph(gptr.sptr());
      Map<std::string, Value> node_tensors = args[1];
      Map<std::string, Value> edge_tensors = args[2];
      GraphData gd = GraphData::Create();
      gd->SetData(imGPtr, node_tensors, edge_tensors);
      *rv = gd;
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_SaveDGLGraphs_V0")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      List<GraphData> graph_data = args[1];
      Map<std::string, Value> labels = args[2];
      std::vector<NamedTensor> labels_list;
      for (auto kv : labels) {
        std::string name = kv.first;
        Value v = kv.second;
        NDArray ndarray = static_cast<NDArray>(v->data);
        labels_list.emplace_back(name, ndarray);
      }
      SaveDGLGraphs(filename, graph_data, labels_list);
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GDataGraphHandle")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      GraphData gdata = args[0];
      *rv = gdata->gptr;
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GDataNodeTensors")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      GraphData gdata = args[0];
      Map<std::string, Value> rvmap;
      for (auto kv : gdata->node_tensors) {
        rvmap.Set(kv.first, Value(MakeValue(kv.second)));
      }
      *rv = rvmap;
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GDataEdgeTensors")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      GraphData gdata = args[0];
      Map<std::string, Value> rvmap;
      for (auto kv : gdata->edge_tensors) {
        rvmap.Set(kv.first, Value(MakeValue(kv.second)));
      }
      *rv = rvmap;
    });

uint64_t GetFileVersion(const std::string &filename) {
  auto fs = std::unique_ptr<SeekStream>(
      SeekStream::CreateForRead(filename.c_str(), false));
  CHECK(fs) << "File " << filename << " not found";
  uint64_t magicNum, version;
  fs->Read(&magicNum);
  fs->Read(&version);
  CHECK_EQ(magicNum, kDGLSerializeMagic) << "Invalid DGL files";
  return version;
}

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GetFileVersion")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      *rv = static_cast<int64_t>(GetFileVersion(filename));
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_LoadGraphFiles_V1")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      List<Value> idxs = args[1];
      bool onlyMeta = args[2];
      auto idx_list = ListValueToVector<dgl_id_t>(idxs);
      *rv = LoadDGLGraphs(filename, idx_list, onlyMeta);
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_DGLAsHeteroGraph")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      GraphRef g = args[0];
      ImmutableGraphPtr ig =
          std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
      CHECK(ig) << "graph is not readonly";
      *rv = HeteroGraphRef(ig->AsHeteroGraph());
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_LoadGraphFiles_V2")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      List<Value> idxs = args[1];
      auto idx_list = ListValueToVector<dgl_id_t>(idxs);
      *rv = List<HeteroGraphData>(LoadHeteroGraphs(filename, idx_list));
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_LoadCCGFile")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string file_path = args[0];
    *rv = LoadCCG(file_path);
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GetVNumFromCCGData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    CCGData ccg_data = args[0];
    *rv = (int64_t)(ccg_data->n_nodes);
  });

  
DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_SaveFeat")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    HeteroGraphData gdata = args[1];
    auto fs = std::unique_ptr<DGLStream>(
    DGLStream::Create(filename.c_str(), "w", false, ANY_CODE));
    CHECK(fs->IsValid()) << "File name " << filename << " is not a valid name";

    fs->Write(gdata->node_tensors);
    fs->Write(gdata->edge_tensors);
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_LoadFeat")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    HeteroGraphData gdata = args[1];
    auto fs = std::unique_ptr<DGLStream>(
    DGLStream::Create(filename.c_str(), "r", false, ANY_CODE));
    CHECK(fs->IsValid()) << "File name " << filename << " is not a valid name";
    fs->Read(&(gdata->node_tensors));
    fs->Read(&(gdata->edge_tensors));
    *rv = gdata;
  });
  

}  // namespace serialize

DGL_REGISTER_GLOBAL("dataloading.neighbor_sampler._CAPI_AllocNextDoorData")
.set_body([](DGLArgs args, DGLRetValue *rv) {
  serialize::CCGData ccg_data = args[0];
  uint64_t seed_nodes_size = args[1];
  IdArray fanouts_arr = args[2];
  const int device_type = args[3];
  const int device_id = args[4];
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;

  const auto& _fanouts = fanouts_arr.ToVector<int64_t>();
  std::vector<int32_t> fanouts;
  for (auto f : _fanouts) {
    fanouts.push_back(static_cast<int32_t>(f));
  }
  if (fanouts[0] == -1) { // CCGMultiLayerFullNeighborSampler
    *rv = true;
    return;
  }

  ccg_data->nextDoorData = new NextDoorData;
  ccg_data->nextDoorData->setNumber(ccg_data->n_nodes, seed_nodes_size, fanouts);  
  allocNextDoorDataOnDevice(*(ccg_data->nextDoorData), ctx);
  setNextDoorData(ccg_data->nextDoorData, ccg_data->gpu_ccg, ccg_data->curand_states);

  // std::cout<<"Alloc NextDoorData done."<<std::endl;

  *rv=true;
});

}  // namespace dgl
