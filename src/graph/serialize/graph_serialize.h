/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/serialize/graph_serialize.h
 * @brief Graph serialization header
 */
#ifndef DGL_GRAPH_SERIALIZE_GRAPH_SERIALIZE_H_
#define DGL_GRAPH_SERIALIZE_GRAPH_SERIALIZE_H_

#include <dgl/array.h>
#include <dgl/graph.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../c_api_common.h"
#include "dglgraph_data.h"
#include "heterograph_data.h"
#include "../sampling/ccgsample/nextdoor.h"

using dgl::ImmutableGraph;
using dgl::runtime::NDArray;
using namespace dgl::runtime;

namespace dgl {
namespace serialize {

enum GraphType : uint64_t {
  kMutableGraph = 0ull,
  kImmutableGraph = 1ull,
  kHeteroGraph = 2ull
};

constexpr uint64_t kDGLSerializeMagic = 0xDD2E4FF046B4A13F;

class StorageMetaDataObject : public runtime::Object {
 public:
  // For saving DGLGraph
  dgl_id_t num_graph;
  Value nodes_num_list;
  Value edges_num_list;
  Map<std::string, Value> labels_list;
  List<GraphData> graph_data;

  static constexpr const char *_type_key = "graph_serialize.StorageMetaData";

  void SetMetaData(
      dgl_id_t num_graph, std::vector<int64_t> nodes_num_list,
      std::vector<int64_t> edges_num_list,
      std::vector<NamedTensor> labels_list);

  void SetGraphData(std::vector<GraphData> gdata);

  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("num_graph", &num_graph);
    v->Visit("nodes_num_list", &nodes_num_list);
    v->Visit("edges_num_list", &edges_num_list);
    v->Visit("labels", &labels_list);
    v->Visit("graph_data", &graph_data);
  }

  DGL_DECLARE_OBJECT_TYPE_INFO(StorageMetaDataObject, runtime::Object);
};

class StorageMetaData : public runtime::ObjectRef {
 public:
  DGL_DEFINE_OBJECT_REF_METHODS(
      StorageMetaData, runtime::ObjectRef, StorageMetaDataObject);

  /** @brief create a new StorageMetaData reference */
  static StorageMetaData Create() {
    return StorageMetaData(std::make_shared<StorageMetaDataObject>());
  }
};

class CCGDataObject : public runtime::Object {
 public:
  // For saving CCG
  uint64_t n_nodes, ubl;
  std::vector<uint32_t> graph, offset;
  void* gpu_ccg;
  void* curand_states;
  dgl::tcpdgl::NextDoorData* nextDoorData;

  CCGDataObject() {
    this->n_nodes = 0;
    this->ubl = 0;
    this->graph = std::vector<uint32_t>();
    this->offset = std::vector<uint32_t>();
    this->gpu_ccg = nullptr;
    this->nextDoorData = nullptr;
  }

  void Save(dmlc::Stream *fs) const {
    // fs->Write(gptr);
    // fs->Write(node_tensors);
    // fs->Write(edge_tensors);
    // fs->Write(ntype_names);
    // fs->Write(etype_names);
  }

  bool Load(dmlc::Stream *fs) {
    // fs->Read(&gptr);
    // fs->Read(&node_tensors);
    // // std::cout<<"read edge_tensors"<<std::endl;
    // fs->Read(&edge_tensors);
    // // std::cout<<"read ntype_names"<<std::endl;
    // fs->Read(&ntype_names);
    // // std::cout<<"read etype_names"<<std::endl;
    // fs->Read(&etype_names);
    return true;
  }

  static constexpr const char *_type_key = "graph_serialize.CCGData";

  DGL_DECLARE_OBJECT_TYPE_INFO(CCGDataObject, runtime::Object);
};

class CCGData : public runtime::ObjectRef {
 public:
  DGL_DEFINE_OBJECT_REF_METHODS(CCGData, runtime::ObjectRef,
                                CCGDataObject);

  /*! \brief create a new CCGData reference */
  static CCGData Create() {
    return CCGData(std::make_shared<CCGDataObject>());
  }
};

CCGData LoadCCG(const std::string &file_path);

StorageMetaData LoadDGLGraphFiles(
    const std::string &filename, std::vector<dgl_id_t> idx_list, bool onlyMeta);

StorageMetaData LoadDGLGraphs(
    const std::string &filename, std::vector<dgl_id_t> idx_list, bool onlyMeta);

bool SaveDGLGraphs(
    std::string filename, List<GraphData> graph_data,
    std::vector<NamedTensor> labels_list);

std::vector<HeteroGraphData> LoadHeteroGraphs(
    const std::string &filename, std::vector<dgl_id_t> idx_list);

ImmutableGraphPtr ToImmutableGraph(GraphPtr g);

}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_SERIALIZE_GRAPH_SERIALIZE_H_
