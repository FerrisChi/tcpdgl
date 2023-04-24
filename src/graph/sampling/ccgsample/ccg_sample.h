/**
 *  Copyright (c) 2023 by Contributors
 * @file graph/sampling/ccgsample/ccg_sample.h
 * @brief CCG DGL sampler
 */

#ifndef CCG_SAMPLE_H_
#define CCG_SAMPLE_H_

#pragma once

#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <dgl/runtime/device_api.h>

#include <dgl/sampling/rand_num_gen.cuh>
#include <dgl/aten/coo.h>
#include <dgl/runtime/c_runtime_api.h>
#include "./ccg.cuh"
#include "./nextdoor.h"

namespace dgl {

namespace tcpdgl
{

const size_t N_THREADS = 256;
const bool useGridKernel = true;
const bool useSubWarpKernel = false;
const bool useThreadBlockKernel = true;
const bool combineTwoSampleStores = false;
const bool enableLoadBalancing = false;

// extern variables for sampling
// graph info
extern VertexID_t n_nodes;
extern std::vector<int64_t> fanouts;
extern int64_t *d_fanouts;
extern int64_t trace_length;

// sample result
extern std::vector<dgl::aten::COOMatrix> vecCOO;
extern IdArray traces;
extern VertexID_t *out_trace;

// step result
extern dgl::NDArray picked_row;
extern dgl::NDArray picked_col;
extern dgl::NDArray picked_idx;
extern VertexID_t *out_rows;
extern VertexID_t *out_cols;
extern EdgePos_t *out_idxs;
extern int64_t spl_len;


struct FullLayersData {
  dgl::NDArray d_picked_row, d_picked_col, d_picked_idx;
  VertexID_t* d_out_rows;
  VertexID_t* d_out_cols;
  EdgePos_t* d_out_idxs;
  
  EdgePos_t* d_deg;
  EdgePos_t* d_degoffset;

  FullLayersData() {}

  void setSampleNum(int samples_num, int device = 0) {
    const DGLContext& ctx = DGLContext{kDGLCUDA, device};
    d_picked_row = dgl::aten::NewIdArray(samples_num, ctx, sizeof(VertexID_t) * 8);
    d_picked_col = dgl::aten::NewIdArray(samples_num, ctx, sizeof(VertexID_t) * 8);
    d_picked_idx = dgl::aten::NewIdArray(samples_num, ctx, sizeof(EdgePos_t) * 8);
    d_out_rows = static_cast<VertexID_t*>(d_picked_row->data);
    d_out_cols = static_cast<VertexID_t*>(d_picked_col->data);
    d_out_idxs = static_cast<VertexID_t*>(d_picked_idx->data);
  }
};
  
struct BCGVertex {

  VertexID_t vertex;
  VertexID_t outd;
  int h, k;
  
  // layer_bit[i] = bits of a representation of neighbor vertex in layer i
  int layer_bit[5];

  // in first layer, i>=ppos -> neighbor vertex > now vertex
  int ppos;

  Offset_t offset;
  const Graph_t* graph;
  
  const int T_h = 2;
  const int BIT_h = 1; // bit number of (h - 1)
  const int T_k = 32;
  const int T_D = 10; // 度数阈值

  // __host__ __device__
  // Graph_t get_32b(Offset_t bit_offset);

  // __host__ __device__
  // int get_vlc_bit(Offset_t &bit_offset);

  __host__ __device__
  Graph_t decode(Offset_t& bit_offset, int vlc_bit) {
    auto ret = get_32b(bit_offset) >> (32 - vlc_bit);
    bit_offset += vlc_bit;
    return ret;
  }

  __host__ __device__
  Graph_t decode_left(Offset_t bit_offset, int vlc_bit) {
    auto ret = get_32b(bit_offset) >> (32 - vlc_bit);
    // bit_offset += vlc_bit;
    return ret;
  }

  __host__ __device__
  VertexID_t decode_vlc(Offset_t &bit_offset) {
    auto vlc_bit = get_vlc_bit(bit_offset);
    return decode(bit_offset, vlc_bit) - 1;
  }

  __host__ __device__
  void get_kh(Offset_t &bit_offset) {
      unsigned int tmp = decode(bit_offset, 8);
      k = (tmp >> 3) + 1;
      h = (tmp & 0b111) + 1;
      return;
  }

  __host__ __device__
  void get_layer_bit(Offset_t &bit_offset) {
      uint32_t tmp = decode(bit_offset, h * 5);
      // for (int i = h - 1; i >= 0; --i, tmp >>= 5) {
      //     layer_bit[i] = (tmp & 31) + 1;
      // }
      if (h == 3) {
          layer_bit[2] = (tmp & 31) + 1;
          tmp >>= 5;
      }
      if (h >= 2) {
          layer_bit[1] = (tmp & 31) + 1;
          tmp >>= 5;
      }
      layer_bit[0] = (tmp & 31) + 1;
      return;
  }

  // __device__
  // void get_ppos(Offset_t &bit_offset) {
  //     ppos = decode(bit_offset, 32 - __clz(k));
  //     return;
  // }
  __host__ __device__
  Graph_t
  get_32b(Offset_t bit_offset)
  {
    Offset_t chunk = bit_offset / GRAPH_LEN;
    Graph_t h_bit = graph[chunk];
    Graph_t l_bit = graph[chunk + 1];
    Offset_t chunk_offset = bit_offset % GRAPH_LEN;
    #ifdef __CUDA_ARCH__
      // Device version of the function
    return __funnelshift_l(l_bit, h_bit, chunk_offset);
    #else
    // Host version of the function
    return 0;
    #endif
  }

  __host__ __device__ int get_vlc_bit(Offset_t &bit_offset)
  {
    auto tmp = get_32b(bit_offset);
    #ifdef __CUDA_ARCH__
    auto x = __clz(tmp);
    bit_offset += x;
    return x + 1;
    #else
    auto x = __builtin_clz(tmp);
    bit_offset += x;
    return x + 1;
    #endif
  }

  __host__ __device__
  BCGVertex(VertexID_t _vertex, const Graph_t *_graph, const Offset_t _offset) : vertex(_vertex), graph(_graph), offset(_offset)
  {
    // if (vertex == 0) printf("init0 os=%ld ", offset);
    outd = decode_vlc(offset);
    // if (vertex == 0) printf("os=%ld ", offset);
    // get_kh(offset);
    k = decode_vlc(offset);
    // if (vertex == 0) printf("os=%ld ", offset);
    h = decode(offset, BIT_h) + 1;
    // if (vertex == 0) printf("os=%ld ", offset); 
    #ifdef __CUDA_ARCH__
    ppos = decode(offset, 32 - __clz(k));
    #else
    ppos = decode(offset, 32 - __builtin_clz(k));
    #endif
    // if (vertex == 0) printf("os=%ld ", offset);
    get_layer_bit(offset);
    // if (vertex == 0) printf("init0 d=%d k=%d h=%d p=%d os=%ld\n", outd, k, h, ppos, offset);

    Offset_t chunk = offset / GRAPH_LEN;
    graph += chunk;
    offset -= chunk * GRAPH_LEN;
    // if (vertex == 0) printf("init0 os=%ld\n", offset);

    return;
  }

  __host__ __device__
  BCGVertex(const Graph_t *_graph, const Offset_t _offset) : graph(_graph), offset(_offset)
  {
    outd = decode_vlc(offset);
    return;
  }

  // __host__ __device__
  // BCGVertex(VertexID_t _vertex, const Graph_t* _graph, const Offset_t _offset);

  // __host__ __device__
  // BCGVertex(const Graph_t* _graph, const Offset_t _offset);

  __host__ __device__
  BCGVertex() {}

  __device__
  VertexID_t get_vertex2(VertexID_t neighbor) {
    // if (vertex == 0) {
    //   printf("in_g2 %d %d\n", vertex, outd);
    // }
    auto l2_id = neighbor - k;
    auto l1_id = (l2_id >= 0 ? l2_id / k : neighbor);
    VertexID_t l1_v = decode_left(offset + l1_id * layer_bit[0], layer_bit[0]) + 1;
    VertexID_t l2_v = 0;
    if (l2_id >= 0) l2_v = (l2_id % k <= ((k - 1) >> 1) ? -1 : 1) * (decode_left(offset + k * layer_bit[0] + l2_id * layer_bit[1], layer_bit[1]) + 1);
    return vertex + (l1_id < ppos ? -1 : 1) * l1_v + l2_v;
  }

  __device__
  VertexID_t get_vertex3(VertexID_t neighbor) {
    short this_h = 0;
    this_h += (neighbor >= k) + (neighbor >= k * k + k);

    neighbor -= (neighbor >= k) * k;
    VertexID_t ret = vertex;
    if (this_h >= 2) { 
        neighbor -= k * k;
        ret += (neighbor % k <= ((k - 1) >> 1) ? -1 : 1) * (decode_left(offset + k * layer_bit[0] + k * k * layer_bit[1] + neighbor * layer_bit[2], layer_bit[2]) + 1);
        neighbor /= k;
    }
    if (this_h >= 1) {
        ret += (neighbor % k <= ((k - 1) >> 1) ? -1 : 1) * (decode_left(offset + k * layer_bit[0] + neighbor * layer_bit[1], layer_bit[1]) + 1);
        neighbor /= k;
    }
    return ret + (neighbor < ppos ? -1 : 1) * (decode_left(offset + neighbor * layer_bit[0], layer_bit[0]) + 1);
  }

  __host__ __device__
  VertexID_t get_vertex(VertexID_t neighbor) {
    short this_h = 0;
    VertexID_t layer_num = k;
    // bit_pos[i] = start offset in bit of layer i
    Offset_t bit_pos[5];
    bit_pos[0] = offset;

    while (neighbor >= layer_num) {
        neighbor -= layer_num;
        bit_pos[this_h + 1] = bit_pos[this_h] + layer_bit[this_h] * layer_num;
        ++this_h;
        layer_num *= k;
    }

    VertexID_t cur_vertex = vertex;
    VertexID_t tmp_vertex = decode_left(bit_pos[this_h] + neighbor * layer_bit[this_h], layer_bit[this_h]) + 1;
    while (--this_h >= 0) {
        cur_vertex = cur_vertex + ((neighbor % k) <= ((k - 1) >> 1) ? -1: 1) * tmp_vertex;
        neighbor /= k;
        tmp_vertex = decode_left(bit_pos[this_h] + neighbor * layer_bit[this_h], layer_bit[this_h]) + 1;
    }
    return cur_vertex + (neighbor < ppos ? -1 : 1) * tmp_vertex;
  }
};

class CCGNeighborApp {
public:
  
  __host__
  void init(const std::vector<int64_t> &_fanouts, const DGLContext &ctx)
  {
    std::vector<int64_t> f_copy(_fanouts.begin(), _fanouts.end());
    fanouts = std::move(f_copy);
    vecCOO.clear();
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::NeighborSampling;
  }

  __host__ 
  void initStepSample(const int64_t &_spl_len, const DGLContext &ctx)
  {
    spl_len = _spl_len;
    picked_row = dgl::aten::NewIdArray(spl_len, ctx, sizeof(VertexID_t) * 8); // DGL_ARRAY
    picked_col = dgl::aten::NewIdArray(spl_len, ctx, sizeof(VertexID_t) * 8);
    picked_idx = dgl::aten::NewIdArray(spl_len, ctx, sizeof(EdgePos_t) * 8);
    out_rows = static_cast<VertexID_t *>(picked_row->data);
    out_cols = static_cast<VertexID_t *>(picked_col->data);
    out_idxs = static_cast<EdgePos_t *>(picked_idx->data);
  }

  __host__
  void procStepSample()
  {
    auto _picked_row = picked_row.CreateView({spl_len}, picked_row->dtype);
    auto _picked_col = picked_col.CreateView({spl_len}, picked_col->dtype);
    auto _picked_idx = picked_idx.CreateView({spl_len}, picked_idx->dtype);
    // std::cout << "CCGSample COO Matrix: length: " << spl_len << std::endl;
    // gpuprint<<<1,1>>>(out_rows, spl_len);
    // cudaDeviceSynchronize();
    // gpuprint<<<1,1>>>(out_cols, spl_len);
    // cudaDeviceSynchronize();
    // gpuprint<<<1,1>>>(out_idxs, spl_len);
    // cudaDeviceSynchronize();
    vecCOO.emplace_back(n_nodes, n_nodes, _picked_col, _picked_row, _picked_idx);
  }

  __host__ int steps() {return fanouts.size();}
  
  __host__
  int getFinalSampleSize()
  {
    size_t finalSampleSize = 0;
    size_t neighborsToSampleAtStep = 1;
    for (auto step : fanouts)
    {
      neighborsToSampleAtStep *= step;
      finalSampleSize += neighborsToSampleAtStep;
    }
    return finalSampleSize;
  }

  __device__ inline
  VertexID_t next(const int step, const VertexID_t* transit, const VertexID_t sampleIdx, const VertexID_t numEdges, const EdgePos_t neighbrID, curandState* state, BCGVertex * bcgv) {
    if (numEdges == 0)
      return -1;
    if (numEdges == 1)
      return bcgv->get_vertex2(0);
    // printf("%d\n", *transit);
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);
    // return bcgv->get_vertex(x);
    VertexID_t ret = bcgv->get_vertex2(x);
    return ret;
  }

  __device__ inline
  VertexID_t next(const int step, const VertexID_t* transit, const VertexID_t sampleIdx, const VertexID_t numEdges, const EdgePos_t neighbrID, curandState* state, BCGVertex * bcgv, EdgePos_t& neighbor_pos) {
    if (numEdges == 0)
      return -1;
    neighbor_pos = RandNumGen::rand_int(state, numEdges);
    // if (numEdges == 1)
    //   return bcgv->get_vertex2(0);
    return bcgv->get_vertex(neighbor_pos);
    // VertexID_t ret = bcgv->get_vertex(neighbor_pos);
    // // printf("%ld -> %ld\n", *transit, ret);
    // return ret;
  }

  const int VERTICES_PER_SAMPLE = 1;

  __host__ __device__ EdgePos_t numSamples(VertexID_t n_nodes) {
    return n_nodes / VERTICES_PER_SAMPLE;
  }

  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, VertexID_t n_nodes, CCGSample& sample) {
    std::vector<VertexID_t> initialValue;
    for (int i = 0; i < VERTICES_PER_SAMPLE; i++)
      initialValue.push_back(sampleIdx);
    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(void) {
    return VERTICES_PER_SAMPLE; // 1
  }

  __host__ __device__ bool hasExplicitTransits() {
    return false;
  }

  __host__ __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, CCGSample& sample, int transitIdx, curandState* randState) {
    return -1;
  }

  __host__ __device__
  CCGSample initializeSample(const VertexID_t sampleID) {
    CCGSample sample;
    return sample;
  }
};

class CCGRandomWalkApp {
public:

  __host__ int steps() {return trace_length - 1;}

  __host__ __device__ int samplingType()
  {
    return SamplingType::RandomWalkSampling;
  }

  __host__
  void init(const IdArray &seeds, int length, const DGLContext &ctx)
  {
    int64_t num_seeds = seeds.NumElements();
    trace_length = length;
    traces = IdArray::Empty({num_seeds, trace_length}, seeds->dtype, ctx);
    out_trace = traces.Ptr<VertexID_t>();
    fanouts = std::vector<VertexID_t>(trace_length - 1, 1);
  }

  __host__
  int getFinalSampleSize() {
    return trace_length;
  }

  __host__ __device__ 
  int stepSize(int k) {
    return 1;
  }

  const int VERTICES_PER_SAMPLE = 1;

  __host__ __device__ EdgePos_t numSamples(VertexID_t n_nodes)
  {
    return n_nodes / VERTICES_PER_SAMPLE;
  }

  __host__
  void initStepSample(const int64_t &_spl_len, const DGLContext &ctx)
  {
    spl_len = _spl_len;
  }

  __host__
  void procStepSample()
  {

  }

  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, VertexID_t n_nodes, CCGSample& sample) {
    std::vector<VertexID_t> initialValue;
    for (int i = 0; i < VERTICES_PER_SAMPLE; i++)
      initialValue.push_back(sampleIdx);
    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(void)
  {
    return VERTICES_PER_SAMPLE;
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __host__ __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    return -1;
  }

  template<class SampleType>
  __host__ SampleType initializeSample(const VertexID_t sampleID)
  {
    CCGSample sample;
    return sample;
  }

  __device__ inline
  VertexID_t next(const int step, const VertexID_t* transit, const VertexID_t sampleIdx, const VertexID_t numEdges, const EdgePos_t neighbrID, curandState* state, BCGVertex * bcgv, EdgePos_t& neighbor_pos) {
    if (numEdges == 0)
      return -1;
    neighbor_pos = RandNumGen::rand_int(state, numEdges);
    return bcgv->get_vertex(neighbor_pos);
    // VertexID_t ret = bcgv->get_vertex(neighbor_pos);
    // // printf("%ld -> %ld\n", *transit, ret);
    // return ret;
  }

  // template<typename SampleType, typename EdgeArray, typename WeightArray>
  // __device__ inline
  // VertexID_t next(int step, const VertexID_t* transit, const VertexID_t sampleIdx,
  //               SampleType* sample, 
  //               const float max_weight,
  //               EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
  //               const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  // {
  //   if (numEdges == 0) {
  //     return -1;
  //   }
  //   if (numEdges == 1) {
  //     return transitEdges[0];
  //   }
    
  //   EdgePos_t x = RandNumGen::rand_int(state, numEdges);
  //   float y = curand_uniform(state)*max_weight;

  //   while (y > transitEdgeWeights[x]) {
  //     x = RandNumGen::rand_int(state, numEdges);
  //     y = curand_uniform(state)*max_weight;
  //   }

  //   return transitEdges[x];
  // }
};

std::vector<dgl::aten::COOMatrix> CCGSampleNeighbors(uint64_t n_nodes, void *gpu_ccg, void *crs, NextDoorData *nextDoorData, IdArray &seed_nodes, const std::vector<int64_t> &fanouts);
void *CCGCopyTo(uint64_t n_nodes, int ubl, const std::vector<uint32_t> &g_data, const std::vector<uint32_t> &g_offset, const DGLContext &ctx);
void *InitCurand(const DGLContext &ctx);

__global__ void setGPUSeeds(int initialVertexSize, VertexID_t *seed_nodes, VertexID_t seed_num, VertexID_t n_nodes, CCGSample *samples, VertexID_t *initialContents, VertexID_t *initialTransitToSampleValues);

void allocNextDoorDataOnDevice(NextDoorData& data, const DGLContext &ctx);
void setNextDoorData(NextDoorData* data, void* gpu_ccg, void *crs);

std::vector<dgl::aten::COOMatrix> CCGSampleFullLayers(uint64_t n_nodes, void *gpu_ccg, IdArray &seed_nodes_arr, int64_t num_layers);
__global__ void initializeFullLayersSample(VertexID_t *seed_nodes, VertexID_t seed_num, VertexID_t n_nodes, GPUBCGPartition *ccg, EdgePos_t* deg);


template <DGLDeviceType XPU, typename IdType, typename FloatType>
std::pair<dgl::aten::COOMatrix, FloatArray> CCGLaborSampling(
    void *gpu_ccg,
    VertexID_t n_nodes,
    IdArray rows,
    int64_t num_samples,
    FloatArray prob,
    int importance_sampling,
    IdArray random_seed,
    IdArray NIDs);

EdgePos_t CCGNumEdges();
DegreeArray CCGOutGegrees(IdArray vids);

IdArray CCGRandomWalk(uint64_t n_nodes, void *gpu_ccg, NextDoorData *nextDoorData, IdArray seeds, int64_t length);

}
// }  // namespace sampling
}  // namespace dgl



#endif