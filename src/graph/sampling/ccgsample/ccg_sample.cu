
#include <cstdio>
#include <curand_kernel.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <numeric>
#include <type_traits>
#include <utility>
#include <dgl/aten/coo.h>
#include <dgl/random.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/c_runtime_api.h>


#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

// #include "../../../c_api_common.h"
#include "../../../runtime/cuda/cuda_common.h"
#include "../../../array/cuda/atomic.cuh"
#include "../../../array/cuda/utils.h"
#include "../../transform/cuda/cuda_map_edges.cuh"
#include "../../../array/cuda/dgl_cub.cuh"
#include "../../../array/cuda/functor.cuh"
#include "../../../array/cuda/spmm.cuh"

#include "./utils.h"
#include "./ccg_sample.h"
// #include "./ccg.cuh"

using namespace dgl::aten;
using namespace dgl::runtime;

namespace dgl {

namespace tcpdgl {

const size_t CCGCurandNum = 5L * 1024L * 1024L;

__constant__ char bcgPartitionBuff[sizeof(BCGPartition)];

// graph info
VertexID_t n_nodes;
std::vector<int64_t> fanouts;
int64_t *d_fanouts;
int64_t trace_length;

// sample result
std::vector<dgl::aten::COOMatrix> vecCOO;
IdArray traces;
VertexID_t *out_trace;

// step result
dgl::NDArray picked_row;
dgl::NDArray picked_col;
dgl::NDArray picked_idx;
VertexID_t *out_rows;
VertexID_t *out_cols;
EdgePos_t *out_idxs;
int64_t spl_len;


__global__ void gpuprint(int *arr, int len = 10, int mi = -1)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int i = 0; i < len; ++i)
    if(arr[i]>mi) printf("%d ", arr[i]);
    printf("\n");
  }
}

__global__ void gpuprint(int64_t *arr, int len = 10, int mi = -1)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int i = 0; i < len; ++i)
    if(arr[i]>mi) printf("%ld ", arr[i]);
    printf("\n");
  }
}

__global__ void gpuck(int64_t *arr_col, int64_t *arr_row, int64_t *arr_idx, int len, int nodes, int edges)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int i = 0; i < len; ++i){
      if(arr_idx[i]==0) printf("%d: (%ld, %ld, %ld)\n", i, arr_row[i], arr_col[i], arr_idx[i]);
      else if(arr_col[i]>=nodes || arr_row[i]>=nodes) printf("(%ld, %ld, %ld)\n", arr_row[i], arr_col[i], arr_idx[i]);
      else if(arr_col[i]<0 || arr_row[i]<0) printf("(%ld, %ld, %ld)\n", arr_row[i], arr_col[i], arr_idx[i]);
    }
  }
}

__global__ void gpucknull(int64_t *arr, int64_t len = 10)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int64_t i = 0; i < len; ++i)
    if(arr[i] != 20) printf("(%ld %ld)", i, arr[i]);
    printf("\n");
  }
}

enum TransitKernelTypes
{
  GridKernel = 1,
  ThreadBlockKernel = 2,
  SubWarpKernel = 3,
  IdentityKernel = 4,
  NumKernelTypes = 4
};

enum TransitParallelMode
{
  // Describes the execution mode of Transit Parallel.
  NextFuncExecution,                 // Execute the next function
  CollectiveNeighborhoodSize,        // Compute size of collective neighborhood
  CollectiveNeighborhoodComputation, // Compute the collective neighborhood
};

__host__ __device__
    EdgePos_t
    subWarpSizeAtStep(EdgePos_t x)
{
  // SubWarpSize is set to next power of 2
  if (x && (!(x & (x - 1))))
    return x;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

__global__ void init_curand_states(curandState *states, size_t num_states)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < num_states)
    curand_init(threadIdx.x, thread_id, 0, &states[thread_id]);
}

__host__ __device__ bool isValidSampledVertex(VertexID_t neighbor, VertexID_t InvalidVertex)
{
  return neighbor != InvalidVertex && neighbor != -1;
}

void allocNextDoorDataOnDevice(NextDoorData &data, const DGLContext &ctx)
{
  data.devices = {0};
  auto device = runtime::DeviceAPI::Get(ctx);

  // size_t free, free1, tot;
  // CUDA_CALL(cudaMemGetInfo(&free, &tot));

  const auto samples_num = data.sampleNum;
  const auto finalSampleSize = data.finalSampleSize;
  const auto maxNeighborsToSample = data.maxNeighborsToSample;

  data.INVALID_VERTEX = data.n_nodes;
  int maxBits = 0;
  while ((data.INVALID_VERTEX >> maxBits) != 0)
  {
    maxBits++;
  }
  data.maxBits = maxBits;

  // Allocate storage for final samples on GPU
  data.dSamplesToTransitMapKeys = std::vector<VertexID_t *>(data.devices.size(), nullptr);
  data.dSamplesToTransitMapValues = std::vector<VertexID_t *>(data.devices.size(), nullptr);
  data.dTransitToSampleMapKeys = std::vector<VertexID_t *>(data.devices.size(), nullptr);
  data.dTransitToSampleMapValues = std::vector<VertexID_t *>(data.devices.size(), nullptr);
  data.dSampleInsertionPositions = std::vector<VertexID_t *>(data.devices.size(), nullptr);
  data.dNeighborhoodSizes = std::vector<EdgePos_t *>(data.devices.size(), nullptr);
  data.dCurandStates = std::vector<curandState *>(data.devices.size(), nullptr);
  data.maxThreadsPerKernel = std::vector<size_t>(data.devices.size(), 0);
  data.dFinalSamples = std::vector<VertexID_t *>(data.devices.size(), nullptr);
  // data.dInitialSamples = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dOutputSamples = std::vector<CCGSample *>(data.devices.size(), nullptr);

  const size_t numDevices = data.devices.size();
  for (size_t deviceIdx = 0; deviceIdx < data.devices.size(); deviceIdx++)
  {
    // auto device = data.devices[deviceIdx];
    // Per Device Allocation
    // CHK_CU(cudaSetDevice(device));

    const size_t perDeviceNumSamples = PartDivisionSize(samples_num, deviceIdx, numDevices);
    const size_t deviceSampleStartPtr = PartStartPointer(samples_num, deviceIdx, numDevices);

    // Allocate storage and copy initial samples on GPU
    // size_t partDivisionSize = CCGApp().initialSampleSize() * perDeviceNumSamples;
    // size_t partStartPtr = CCGApp().initialSampleSize() * deviceSampleStartPtr;
    // CHK_CU(cudaMalloc(&data.dInitialSamples[deviceIdx], sizeof(VertexID_t)*partDivisionSize));
    // CHK_CU(cudaMemcpy(data.dInitialSamples[deviceIdx], &data.initialContents[0] + partStartPtr,
    //                   sizeof(VertexID_t)*partDivisionSize, cudaMemcpyHostToDevice));

    // Allocate storage for samples on GPU
    // if (sizeof(CCGSample) > 0)
    // {
    //   CHK_CU(cudaMalloc(&data.dOutputSamples[deviceIdx], sizeof(CCGSample) * perDeviceNumSamples));

    //   //   // dgl::NDArray _doutput = dgl::aten::NewIdArray(perDeviceNumSamples, ctx, sizeof(CCGSample) * 8);
    //   //   // data.dOutputSamples[deviceIdx] = static_cast<VertexID_t*>(picked_row->data);

    //   //   CHK_CU(cudaMemcpy(data.dOutputSamples[deviceIdx], &data.samples[0] + deviceSampleStartPtr, sizeof(CCGSample)*perDeviceNumSamples,
    //   //                     cudaMemcpyHostToDevice));
    // }
    // CHK_CU(cudaMalloc(&data.dFinalSamples[deviceIdx], sizeof(VertexID_t) * finalSampleSize * perDeviceNumSamples));
    // utils::gpu_memset(data.dFinalSamples[deviceIdx], data.INVALID_VERTEX, finalSampleSize * perDeviceNumSamples);

    // Samples to Transit Map
    std::cout<<"alloc " << perDeviceNumSamples << " * " << subWarpSizeAtStep(maxNeighborsToSample) <<std::endl;
    data.dSamplesToTransitMapKeys[deviceIdx] = static_cast<VertexID_t*>(device->AllocDataSpace(ctx, sizeof(VertexID_t) * perDeviceNumSamples * subWarpSizeAtStep(maxNeighborsToSample), sizeof(VertexID_t), DGLDataType{kDGLInt, 64, 1}));
    data.dSamplesToTransitMapValues[deviceIdx] = static_cast<VertexID_t*>(device->AllocDataSpace(ctx, sizeof(VertexID_t) * perDeviceNumSamples * subWarpSizeAtStep(maxNeighborsToSample), sizeof(VertexID_t), DGLDataType{kDGLInt, 64, 1}));

    // Transit to Samples Map
    data.dTransitToSampleMapKeys[deviceIdx] = static_cast<VertexID_t*>(device->AllocDataSpace(ctx, sizeof(VertexID_t) * perDeviceNumSamples * subWarpSizeAtStep(maxNeighborsToSample), sizeof(VertexID_t), DGLDataType{kDGLInt, 64, 1}));
    data.dTransitToSampleMapValues[deviceIdx] = static_cast<VertexID_t*>(device->AllocDataSpace(ctx, sizeof(VertexID_t) * perDeviceNumSamples * subWarpSizeAtStep(maxNeighborsToSample), sizeof(VertexID_t), DGLDataType{kDGLInt, 64, 1}));

    // Same as initial values of samples for first iteration
    //  CHK_CU(cudaMemcpy(data.dTransitToSampleMapKeys[deviceIdx], &data.initialContents[0] + partStartPtr, sizeof(VertexID_t)*partDivisionSize,
    //                    cudaMemcpyHostToDevice));
    //  CHK_CU(cudaMemcpy(data.dTransitToSampleMapValues[deviceIdx], &data.initialTransitToSampleValues[0] + partStartPtr,
    //                    sizeof(VertexID_t)*partDivisionSize, cudaMemcpyHostToDevice));

    // Insertion positions per transit vertex for each sample
    data.dSampleInsertionPositions[deviceIdx] = static_cast<EdgePos_t*>(device->AllocDataSpace(ctx, sizeof(EdgePos_t) * perDeviceNumSamples, sizeof(EdgePos_t), DGLDataType{kDGLInt, 64, 1}));
    CHK_CU(cudaMalloc(&data.dSampleInsertionPositions[deviceIdx], sizeof(EdgePos_t) * perDeviceNumSamples));
    // CHK_CU(cudaDeviceSynchronize());
  }

  // CUDA_CALL(cudaMemGetInfo(&free1, &tot));
  // std::cout << "[PF] stat allocNextDoorDataOnDevice "<< (free-free1)/1024/1024 << std::endl;
  return;
}

__global__ void setGPUSeeds(int initialVertexSize, VertexID_t *seed_nodes, VertexID_t seed_num, VertexID_t n_nodes, CCGSample *samples, VertexID_t *initialContents, VertexID_t *initialTransitToSampleValues)
{
  VertexID_t sampleIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (sampleIdx < seed_num)
  {
    // CCGSample sample = CCGApp().initializeSample(sampleIdx);
    // samples[sampleIdx] = sample;
    // VertexID_t initialVertices[initialVertexSize];
    // for (int i = 0; i < initialVertexSize; i++)
    //   initialVertices[i] = seed_nodes[sampleIdx];
    for (int i = 0; i < initialVertexSize; ++i)
    {
      // initialContents[sampleIdx * initialVertexSize + i] = initialVertices[i];
      initialContents[sampleIdx * initialVertexSize + i] = seed_nodes[sampleIdx];
      initialTransitToSampleValues[sampleIdx * initialVertexSize + i] = sampleIdx;
    }
  }
  return;
}

template<typename CCGApp>
void initializeNextDoorSample(NextDoorData &data, const IdArray &seed_nodes_arr, const int tot_step)
{
  unsigned long seed_node_size = seed_nodes_arr.NumElements();
  VertexID_t *d_seed_nodes = static_cast<VertexID_t *>(seed_nodes_arr->data);

  data.sampleNum = seed_node_size;

  // std::cout << "initialVertexSize " << CCGApp().VERTICES_PER_SAMPLE << " seed_node_size " << seed_node_size << " data.sampleNum " << data.sampleNum << " data.n_nodes " << data.n_nodes << " tot_step " << tot_step << std::endl;
  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] bg setGPUSeeds " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << std::endl;
  for (size_t deviceIdx = 0; deviceIdx < data.devices.size(); ++deviceIdx)
  {
    // CHK_CU(cudaSetDevice(deviceIdx));
    if (tot_step == 1) 
      setGPUSeeds<<<utils::thread_block_size(seed_node_size, 256UL), 256UL>>>
        (CCGApp().VERTICES_PER_SAMPLE, d_seed_nodes, data.sampleNum, data.n_nodes, data.dOutputSamples[deviceIdx], data.dSamplesToTransitMapValues[deviceIdx], data.dSamplesToTransitMapKeys[deviceIdx]);
    else
      setGPUSeeds<<<utils::thread_block_size(seed_node_size, 256UL), 256UL>>>
        (CCGApp().VERTICES_PER_SAMPLE, d_seed_nodes, data.sampleNum, data.n_nodes, data.dOutputSamples[deviceIdx], data.dTransitToSampleMapKeys[deviceIdx], data.dTransitToSampleMapValues[deviceIdx]);
  }
  // CUDA_SYNC_DEVICE_ALL(data);
  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] ed setGPUSeeds " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << std::endl;

  // gpuprint<<<1, 1>>>(data.dTransitToSampleMapKeys[0]);
  // gpuprint<<<1, 1>>>(data.dTransitToSampleMapValues[0]);
  // CUDA_SYNC_DEVICE_ALL(data);

  // data.samples = std::vector<CCGSample>();
  // data.initialContents = std::vector<int64_t>();
  // data.initialTransitToSampleValues = std::vector<int64_t>();

  // for (size_t sampleIdx = 0; sampleIdx < seed_nodes.size(); ++sampleIdx) {
  //   CCGSample sample = CCGApp().initializeSample(sampleIdx);
  //   data.samples.push_back(sample);
  //   auto initialVertices = CCGApp().initialSample(seed_nodes[sampleIdx], data.n_nodes, data.samples[sampleIdx]);
  //   if ((EdgePos_t)initialVertices.size() != CCGApp().initialSampleSize()) {
  //     printf("Error intialSampleSize at ccg_sample.cu\n");
  //     abort();
  //   }
  //   data.initialContents.insert(data.initialContents.end(), initialVertices.begin(), initialVertices.end());
  //   for (auto v : initialVertices) {
  //     data.initialTransitToSampleValues.push_back(sampleIdx);
  //   }
  // }

  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] ed initspl_for " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << std::endl;

  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] bg initspl_maloc " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << std::endl;
  // const size_t numDevices = data.devices.size();
  // const auto samples_num = data.sampleNum;
  // for(size_t deviceIdx = 0; deviceIdx < data.devices.size(); deviceIdx++) {
  //   auto device = data.devices[deviceIdx];
  //   //Per Device Allocation
  //   CHK_CU(cudaSetDevice(device));

  //   const size_t perDeviceNumSamples = PartDivisionSize(samples_num, deviceIdx, numDevices);
  //   const size_t deviceSampleStartPtr = PartStartPointer(samples_num, deviceIdx, numDevices);

  //   if (sizeof(CCGSample) > 0) {
  //     CHK_CU(cudaMemcpy(data.dOutputSamples[deviceIdx], &data.samples[0] + deviceSampleStartPtr, sizeof(CCGSample)*perDeviceNumSamples,
  //                       cudaMemcpyHostToDevice));
  //   }
  // }
  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] ed initspl_maloc " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << std::endl;

  return;
}

void freeDeviceData(NextDoorData &data)
{
  for (size_t deviceIdx = 0; deviceIdx < data.devices.size(); deviceIdx++)
  {
    auto device = data.devices[deviceIdx];
    CHK_CU(cudaSetDevice(device));
    CHK_CU(cudaFree(data.dSamplesToTransitMapKeys[deviceIdx]));
    CHK_CU(cudaFree(data.dSamplesToTransitMapValues[deviceIdx]));
    CHK_CU(cudaFree(data.dTransitToSampleMapKeys[deviceIdx]));
    CHK_CU(cudaFree(data.dTransitToSampleMapValues[deviceIdx]));
    // CHK_CU(cudaFree(data.dSampleInsertionPositions[deviceIdx]));
    // CHK_CU(cudaFree(data.dCurandStates[deviceIdx]));
    CHK_CU(cudaFree(data.dFinalSamples[deviceIdx]));
    // CHK_CU(cudaFree(data.dInitialSamples[deviceIdx]));
    if (sizeof(CCGSample) > 0)
      CHK_CU(cudaFree(data.dOutputSamples[deviceIdx]));
  }
}

__global__ void get_num_edges(EdgePos_t *num_edges) {
  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
  num_edges[0] = bcg->n_edges;
}

EdgePos_t CCGNumEdges() {
  EdgePos_t *d_num_edges, *num_edges;
  CUDA_CALL(cudaMalloc(&d_num_edges, sizeof(EdgePos_t)));
  num_edges = new EdgePos_t[1];
  get_num_edges<<<1,1>>>(d_num_edges);
  CUDA_CALL(cudaMemcpy(num_edges, d_num_edges, sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
  return *num_edges;
}

__global__ void set_deg(EdgePos_t *deg, Offset_t *offset, Graph_t *graph, VertexID_t n_nodes)
{
  for (VertexID_t v = threadIdx.x + blockIdx.x * blockDim.x; v < n_nodes; v += blockDim.x * gridDim.x)
  {
    auto bcgv = BCGVertex(graph, offset[v]);
    deg[v] = bcgv.outd;
  }
  return;
}

__global__ void get_deg(const int64_t* vids, int64_t len, int64_t* deg)
{
  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
  for (VertexID_t i = threadIdx.x + blockIdx.x * blockDim.x; i < len; i += blockDim.x * gridDim.x)
  {
    int64_t v = vids[i];
    assert(v>=0 && v<bcg->n_nodes);
    deg[i] = bcg->degoffset[v+1] - bcg->degoffset[v];
  }
}

DegreeArray CCGOutGegrees(IdArray vids)
{
  const auto len = vids->shape[0];
  const int64_t* vid_data = static_cast<int64_t*>(vids->data);
  DegreeArray rst = DegreeArray::Empty({len}, vids->dtype, vids->ctx);
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  get_deg<<<min(8192L, (len + 255L) / 256L), 256>>>(vid_data, (int64_t)len, rst_data);
  return rst;
}

__global__ void decode_offset(Graph_t *offset_data, Offset_t *offset, VertexID_t n_nodes, uint8_t ubl)
{
  VertexID_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (VertexID_t gos = 0; gos < n_nodes; gos += blockDim.x * gridDim.x)
  {
    VertexID_t u = gos + tid;
    if (u < n_nodes)
    {
      Offset_t tmp = 1ll * u * ubl;
      // auto type_id = tmp / GRAPH_LEN - sm_os;
      Offset_t type_id = tmp / GRAPH_LEN;
      int16_t bit_offset = tmp - type_id * GRAPH_LEN, bit_len = ubl;
      tmp = (offset_data[type_id] << bit_offset) >> max(GRAPH_LEN - bit_len, bit_offset);
      bit_len -= (GRAPH_LEN - bit_offset);

      if (bit_len > 0)
      {
        tmp = (tmp << min(bit_len, 32)) | (offset_data[type_id + 1] >> max(GRAPH_LEN - bit_len, 0));
        bit_len -= GRAPH_LEN;
        if (bit_len > 0)
        {
          tmp = (tmp << min(bit_len, 32)) | (offset_data[type_id + 2] >> max(GRAPH_LEN - bit_len, 0));
        }
      }
      offset[u] = tmp;
    }
  }
  return;
}

BCGPartition copyPartitionToGPU(uint64_t n_nodes, int ubl, const std::vector<uint32_t> &g_data, const std::vector<uint32_t> &g_offset, GPUBCGPartition *gpuBCGPartition, const DGLContext &ctx)
{
  auto size_offset = sizeof(Offset_t) * (n_nodes + 1);
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  
  gpuBCGPartition->d_offset = static_cast<Offset_t *>(device->AllocDataSpace(ctx, size_offset, sizeof(Offset_t), DGLDataType{kDGLUInt, 64, 1}));
  // CHK_CU(cudaMalloc(&(gpuBCGPartition->d_offset), size_offset));
  gpuBCGPartition->d_degoffset = static_cast<EdgePos_t *>(device->AllocDataSpace(ctx, sizeof(EdgePos_t) * (n_nodes + 1), sizeof(EdgePos_t), DGLDataType{kDGLInt, 64, 1}));
  // CHK_CU(cudaMalloc(&(gpuBCGPartition->d_degoffset), sizeof(EdgePos_t) * (n_nodes + 1)));
  gpuBCGPartition->d_deg = static_cast<EdgePos_t *>(device->AllocDataSpace(ctx, sizeof(EdgePos_t) * (n_nodes + 1), sizeof(EdgePos_t), DGLDataType{kDGLInt, 64, 1}));
  // CHK_CU(cudaMalloc(&(gpuBCGPartition->d_deg), sizeof(E·dgePos_t) * (n_nodes + 1)));
  // cudaDeviceSynchronize();
  Graph_t *d_offset_data;
  d_offset_data = static_cast<Graph_t *>(device->AllocWorkspace(ctx, sizeof(Graph_t) * (g_offset.size()), DGLDataType{kDGLUInt, 32, 1}));
  // CHK_CU(cudaMalloc(&d_offset_data, sizeof(Graph_t) * (g_offset.size())));

  device->CopyDataFromTo(&(g_offset[0]), 0, d_offset_data, 0, sizeof(Graph_t) * g_offset.size(), DGLContext{kDGLCPU, 0}, ctx, DGLDataType{kDGLUInt, 32, 1});
  // CHK_CU(cudaMemcpy(d_offset_data, &(g_offset[0]), sizeof(Graph_t) * g_offset.size(), cudaMemcpyHostToDevice));
  // cudaDeviceSynchronize();
  device->StreamSync(ctx, stream);
  decode_offset<<<min(8192L, n_nodes / 256L), 256>>>(d_offset_data, gpuBCGPartition->d_offset, n_nodes, ubl);
  
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, gpuBCGPartition->d_offset, gpuBCGPartition->d_offset, n_nodes + 1);
  d_temp_storage = device->AllocWorkspace(ctx, temp_storage_bytes);
  // CHK_CU(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, gpuBCGPartition->d_offset, gpuBCGPartition->d_offset, n_nodes + 1);
  // std::cout<<temp_storage_bytes<<" "<<d_temp_storage<<std::endl;
  auto size_graph = sizeof(Graph_t) * g_data.size();
  gpuBCGPartition->d_graph = static_cast<Graph_t*> (device->AllocDataSpace(ctx, size_graph, sizeof(Graph_t), DGLDataType{kDGLUInt, 32, 1}));
  // CHK_CU(cudaMalloc(&(gpuBCGPartition->d_graph), size_graph));
  device->CopyDataFromTo(&(g_data[0]), 0, gpuBCGPartition->d_graph, 0, size_graph, DGLContext{kDGLCPU, 0}, ctx, DGLDataType{kDGLUInt, 0});
  // CHK_CU(cudaMemcpy(gpuBCGPartition->d_graph, &(g_data[0]), size_graph, cudaMemcpyHostToDevice));

  set_deg<<<min(8192L, (n_nodes + 255L) / 256L), 256>>>(gpuBCGPartition->d_deg, gpuBCGPartition->d_offset, gpuBCGPartition->d_graph, n_nodes);
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, gpuBCGPartition->d_deg, gpuBCGPartition->d_degoffset, n_nodes + 1);

  device->FreeWorkspace(ctx, d_temp_storage);
  // CHK_CU(cudaFree(d_temp_storage));
  EdgePos_t n_edges;
  device->CopyDataFromTo(gpuBCGPartition->d_degoffset, n_nodes * sizeof(EdgePos_t), &n_edges, 0, sizeof(EdgePos_t), ctx, DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
  // CHK_CU(cudaMemcpy(&n_edges, gpuBCGPartition->d_degoffset + n_nodes, sizeof(EdgePos_t), cudaMemcpyDeviceToHost));

  BCGPartition device_bcg_partition = BCGPartition(gpuBCGPartition->d_graph, gpuBCGPartition->d_offset, gpuBCGPartition->d_degoffset, n_nodes, n_edges);

  // force sync
  device->StreamSync(ctx, stream);
  std::cout << "CCG (" << size_graph + sizeof(Graph_t) * g_offset.size() << "B, " << size_graph << "B for graph, " << sizeof(Graph_t) * g_offset.size() << "B for offset) loaded and transfered to GPU, including " << n_nodes << " vertices and " << n_edges << " edges." << std::endl;

  return device_bcg_partition;
}

void *InitCurand(const DGLContext &ctx)
{
  curandState *ret;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // size_t free, free1, tot;
  // CUDA_CALL(cudaMemGetInfo(&free, &tot));

  ret = static_cast<curandState*>(device->AllocDataSpace(ctx, CCGCurandNum * sizeof(curandState), alignof(curandState), DGLDataType{kDGLInt, 32, 1}));
  init_curand_states<<<utils::thread_block_size(CCGCurandNum, 256UL), 256UL>>>(ret, CCGCurandNum);
  device->StreamSync(ctx, stream);

  // CUDA_CALL(cudaMemGetInfo(&free1, &tot));
  // std::cout << "[PF] stat InitCurand "<< (free-free1)/1024/1024 << std::endl;
  // device->StreamSync(ctx, stream);
  return (void *)ret;
}

void *CCGCopyTo(uint64_t n_nodes, int ubl, const std::vector<uint32_t> &g_data, const std::vector<uint32_t> &g_offset, const DGLContext &ctx)
{
  size_t free, free1, tot;
  CUDA_CALL(cudaMemGetInfo(&free, &tot));

  GPUBCGPartition *gpuBCGPartition = new GPUBCGPartition;
  BCGPartition deviceBCGPartition = copyPartitionToGPU(n_nodes, ubl, g_data, g_offset, gpuBCGPartition, ctx);

  CUDA_CALL(cudaMemcpyToSymbol(bcgPartitionBuff, &deviceBCGPartition, sizeof(BCGPartition)));
  gpuBCGPartition->d_bcg = (BCGPartition *)bcgPartitionBuff;

  CUDA_CALL(cudaMemGetInfo(&free1, &tot));
  std::cout << "[PF] stat CCG_gpu "<< (free-free1)/1024/1024 << std::endl;
  return (void *)gpuBCGPartition;
}

__global__ void invalidVertexStartPos(int step, VertexID_t *transitToSamplesKeys, size_t totalTransits,
                                      const VertexID_t invalidVertex, EdgePos_t *outputStartPos)
{
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadId >= totalTransits)
  {
    return;
  }

  // If first transit is invalid.
  if (threadId == 0)
  {
    if (transitToSamplesKeys[0] == invalidVertex)
    {
      *outputStartPos = 0;
    }
    // printf("outputStartPos %d\n", *outputStartPos);
    return;
  }

  // TODO: Optimize this using overlaped tilling
  if (transitToSamplesKeys[threadId - 1] != invalidVertex &&
      transitToSamplesKeys[threadId] == invalidVertex)
  {
    *outputStartPos = threadId;
    return;
    // printf("outputStartPos %d\n", *outputStartPos);
  }
}

template <int TB_THREADS, TransitKernelTypes kTy, bool WRITE_KERNELTYPES>
__global__ void partitionTransitsInKernels(int step, EdgePos_t *uniqueTransits, EdgePos_t *uniqueTransitCounts,
                                            EdgePos_t *transitPositions,
                                            EdgePos_t uniqueTransitCountsNum, VertexID_t invalidVertex,
                                            EdgePos_t *gridKernelTransits, EdgePos_t *gridKernelTransitsNum,
                                            EdgePos_t *threadBlockKernelTransits, EdgePos_t *threadBlockKernelTransitsNum,
                                            EdgePos_t *subWarpKernelTransits, EdgePos_t *subWarpKernelTransitsNum,
                                            EdgePos_t *identityKernelTransits, EdgePos_t *identityKernelTransitsNum,
                                            int *kernelTypeForTransit, VertexID_t *transitToSamplesKeys, int fanout)
{
  const int SHMEM_SIZE = 7 * TB_THREADS;
  typedef cub::BlockScan<EdgePos_t, TB_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ EdgePos_t shGridKernelTransits[SHMEM_SIZE];
  __shared__ EdgePos_t threadToTransitPrefixSum[TB_THREADS];
  __shared__ EdgePos_t threadToTransitPos[TB_THREADS];
  __shared__ VertexID_t threadToTransit[TB_THREADS];
  __shared__ EdgePos_t totalThreadGroups;
  __shared__ EdgePos_t threadGroupsInsertionPos;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0)
    totalThreadGroups = 0;

  for (int i = threadIdx.x; i < SHMEM_SIZE; i += blockDim.x)
    shGridKernelTransits[i] = 0;

  __syncthreads();

  VertexID_t transit = uniqueTransits[threadId];
  EdgePos_t trCount = (threadId >= uniqueTransitCountsNum || transit == invalidVertex) ? -1 : uniqueTransitCounts[threadId];
  EdgePos_t trPos = (threadId >= uniqueTransitCountsNum || transit == invalidVertex) ? -1 : transitPositions[threadId];
  int subWarpSize = subWarpSizeAtStep(fanout);

  int kernelType = -1;
  if (useGridKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::GridLevel)
  {
    kernelType = TransitKernelTypes::GridKernel;
    printf("%d GridKernel transit %ld\n", threadId, transit);
  }
  else if (useThreadBlockKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::BlockLevel)
  {
    kernelType = TransitKernelTypes::ThreadBlockKernel;
  }
  else if (useSubWarpKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::SubWarpLevel)
  {
    kernelType = TransitKernelTypes::SubWarpKernel;
  }
  else
  {
    kernelType = TransitKernelTypes::IdentityKernel;
  }

  if (WRITE_KERNELTYPES)
  {
    if (threadId < uniqueTransitCountsNum && kernelType != IdentityKernel && transit != invalidVertex)
    {
      kernelTypeForTransit[transit] = kernelType;
    }

    if (kernelType == IdentityKernel && transit != invalidVertex && trCount != -1)
    {
      *identityKernelTransitsNum = 1;
    }
  }

  __syncthreads();

  EdgePos_t numThreadGroups = 0;
  EdgePos_t *glKernelTransitsNum, *glKernelTransits;
  const int threadGroupSize = (kTy == TransitKernelTypes::GridKernel) ? LoadBalancing::LoadBalancingThreshold::GridLevel / subWarpSize : ((kTy == TransitKernelTypes::ThreadBlockKernel) ? LoadBalancing::LoadBalancingThreshold::BlockLevel / subWarpSize : ((kTy == TransitKernelTypes::SubWarpKernel) ? LoadBalancing::LoadBalancingThreshold::SubWarpLevel : -1));

  if (kTy == TransitKernelTypes::GridKernel && useGridKernel)
  {
    if (kernelType == TransitKernelTypes::GridKernel)
    {
      numThreadGroups = DIVUP(trCount, threadGroupSize);
      threadToTransitPos[threadIdx.x] = trPos;
      threadToTransit[threadIdx.x] = transit;
    }
    else
    {
      numThreadGroups = 0;
      threadToTransitPos[threadIdx.x] = 0;
      threadToTransit[threadIdx.x] = -1;
    }
    glKernelTransitsNum = gridKernelTransitsNum;
    glKernelTransits = gridKernelTransits;
  }
  else if (kTy == TransitKernelTypes::ThreadBlockKernel && useThreadBlockKernel)
  {
    if (kernelType == TransitKernelTypes::ThreadBlockKernel)
    {
      numThreadGroups = DIVUP(trCount, threadGroupSize);
      threadToTransitPos[threadIdx.x] = trPos;
      threadToTransit[threadIdx.x] = transit;
    }
    else
    {
      numThreadGroups = 0;
      threadToTransitPos[threadIdx.x] = 0;
      threadToTransit[threadIdx.x] = -1;
    }
    glKernelTransitsNum = threadBlockKernelTransitsNum;
    glKernelTransits = threadBlockKernelTransits;
    // if (blockIdx.x == 0) {
    //   printf("threadIdx.x %d transit %d\n", threadIdx.x, transit);
    // }
  }
  else if (kTy == TransitKernelTypes::SubWarpKernel && useSubWarpKernel)
  {
    if (kernelType == TransitKernelTypes::SubWarpKernel)
    {
      numThreadGroups = DIVUP(trCount, threadGroupSize);
      threadToTransitPos[threadIdx.x] = trPos;
      threadToTransit[threadIdx.x] = transit;
    }
    else
    {
      numThreadGroups = 0;
      threadToTransitPos[threadIdx.x] = 0;
      threadToTransit[threadIdx.x] = -1;
    }
    glKernelTransitsNum = subWarpKernelTransitsNum;
    glKernelTransits = subWarpKernelTransits;
  }
  else
  {
    return;
    // continue;
  }

  __syncthreads();
  // Get all grid kernel transits
  EdgePos_t prefixSumThreadData = 0;
  BlockScan(temp_storage).ExclusiveSum(numThreadGroups, prefixSumThreadData);

  __syncthreads();

  if (threadIdx.x == blockDim.x - 1)
  {
    totalThreadGroups = prefixSumThreadData + numThreadGroups;
    // if (kTy == 2 && blockIdx.x == 27) printf("totalThreadGroups %d kTy %d blockIdx.x %d\n", totalThreadGroups, kTy, blockIdx.x);
    threadGroupsInsertionPos = utils::atomicAdd(glKernelTransitsNum, totalThreadGroups);
  }
  __syncthreads();

  threadToTransitPrefixSum[threadIdx.x] = prefixSumThreadData;

  __syncthreads();

  for (int tgIter = 0; tgIter < totalThreadGroups; tgIter += SHMEM_SIZE)
  {
    __syncthreads();
    for (int i = threadIdx.x; i < SHMEM_SIZE; i += blockDim.x)
    {
      shGridKernelTransits[i] = 0;
    }

    __syncthreads();

    int prefixSumIndex = prefixSumThreadData - tgIter;
    if (prefixSumIndex < 0 && prefixSumIndex + numThreadGroups > 0)
    {
      prefixSumIndex = 0;
    }
    if (numThreadGroups > 0)
    {
      if (prefixSumIndex >= 0 && prefixSumIndex < SHMEM_SIZE)
      {
        shGridKernelTransits[prefixSumIndex] = threadIdx.x;
      }
    }

    __syncthreads();

    for (int tbs = threadIdx.x; tbs < DIVUP(min((EdgePos_t)SHMEM_SIZE, totalThreadGroups - tgIter), TB_THREADS) * TB_THREADS; tbs += blockDim.x)
    {
      __syncthreads();
      EdgePos_t d = 0, e = 0;
      if (tbs < TB_THREADS)
      {
        d = (tbs < totalThreadGroups) ? shGridKernelTransits[tbs] : 0;
      }
      else if (threadIdx.x == 0)
      {
        d = (tbs < totalThreadGroups) ? max(shGridKernelTransits[tbs], shGridKernelTransits[tbs - 1]) : 0;
      }
      else
      {
        d = (tbs < totalThreadGroups) ? shGridKernelTransits[tbs] : 0;
      }

      __syncthreads();
      BlockScan(temp_storage).InclusiveScan(d, e, cub::Max());
      __syncthreads();

      if (tbs < totalThreadGroups)
        shGridKernelTransits[tbs] = e;

      __syncthreads();

      if (tbs + tgIter < totalThreadGroups)
      {
        EdgePos_t xx = shGridKernelTransits[tbs];
        assert(xx >= 0 && xx < blockDim.x);
        int previousTrPrefixSum = (tbs < totalThreadGroups && xx >= 0) ? threadToTransitPrefixSum[xx] : 0;

        EdgePos_t startPos = threadToTransitPos[xx];
        EdgePos_t pos = startPos + threadGroupSize * (tbs + tgIter - previousTrPrefixSum);

        VertexID_t transit = threadToTransit[xx];
        if (transit != -1)
        {
          int idx = threadGroupsInsertionPos + tbs + tgIter;
          glKernelTransits[idx] = pos;
          assert(kernelTypeForTransit[transit] == kTy);
          assert(transitToSamplesKeys[pos] == transit);
        }
      }

      __syncthreads();
    }

    __syncthreads();
  }
  __syncthreads();
}

#define STORE_TRANSIT_INDEX false

template <typename CCGApp, TransitParallelMode tpMode, int CollNeighStepSize>
__global__ void samplingKernel(const int step, const size_t threadsExecuted, const size_t currExecutionThreads,
                                const VertexID_t deviceFirstSample, const VertexID_t invalidVertex,
                                const VertexID_t *transitToSamplesKeys, const VertexID_t *transitToSamplesValues,
                                const size_t transitToSamplesSize, CCGSample *samples, const size_t NumSamples,
                                VertexID_t *samplesToTransitKeys, VertexID_t *samplesToTransitValues,
                                const size_t finalSampleSize,
                                EdgePos_t *sampleNeighborhoodSizes, EdgePos_t *sampleNeighborhoodPos,
                                VertexID_t *collectiveNeighborhoodCSRRows,
                                EdgePos_t *collectiveNeighborhoodCSRCols,
                                curandState *randStates, int tot_step, int64_t *fanouts,
                                VertexID_t *out_rows, VertexID_t *out_cols, EdgePos_t *out_idxs,
                                VertexID_t *out_trace, int64_t trace_length)
{
  // if (threadIdx.x == 0 && blockIdx.x == 0) printf("SK %d\n", step);
  EdgePos_t threadId = threadIdx.x + blockDim.x * blockIdx.x;

  if (threadId >= currExecutionThreads)
    return;

  curandState *randState = &randStates[threadId];

  threadId += threadsExecuted;
  // int stepSize;
  // if (tpMode == NextFuncExecution)
  // {
  //   stepSize = (int)fanouts[step];
  // }
  // else if (tpMode == CollectiveNeighborhoodComputation)
  // {
  //   stepSize = CollNeighStepSize;
  // }
  // else if (tpMode == CollectiveNeighborhoodSize)
  // {
  //   stepSize = 1;
  // }
  EdgePos_t transitIdx = threadId / fanouts[step];
  EdgePos_t transitNeighborIdx = threadId % fanouts[step];
  EdgePos_t numTransits = fanouts[step];

  VertexID_t sampleIdx = transitToSamplesValues[transitIdx];

  assert(sampleIdx < NumSamples);
  VertexID_t transit = transitToSamplesKeys[transitIdx];
  // printf("%d %d %d %d %d %d\n", threadIdx.x + blockDim.x * blockIdx.x, threadId, transitIdx, sampleIdx, NumSamples, transit);
  // if (threadId == 0)  assert(0);
  VertexID_t neighbor = invalidVertex;
  EdgePos_t neighbor_pos = 0;

  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];

  // assert(transit < invalidVertex);
  if (transit != invalidVertex)
  {
    BCGVertex bcgv(transit, bcg->graph, bcg->offset[transit]);
    EdgePos_t numTransitEdges = bcgv.outd;

    if (numTransitEdges != 0 && (tpMode == NextFuncExecution || tpMode == CollectiveNeighborhoodComputation))
    {
      // Execute next in this mode only

      if (tpMode == NextFuncExecution)
      {
        neighbor = CCGApp().next(step, &transit, sampleIdx,
                                  numTransitEdges, transitNeighborIdx, randState, &bcgv, neighbor_pos);
        neighbor_pos += bcg->degoffset[transit];
      }
    }
    else if (tpMode == CollectiveNeighborhoodSize)
    {
      // Compute size of collective neighborhood for each sample.
      utils::atomicAdd(&sampleNeighborhoodSizes[(sampleIdx - deviceFirstSample)], numTransitEdges);
    }
  }

  // __syncwarp();
  if (tpMode == NextFuncExecution)
  {
    EdgePos_t insertionPos = 0;
    // TODO: templatize over hasExplicitTransits()
    if (step != tot_step - 1)
    {
      // No need to store at last step
      if (CCGApp().hasExplicitTransits())
      {
        VertexID_t newTransit = CCGApp().stepTransits(step + 1, sampleIdx, samples[(sampleIdx - deviceFirstSample)], threadId % numTransits, randState);
        samplesToTransitValues[threadId] = newTransit != -1 ? newTransit : invalidVertex;
      }
      else
      {
        samplesToTransitValues[threadId] = neighbor != -1 ? neighbor : invalidVertex;
      }
      samplesToTransitKeys[threadId] = sampleIdx;
    }

    if (CCGApp().samplingType() == SamplingType::NeighborSampling)
    {
      insertionPos = threadId;
      // insertionPos = (sampleIdx - deviceFirstSample) * CCGApp().initialSampleSize() * CCGApp().stepSize(step - 1) + 
      assert(insertionPos < transitToSamplesSize);
      out_rows[insertionPos] = neighbor != -1 ? transit : invalidVertex;
      out_cols[insertionPos] = neighbor != -1 ? neighbor : invalidVertex;
      out_idxs[insertionPos] = neighbor != -1 ? neighbor_pos : -1;
    }
    else if (CCGApp().samplingType() == SamplingType::RandomWalkSampling)
    {
      insertionPos = trace_length * (sampleIdx - deviceFirstSample) + step + 1; // seed takes up a pos
      // if(!threadId) printf("[sampling kernel]: %d %ld %ld %ld %ld %ld %ld\n", step, threadId, sampleIdx, deviceFirstSample, transitIdx, neighbor, insertionPos);
      out_trace[insertionPos] = neighbor != -1 ? neighbor : invalidVertex;
    }
  }
}

template <typename CCGApp, int THREADS, bool COALESCE_CURAND_LOAD, bool HAS_EXPLICIT_TRANSITS>
__global__ void identityKernel(const int step, const VertexID_t deviceFirstSample, const VertexID_t invalidVertex,
                                const VertexID_t *transitToSamplesKeys, const VertexID_t *transitToSamplesValues,
                                const size_t transitToSamplesSize, CCGSample *samples, const size_t NumSamples,
                                VertexID_t *samplesToTransitKeys, VertexID_t *samplesToTransitValues,
                                VertexID_t *finalSamples, const size_t finalSampleSize, EdgePos_t *sampleInsertionPositions,
                                curandState *randStates, const int *kernelTypeForTransit, int64_t numTransits, int64_t *fanouts,
                                int tot_step, VertexID_t *out_rows, VertexID_t *out_cols, EdgePos_t *out_idxs,
                                VertexID_t *out_trace, int64_t trace_length)
{
  // if (threadIdx.x == 0 && blockIdx.x == 0) printf("IK %d\n", step);

  __shared__ unsigned char shMemCuRand[sizeof(curandState) * THREADS];

  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  curandState *curandSrcPtr;

  if (COALESCE_CURAND_LOAD)
  {
    const int intsInRandState = sizeof(curandState) / sizeof(int);
    int *shStateBuff = (int *)&shMemCuRand[0];

    int *randStatesAsInts = (int *)randStates;

    for (int i = threadIdx.x; i < intsInRandState * blockDim.x; i += blockDim.x)
    {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x * blockIdx.x];
    }

    __syncthreads();
    curandSrcPtr = (curandState *)(&shStateBuff[threadIdx.x * intsInRandState]);
  }
  else
  {
    curandSrcPtr = &randStates[threadId];
  }
  curandState *localRandState = curandSrcPtr;

  for (; threadId < transitToSamplesSize; threadId += gridDim.x * blockDim.x)
  {
    // for (int threadId0 = 0; threadId0 < transitToSamplesSize; threadId0 += gridDim.x * blockDim.x) {
    //   threadId = threadId0 + threadIdx.x + blockDim.x * blockIdx.x;
    //__shared__ VertexID newNeigbhors[N_THREADS];
    EdgePos_t transitIdx;
    EdgePos_t transitNeighborIdx;
    VertexID_t transit;
    int kernelTy;
    bool continueExecution = true;

    continueExecution = threadId < transitToSamplesSize;

    int subWarpSize = subWarpSizeAtStep(fanouts[step]);
    transitIdx = threadId / subWarpSize;
    transitNeighborIdx = threadId % subWarpSize;
    if (continueExecution && transitNeighborIdx == 0)
    {
      transit = transitToSamplesKeys[transitIdx];
      kernelTy = kernelTypeForTransit[transit];
    }

    transit = __shfl_sync(FULL_WARP_MASK, transit, 0, subWarpSize);
    kernelTy = __shfl_sync(FULL_WARP_MASK, kernelTy, 0, subWarpSize);

    continueExecution = continueExecution && transitNeighborIdx < fanouts[step];

    if ((useGridKernel && kernelTy == TransitKernelTypes::GridKernel && numTransits > 1) ||
        (useSubWarpKernel && kernelTy == TransitKernelTypes::SubWarpKernel && numTransits > 1) ||
        (useThreadBlockKernel && kernelTy == TransitKernelTypes::ThreadBlockKernel && numTransits > 1))
    {
      // printf("diff kernel: %ld, transit: %ld, pos: %ld, kernelTy: %d\n", threadId, transit, transitIdx * fanouts[step] + transitNeighborIdx, kernelTy);
      continueExecution = false;
    }

    BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];

    VertexID_t sampleIdx = -1;

    if (continueExecution && transitNeighborIdx == 0)
    {
      sampleIdx = transitToSamplesValues[transitIdx];
    }

    sampleIdx = __shfl_sync(FULL_WARP_MASK, sampleIdx, 0, subWarpSize);
    VertexID_t neighbor = invalidVertex;
    EdgePos_t neighbor_pos = 0;
    // assert(sampleIdx < NumSamples);

    if (continueExecution and transit != invalidVertex and transit < bcg->n_nodes)
    {
      BCGVertex bcgv(transit, bcg->graph, bcg->offset[transit]);
      EdgePos_t numTransitEdges = bcgv.outd;

      if (numTransitEdges != 0)
      {
        neighbor = CCGApp().next(step, &transit, sampleIdx, numTransitEdges, transitNeighborIdx, localRandState, &bcgv, neighbor_pos);
        neighbor_pos += bcg->degoffset[transit];
      }
    }
    // printf("Thread %d: %ld %ld %ld\n", threadId, transit, neighbor_pos, neighbor);
    __syncwarp();

    if (continueExecution)
    {
      if (step != tot_step - 1)
      {
        EdgePos_t pos = transitIdx * fanouts[step] + transitNeighborIdx;
        // No need to store at last step
        if (HAS_EXPLICIT_TRANSITS)
        {
          VertexID_t newTransit = CCGApp().stepTransits(step + 1, sampleIdx, samples[sampleIdx - deviceFirstSample], transitIdx, localRandState);
          samplesToTransitValues[threadId] = newTransit != -1 ? newTransit : invalidVertex;
        }
        else
        {
          // samplesToTransitValues[threadId] = neighbor != -1 ? neighbor : invalidVertex;
          samplesToTransitValues[pos] = neighbor != -1 ? neighbor : invalidVertex;
        }
        samplesToTransitKeys[pos] = sampleIdx;
      }
    }

    __syncwarp();

    
    EdgePos_t insertionPos = 0;
    if (CCGApp().samplingType() == SamplingType::NeighborSampling)
    {
      EdgePos_t finalSampleSizeTillPreviousStep = 0;
      EdgePos_t neighborsToSampleAtStep = 1;
      // FIXME: in deepwalk if there is an invalid vertex at step k, it will not store the
      // transits of step k -1 due to coalescing the stores.
      if (step == 0)
      {
        insertionPos = transitNeighborIdx;
      }
      else
      {
        for (int _s = 0; _s < step; _s++)
        {
          neighborsToSampleAtStep *= fanouts[_s];
          finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
        }
        EdgePos_t insertionStartPosForTransit = 0;

        if (threadIdx.x % subWarpSize == 0)
        {
          insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx - deviceFirstSample], fanouts[step]);
        }
        insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
        // insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
        insertionPos = insertionStartPosForTransit + transitNeighborIdx;
      }
      __syncwarp();
    }

    if (continueExecution)
    {
      if (isValidSampledVertex(neighbor, invalidVertex))
      {
        // printf("%ld -(%ld)-> %ld\n", transit, neighbor, neighbor_pos);
        // printf("Pos %ld\n", (sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * fanouts[step] + insertionPos);
        if (CCGApp().samplingType() == SamplingType::NeighborSampling)
        {
          insertionPos = transitIdx * fanouts[step] + transitNeighborIdx;
          // insertionPos = (sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * fanouts[step] + insertionPos;
          // printf("Thread %d: sample %ld transit %ld out[%ld] = %ld\n", threadId, sampleIdx, transitIdx, insertionPos, neighbor);
          // assert(transit!=-1 && transit != invalidVertex);
          // assert(neighbor!=-1 && neighbor != invalidVertex);
          // assert(transit != 0 || neighbor != 0);
          out_rows[insertionPos] = neighbor != -1 ? transit : invalidVertex;
          out_cols[insertionPos] = neighbor != -1 ? neighbor : invalidVertex;
          out_idxs[insertionPos] = neighbor != -1 ? neighbor_pos : -1;
        }
        else if (CCGApp().samplingType() == SamplingType::RandomWalkSampling)
        {
          insertionPos = (sampleIdx - deviceFirstSample) * trace_length + step + 1;
          out_trace[insertionPos] = neighbor != -1 ? neighbor : invalidVertex;
        }
      }
    }
  }

  //Write back the updated curand states
  if (COALESCE_CURAND_LOAD)
  {
    const int intsInRandState = sizeof(curandState) / sizeof(int);
    int* shStateBuffAsInts = (int*)&shMemCuRand[0];
    int* randStatesAsInts = (int*)randStates;
  
    for (int i = threadIdx.x; i < intsInRandState * blockDim.x; i += blockDim.x)
    {
      randStatesAsInts[i + blockDim.x*blockIdx.x] = shStateBuffAsInts[i];
    }
  }
}

template <typename CCGApp, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, int TRANSITS_PER_THREAD, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE, int SUB_WARP_SIZE>
__global__ void threadBlockKernel(const int step, const VertexID_t deviceFirstSample,
                                  const VertexID_t invalidVertex,
                                  const VertexID_t *transitToSamplesKeys, const VertexID_t *transitToSamplesValues,
                                  const size_t transitToSamplesSize, CCGSample *samples, const size_t NumSamples,
                                  VertexID_t *samplesToTransitKeys, VertexID_t *samplesToTransitValues,
                                  VertexID_t *finalSamples, const size_t finalSampleSize, EdgePos_t *sampleInsertionPositions,
                                  curandState *randStates, const int *kernelTypeForTransit, const VertexID_t *threadBlockKernelPositions,
                                  const EdgePos_t threadBlockKernelPositionsNum, int totalThreadBlocks,
                                  int numTransitsAtStepPerSample, int finalSampleSizeTillPreviousStep, int64_t *fanouts,
                                  int tot_step, VertexID_t *out_rows, VertexID_t *out_cols, EdgePos_t *out_idxs,
                                  VertexID_t *out_trace, int64_t trace_length)
{
// if (threadIdx.x == 0 && blockIdx.x == 0) printf("TBK %d\n", step);
#define EDGE_CACHE_SIZE (CACHE_EDGES ? CACHE_SIZE : 0)
#define CURAND_SHMEM_SIZE (sizeof(curandState) * THREADS)
#define NUM_THREAD_GROUPS (THREADS / LoadBalancing::LoadBalancingThreshold::BlockLevel)

  union unionShMem
  {
    struct
    {
      Graph_t graphCache[EDGE_CACHE_SIZE * NUM_THREAD_GROUPS];
      EdgePos_t mapStartPos[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD];
      EdgePos_t subWarpTransits[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD][LoadBalancing::LoadBalancingThreshold::BlockLevel / SUB_WARP_SIZE];
      EdgePos_t subWarpSampleIdx[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD][LoadBalancing::LoadBalancingThreshold::BlockLevel / SUB_WARP_SIZE];
      VertexID_t transitVertices[NUM_THREAD_GROUPS][TRANSITS_PER_THREAD];
    };
    unsigned char shMemAlloc[sizeof(curandState) * THREADS];
  };
  __shared__ unionShMem shMem;
  
  // clock_t start = clock();
  // CSR::Edge* edgesInShMem = CACHE_EDGES ? (CSR::Edge*)(&shMem.edgeAndWeightCache[0] + EDGE_CACHE_SIZE*(threadIdx.x/LoadBalancing::LoadBalancingThreshold::BlockLevel)) : nullptr;
  // float* edgeWeightsInShMem = CACHE_WEIGHTS ? (float*)&shMem.edgeAndWeightCache[EDGE_CACHE_SIZE] : nullptr;
  // Graph_t *graphInShMem = CACHE_EDGES ? shMem.graphCache + EDGE_CACHE_SIZE * (threadIdx.x / LoadBalancing::LoadBalancingThreshold::BlockLevel) : nullptr;

  const int stepSize = fanouts[step];
  curandState *curandSrcPtr;

  const int subWarpSize = SUB_WARP_SIZE;

  const int intsInRandState = sizeof(curandState) / sizeof(int);
  int *shStateBuff = (int *)&shMem.shMemAlloc[0];

  int *randStatesAsInts = (int *)randStates;

  // Load curand only for the number of threads that are going to do sampling in this warp
  for (int i = threadIdx.x; i < intsInRandState * (blockDim.x / subWarpSize) * stepSize; i += blockDim.x)
  {
    shStateBuff[i] = randStatesAsInts[i + blockDim.x * blockIdx.x];
  }

  __syncthreads();
  if (threadIdx.x % subWarpSize < stepSize)
  {
    // Load curand only for the threads that are going to do sampling.
    curandSrcPtr = (curandState *)(&shStateBuff[threadIdx.x * intsInRandState]);
  }

  curandState localRandState = (threadIdx.x % subWarpSize < stepSize) ? *curandSrcPtr : curandState();

  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];

  // clock_t init_sh = clock();
  // if(!threadIdx.x && !blockIdx.x) printf("init_sh %lf\n", ((double)init_sh - start)/CLOCKS_PER_SEC);
  // int cnt=0;
  for (int fullBlockIdx = blockIdx.x; fullBlockIdx < totalThreadBlocks; fullBlockIdx += gridDim.x)
  {
    // cnt++;
    // if(!blockIdx.x && !threadIdx.x) printf("%d next block: %d\n", cnt, fullBlockIdx);
    EdgePos_t transitIdx = 0;
    static_assert(NUM_THREAD_GROUPS * TRANSITS_PER_THREAD <= THREADS);
    int fullWarpIdx = (threadIdx.x + fullBlockIdx * blockDim.x) / LoadBalancing::LoadBalancingThreshold::BlockLevel;

    if (threadIdx.x < NUM_THREAD_GROUPS * TRANSITS_PER_THREAD)
    {
      const int warpIdx = threadIdx.x / TRANSITS_PER_THREAD;
      const int transitIdx = threadIdx.x % TRANSITS_PER_THREAD;
      const int __fullWarpIdx = warpIdx + (fullBlockIdx * blockDim.x) / LoadBalancing::LoadBalancingThreshold::BlockLevel;

      if (TRANSITS_PER_THREAD * __fullWarpIdx + transitIdx < threadBlockKernelPositionsNum)
        shMem.mapStartPos[warpIdx][transitIdx] = threadBlockKernelPositions[TRANSITS_PER_THREAD * __fullWarpIdx + transitIdx];
      else
        shMem.mapStartPos[warpIdx][transitIdx] = -1;
    }

    __syncthreads();

    const int NUM_SUBWARPS_IN_TB = NUM_THREAD_GROUPS * (LoadBalancing::LoadBalancingThreshold::BlockLevel / SUB_WARP_SIZE);
    static_assert(NUM_SUBWARPS_IN_TB * TRANSITS_PER_THREAD <= THREADS);

    if (threadIdx.x < NUM_SUBWARPS_IN_TB * TRANSITS_PER_THREAD)
    {
      // Coalesce loads of transits per sub-warp by loading transits for all sub-warps in one warp.
      // FIXME: Fix this when SUB_WARP_SIZE < 32
      int subWarpIdx = threadIdx.x / TRANSITS_PER_THREAD;
      // the I-th transit of current thread
      int transitI = threadIdx.x % TRANSITS_PER_THREAD;
      transitIdx = shMem.mapStartPos[subWarpIdx][transitI];
      // TODO: Specialize this for subWarpSize = 1.
      VertexID_t transit = invalidVertex;
      if (transitIdx != -1)
      {
        transit = transitToSamplesKeys[transitIdx];
        shMem.subWarpSampleIdx[subWarpIdx][transitI][0] = transitToSamplesValues[transitIdx];
      }
      shMem.subWarpTransits[subWarpIdx][transitI][0] = transit;
    }
    __syncthreads();

    if (threadIdx.x < NUM_SUBWARPS_IN_TB * TRANSITS_PER_THREAD)
    {
      // Load transit Vertex Object in a coalesced manner
      // TODO: Fix this for subwarpsize < 32
      int transitI = threadIdx.x % TRANSITS_PER_THREAD;
      int subWarpIdx = threadIdx.x / TRANSITS_PER_THREAD;
      VertexID_t transit = shMem.subWarpTransits[subWarpIdx][transitI][0];
      if (transit != invalidVertex)
        shMem.transitVertices[subWarpIdx][transitI] = transit;
    }
    __syncthreads();

    // clock_t load_transit = clock();
    // if(!threadIdx.x && !blockIdx.x) printf("load_transit %lf\n", ((double)load_transit - init_sh)/CLOCKS_PER_SEC);
    for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++)
    {
      int threadBlockWarpIdx = threadIdx.x / subWarpSize;
      // TODO: Support this for SubWarp != 32

      if (TRANSITS_PER_THREAD * fullWarpIdx + transitI >= threadBlockKernelPositionsNum)
        continue;

      __syncwarp(); // TODO: Add mask based on subwarp
      VertexID_t transit = -1;
      bool invalidateCache = false;
      if (threadIdx.x % subWarpSize == 0)
      {
        invalidateCache = shMem.subWarpTransits[threadBlockWarpIdx][transitI][0] != transit || transitI == 0;
      }

      invalidateCache = __shfl_sync(FULL_WARP_MASK, invalidateCache, 0, subWarpSize);

      transit = shMem.subWarpTransits[threadBlockWarpIdx][transitI][0];
      if (transit == invalidVertex)
        continue;

      __syncwarp();

      VertexID_t shMemTransitVertex = shMem.transitVertices[threadBlockWarpIdx][transitI];
      BCGVertex bcgv(shMemTransitVertex, bcg->graph, bcg->offset[shMemTransitVertex]);
      EdgePos_t numEdgesInShMem = bcgv.outd;

      bool continueExecution = true;

      if (subWarpSize == 32) // true
      {
        assert(transit == shMemTransitVertex);
        // A thread will run next only when it's transit is same as transit of the threadblock.
        // transitIdx = shMem.mapStartPos[threadBlockWarpIdx][transitI] + threadIdx.x / subWarpSize; // threadId/stepSize(step);
        transitIdx = shMem.mapStartPos[threadBlockWarpIdx][transitI];
        // assert(transitIdx < transitToSamplesSize);
        // if(transitIdx >= transitToSamplesSize) printf("%d tranist id %ld >= %ld %d\n", threadIdx.x, transitIdx, transitToSamplesSize, threadIdx.x / subWarpSize);
        VertexID_t transitNeighborIdx = threadIdx.x % subWarpSize;
        VertexID_t sampleIdx = shMem.subWarpSampleIdx[threadBlockWarpIdx][transitI][0];

        continueExecution = (transitNeighborIdx < stepSize) && (transitIdx != -1);

        VertexID_t neighbor = invalidVertex;
        EdgePos_t neighbor_pos = 0;
        if (numEdgesInShMem > 0 && continueExecution)
        {
          neighbor = CCGApp().next(step, &transit, sampleIdx, numEdgesInShMem, transitNeighborIdx, &localRandState, &bcgv, neighbor_pos);
          neighbor_pos += bcg->degoffset[transit];
        }

        // clock_t done_next = clock();
        // if(!threadIdx.x && !blockIdx.x) printf("done_next %lf\n", ((double)done_next - load_transit)/CLOCKS_PER_SEC);
        if (continueExecution)
        {
          int pos = transitIdx * fanouts[step] + transitNeighborIdx;
          if (step != tot_step - 1)
          {
            // No need to store at last step
            samplesToTransitKeys[pos] = sampleIdx; // TODO: Update this for khop to transitIdx + transitNeighborIdx
            if (CCGApp().hasExplicitTransits())
            {
              VertexID_t newTransit = CCGApp().stepTransits(step, sampleIdx, samples[sampleIdx - deviceFirstSample], transitIdx, &localRandState);
              // samplesToTransitValues[transitIdx] = newTransit != -1 ? newTransit : invalidVertex;
              samplesToTransitValues[pos] = newTransit != -1 ? newTransit : invalidVertex;
            }
            else
            {
              // samplesToTransitValues[transitIdx] = neighbor != -1 ? neighbor : invalidVertex;
              samplesToTransitValues[pos] = neighbor != -1 ? neighbor : invalidVertex;
            }
          }
        }

        EdgePos_t insertionPos = transitNeighborIdx;

        if (step == 0)
          insertionPos = transitNeighborIdx;
        else
        {
          EdgePos_t insertionStartPosForTransit = 0;
          // FIXME:
          if (isValidSampledVertex(neighbor, invalidVertex) && threadIdx.x % subWarpSize == 0)
          {
            insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx - deviceFirstSample], stepSize);
          }
          insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
          // insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
          insertionPos = insertionStartPosForTransit + transitNeighborIdx;
          // printf("%ld -(%ld)-> %ld -- %ld %ld\n", transit, neighbor, neighbor_pos, insertionStartPosForTransit, transitNeighborIdx);
        }

        if (continueExecution)
          assert(insertionPos < finalSampleSize);

        // clock_t done_insert = clock();
        // if(!threadIdx.x && !blockIdx.x) printf("done_next %lf\n", ((double)done_insert - done_next)/CLOCKS_PER_SEC);

        if (continueExecution && isValidSampledVertex(neighbor, invalidVertex))
        {
          if (CCGApp().samplingType() == SamplingType::NeighborSampling)
          {
            // size_t neighborsToSampleAtStep = 1;
            // for (int _s = 0; _s < step; _s++)
            // {
            //   neighborsToSampleAtStep *= fanouts[_s];
            // }
            // printf("%ld -(%ld)-> %ld\n", transit, neighbor, neighbor_pos);
            // if ((sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * stepSize + insertionPos >= 20 * 5 * 20)
            // printf("%ld %d Pos %ld\n", insertionPos, transitNeighborIdx, (sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * stepSize + insertionPos);
            // printf("P=%ld  ", (sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * stepSize + insertionPos);
            // printf("P=%ldx%ld B=%do%d ", sampleIdx, insertionPos, blockIdx.x, threadIdx.x);
            // insertionPos = (sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * stepSize + insertionPos;
            insertionPos = transitIdx * fanouts[step] + transitNeighborIdx;
            // assert(insertionPos < transitToSamplesSize);
            if(insertionPos >= transitToSamplesSize) printf("sample %ld transit(id: %ld) %ld out[%ld] = %ld\n", sampleIdx, transitIdx, transit, insertionPos, neighbor);
            assert(neighbor != -1);
            assert(transit!=-1);
            out_rows[insertionPos] = transit;
            out_cols[insertionPos] = neighbor;
            out_idxs[insertionPos] = neighbor_pos;
          }
          else if (CCGApp().samplingType() == SamplingType::RandomWalkSampling)
          {
            insertionPos = (sampleIdx - deviceFirstSample) * trace_length + step + 1;
            out_trace[insertionPos] = neighbor;
          }
        }
        // clock_t done_out = clock();
        // if(!threadIdx.x && !blockIdx.x) printf("done_out %lf\n", ((double)done_out - done_insert)/CLOCKS_PER_SEC);
      }
    }
    
  }
}

template <typename CCGApp, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool COALESCE_GL_LOADS, int TRANSITS_PER_THREAD,
          bool COALESCE_CURAND_LOAD, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE, int SUB_WARP_SIZE>
__global__ void gridKernel(const int step, const VertexID_t deviceFirstSample,
                            const VertexID_t invalidVertex,
                            const VertexID_t *transitToSamplesKeys, const VertexID_t *transitToSamplesValues,
                            const size_t transitToSamplesSize, CCGSample *samples, const size_t NumSamples,
                            VertexID_t *samplesToTransitKeys, VertexID_t *samplesToTransitValues,
                            VertexID_t *finalSamples, const size_t finalSampleSize, EdgePos_t *sampleInsertionPositions,
                            curandState *randStates, const int *kernelTypeForTransit, const VertexID_t *gridKernelTBPositions,
                            const EdgePos_t gridKernelTBPositionsNum, int totalThreadBlocks, int numTransitsPerStepForSample,
                            int finalSampleSizeTillPreviousStep, int64_t *fanouts, int tot_step,
                            VertexID_t *out_rows, VertexID_t *out_cols, EdgePos_t *out_idxs, VertexID_t *out_trace, int64_t trace_length) 
{
  // if (threadIdx.x == 0 && blockIdx.x == 0) printf("GK %d\n", step);

#define EDGE_CACHE_SIZE (CACHE_EDGES ? CACHE_SIZE : 0)
#define WEIGHT_CACHE_SIZE (CACHE_WEIGHTS ? CACHE_SIZE * sizeof(float) : 0)
#define CURAND_SHMEM_SIZE (sizeof(curandState) * THREADS)
  // #define COALESCE_GL_LOADS_SHMEM_SIZE ()

  union unionShMem
  {
    struct
    {
      // unsigned char edgeAndWeightCache[EDGE_CACHE_SIZE+WEIGHT_CACHE_SIZE];
      Graph_t graphCache[EDGE_CACHE_SIZE];
      bool invalidateCache;
      VertexID_t transitForTB;
      EdgePos_t mapStartPos[TRANSITS_PER_THREAD];
      EdgePos_t subWarpTransits[TRANSITS_PER_THREAD][THREADS / SUB_WARP_SIZE];
      EdgePos_t subWarpSampleIdx[TRANSITS_PER_THREAD][THREADS / SUB_WARP_SIZE];
      VertexID_t transitVertices[TRANSITS_PER_THREAD];
      // unsigned char transitVertices[TRANSITS_PER_THREAD*sizeof(CSR::Vertex)];
    };
    unsigned char shMemAlloc[sizeof(curandState) * THREADS];
  };
  __shared__ unionShMem shMem;

  //__shared__ bool globalLoadBV[COALESCE_GL_LOADS ? CACHE_SIZE : 1];

  // CSR::Edge* edgesInShMem = CACHE_EDGES ? (CSR::Edge*)&shMem.edgeAndWeightCache[0] : nullptr;
  // float* edgeWeightsInShMem = CACHE_WEIGHTS ? (float*)&shMem.edgeAndWeightCache[EDGE_CACHE_SIZE] : nullptr;
  // Graph_t *graphInShMem = CACHE_EDGES ? shMem.graphCache : nullptr;
  // Graph_t* graphInShMem = CACHE_EDGES ? shMem.graphCache + EDGE_CACHE_SIZE * (threadIdx.x / LoadBalancing::LoadBalancingThreshold::BlockLevel) : nullptr;

  int threadId = threadIdx.x + blockDim.x * blockIdx.x;

  curandState *curandSrcPtr;
  const int stepSize = fanouts[step];

  const int subWarpSize = SUB_WARP_SIZE;

  if (COALESCE_CURAND_LOAD)
  {
    const int intsInRandState = sizeof(curandState) / sizeof(int);
    int *shStateBuff = (int *)&shMem.shMemAlloc[0];

    int *randStatesAsInts = (int *)randStates;

    // Load curand only for the number of threads that are going to do sampling in this warp
    for (int i = threadIdx.x; i < intsInRandState * (blockDim.x / subWarpSize) * stepSize; i += blockDim.x)
    {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x * blockIdx.x];
    }

    __syncthreads();
    if (threadIdx.x % subWarpSize < stepSize)
    {
      // Load curand only for the threads that are going to do sampling.
      //  int ld = threadIdx.x - (threadIdx.x/subWarpSize)*(subWarpSize-stepSize);
      curandSrcPtr = (curandState *)(&shStateBuff[threadIdx.x * intsInRandState]);
    }
  }
  else
  {
    curandSrcPtr = &randStates[threadId];
  }

  curandState localRandState = (threadIdx.x % subWarpSize < stepSize) ? *curandSrcPtr : curandState();

  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];

  for (int fullBlockIdx = blockIdx.x; fullBlockIdx < totalThreadBlocks; fullBlockIdx += gridDim.x)
  {

    EdgePos_t transitIdx = 0;
    if (threadIdx.x < TRANSITS_PER_THREAD)
    {
      if (TRANSITS_PER_THREAD * fullBlockIdx + threadIdx.x < gridKernelTBPositionsNum)
      {
        shMem.mapStartPos[threadIdx.x] = gridKernelTBPositions[TRANSITS_PER_THREAD * fullBlockIdx + threadIdx.x];
      }
      else
      {
        shMem.mapStartPos[threadIdx.x] = -1;
      }
    }

    __syncthreads();
    if (threadIdx.x < THREADS / SUB_WARP_SIZE * TRANSITS_PER_THREAD)
    {
      // Coalesce loads of transits per sub-warp by loading transits for all sub-warps in one warp.
      //  Assign THREADS/SUB_WARP_SIZE threads to each Transit in TRANSITS_PER_THREAD
      //  static_assert ((THREADS/SUB_WARP_SIZE * TRANSITS_PER_THREAD) < THREADS);
      int transitI = threadIdx.x / (THREADS / SUB_WARP_SIZE); // * TRANSITS_PER_THREAD);
      transitIdx = shMem.mapStartPos[transitI] + threadIdx.x % (THREADS / SUB_WARP_SIZE);
      // if (!(transitIdx >= 0 && transitIdx < 57863 * 10)) {
      //   printf("transitIdx %d shMem.mapStartPos[transitI] %d\n", transitIdx, shMem.mapStartPos[transitI]);
      // }
      // TODO: Specialize this for subWarpSizez = 1.
      VertexID_t transit = invalidVertex;
      if (transitIdx != -1)
      {
        transit = transitToSamplesKeys[transitIdx];
        shMem.subWarpSampleIdx[transitI][threadIdx.x % (THREADS / SUB_WARP_SIZE)] = transitToSamplesValues[transitIdx];
      }
      shMem.subWarpTransits[transitI][threadIdx.x % (THREADS / SUB_WARP_SIZE)] = transit;
    }

    __syncthreads();
    if (threadIdx.x < TRANSITS_PER_THREAD)
    {
      // Load Transit Vertex of first subwarp in a Coalesced manner
      VertexID_t transit = shMem.subWarpTransits[threadIdx.x][0];
      if (transit != invalidVertex)
        shMem.transitVertices[threadIdx.x] = transit;
    }
    __syncwarp();

    for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++)
    {
      if (TRANSITS_PER_THREAD * (fullBlockIdx) + transitI >= gridKernelTBPositionsNum)
        continue;

      __syncthreads();

      VertexID_t transit = shMem.subWarpTransits[transitI][threadIdx.x / subWarpSize];

      if (threadIdx.x == 0)
      {
        shMem.invalidateCache = shMem.transitForTB != transit || transitI == 0;
        shMem.transitForTB = transit;
      }
      __syncthreads();

      VertexID_t shMemTransitVertex = shMem.transitForTB;

      // assert(4847571 >= shMemTransitVertex);
      BCGVertex bcgv(shMemTransitVertex, bcg->graph, bcg->offset[shMemTransitVertex]);
      EdgePos_t numEdgesInShMem = bcgv.outd;

      // if (CACHE_EDGES && shMem.invalidateCache) {
      //   VertexID_t read_num = bcgv.get_sm_num(CACHE_SIZE, graphInShMem);
      //   // printf("%d Trn=%d rn=%d %p--%p\n", threadIdx.x, shMemTransitVertex, read_num, bcg->graph, bcgv.graph);
      //   for (int i = threadIdx.x; i < read_num; i += blockDim.x)
      //     graphInShMem[i] = bcgv.graph[i];
      // }

      // __syncthreads();
      bool continueExecution = true;

      if (transit == shMem.transitForTB)
      {
        // A thread will run next only when it's transit is same as transit of the threadblock.
        transitIdx = shMem.mapStartPos[transitI] + threadIdx.x / subWarpSize; // threadId/stepSize(step);
        VertexID_t transitNeighborIdx = threadIdx.x % subWarpSize;
        VertexID_t sampleIdx = shMem.subWarpSampleIdx[transitI][threadIdx.x / subWarpSize];
        ;
        // if (threadIdx.x % subWarpSize == 0) {
        //   printf("1271: sampleIdx %d transit %d transitForTB %d numEdgesInShMem %d threadIdx.x %d blockIdx.x %d fullBlockIdx %d\n", sampleIdx, transit, shMem.transitForTB, numEdgesInShMem, threadIdx.x, blockIdx.x, fullBlockIdx);
        // }
        // if (threadIdx.x % subWarpSize == 0) {
        //   sampleIdx = transitToSamplesValues[transitIdx];
        // }

        // sampleIdx = __shfl_sync(FULL_WARP_MASK, sampleIdx, 0, subWarpSize);

        continueExecution = (transitNeighborIdx < stepSize);
        // if (threadIdx.x == 0 && kernelTypeForTransit[transit] != TransitKernelTypes::GridKernel) {
        //   printf("transit %d transitIdx %d gridDim.x %d\n", transit, transitIdx, gridDim.x);
        // }
        // assert (kernelTypeForTransit[transit] == TransitKernelTypes::GridKernel);

        VertexID_t neighbor = invalidVertex;
        EdgePos_t neighbor_pos = 0;
        // if (graph.device_csr->has_vertex(transit) == false)
        //   printf("transit %d\n", transit);

        if (numEdgesInShMem > 0 && continueExecution)
        {
          neighbor = CCGApp().next(step, &transit, sampleIdx, numEdgesInShMem, transitNeighborIdx, &localRandState, &bcgv, neighbor_pos);
          neighbor_pos += bcg->degoffset[transit];
          // printf("%ld -%ld-> %ld\n", transit, neighbor_pos, neighbor);
        }

        if (continueExecution)
        {
          EdgePos_t pos = transitIdx * fanouts[step] + transitNeighborIdx;
          if (step != tot_step - 1)
          {
            // No need to store at last step
            // samplesToTransitKeys[transitIdx] = sampleIdx; // TODO: Update this for khop to transitIdx + transitNeighborIdx
            samplesToTransitKeys[pos] = sampleIdx;
            if (CCGApp().hasExplicitTransits())
            {
              VertexID_t newTransit = CCGApp().stepTransits(step, sampleIdx, samples[sampleIdx - deviceFirstSample], transitIdx, &localRandState);
              samplesToTransitValues[transitIdx] = newTransit != -1 ? newTransit : invalidVertex;
            }
            else
            {
              // samplesToTransitValues[transitIdx] = neighbor != -1 ? neighbor : invalidVertex;
              samplesToTransitValues[pos] = neighbor != -1 ? neighbor : invalidVertex;
            }
          }
        }

        EdgePos_t insertionPos = transitNeighborIdx;

        if (step == 0)
        {
          insertionPos = transitNeighborIdx;
        }
        else
        {
          EdgePos_t insertionStartPosForTransit = 0;
          // FIXME:
          if (isValidSampledVertex(neighbor, invalidVertex) && threadIdx.x % subWarpSize == 0)
          {
            insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx - deviceFirstSample], stepSize);
          }
          insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
          // insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
          insertionPos = insertionStartPosForTransit + transitNeighborIdx;
          // if (continueExecution && insertionPos >= finalSampleSize) {
          //   printf("%d>%ld %d+%d(%d %d)+%d\n", insertionPos, finalSampleSize, finalSampleSizeTillPreviousStep, insertionStartPosForTransit,  stepSize, subWarpSize, transitNeighborIdx);
          // }
        }

        if (continueExecution)
        {
          assert(insertionPos < finalSampleSize);
        }

        if (isValidSampledVertex(neighbor, invalidVertex))
        {
          if (CCGApp().samplingType() == SamplingType::NeighborSampling)
          {
            // size_t neighborsToSampleAtStep = 1;
            // for (int _s = 0; _s < step; _s++)
            // {
            //   neighborsToSampleAtStep *= fanouts[_s];
            // }
            // printf("%ld -(%ld)-> %ld\n", transit, neighbor, neighbor_pos);
            // printf("gkPos %ld\n", (sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * stepSize + insertionPos);
            insertionPos = transitIdx * fanouts[step] + transitNeighborIdx;
            // insertionPos = (sampleIdx - deviceFirstSample) * neighborsToSampleAtStep * stepSize + insertionPos
            out_rows[insertionPos] = neighbor != -1 ? transit : invalidVertex;
            out_cols[insertionPos] = neighbor != -1 ? neighbor : invalidVertex;
            out_idxs[insertionPos] = neighbor != -1 ? neighbor_pos : -1;
          }
          else if (CCGApp().samplingType() == SamplingType::RandomWalkSampling)
          {
            insertionPos = (sampleIdx - deviceFirstSample) * trace_length + step + 1;
            out_trace[insertionPos] = neighbor != -1 ? neighbor : invalidVertex;
          }
        }
        // }
        // TODO: We do not need atomic instead store indices of transit in another array,
        // wich can be accessed based on sample and transitIdx.
      }
    }
  }
}


__global__ void setFristNode(VertexID_t* trace, VertexID_t* seeds, int length) {
  VertexID_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  trace[threadId * length] = seeds[threadId];
}

template <typename CCGApp>
void doTransitParallelSampling(NextDoorData &nextDoorData, const DGLContext &ctx, bool enableLoadBalancing = true)
{
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  int tot_step = CCGApp().steps();
  const auto maxNeighborsToSample = nextDoorData.maxNeighborsToSample;

  const size_t numDevices = nextDoorData.devices.size();

  d_fanouts = static_cast<int64_t *>(device->AllocWorkspace(ctx, sizeof(int64_t) * fanouts.size()));
  device->CopyDataFromTo(fanouts.data(), 0, d_fanouts, 0, sizeof(int64_t) * fanouts.size(), DGLContext{kDGLCPU, 0}, ctx, DGLDataType{kDGLInt, 64, 1});
  size_t finalSampleSize = CCGApp().getFinalSampleSize();

  std::vector<VertexID_t *> d_temp_storage = std::vector<VertexID_t *>(nextDoorData.devices.size());
  std::vector<size_t> temp_storage_bytes = std::vector<size_t>(nextDoorData.devices.size());
  std::vector<VertexID_t *> d_unique_temp_storage = std::vector<VertexID_t *>(nextDoorData.devices.size());
  std::vector<VertexID_t *> d_num_selected_out = std::vector<VertexID_t *>(nextDoorData.devices.size());
  std::vector<size_t> unique_temp_storage_bytes = std::vector<size_t>(nextDoorData.devices.size());
  

  // unique transits in dTransitToSampleMapKeys
  std::vector<VertexID_t *> dUniqueTransits = std::vector<VertexID_t *>(nextDoorData.devices.size());
  // counts of unique transits in dTransitToSampleMapKeys
  std::vector<VertexID_t *> dUniqueTransitsCounts = std::vector<VertexID_t *>(nextDoorData.devices.size());
  // number of unique tranists in dTransitToSampleMapKeys
  std::vector<EdgePos_t *> uniqueTransitNumRuns = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  // number of unique tranists in dTransitToSampleMapKeys
  std::vector<EdgePos_t *> dUniqueTransitsNumRuns = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  // start postion of unique transits in dTransitToSampleMapKeys
  std::vector<EdgePos_t *> dTransitPositions = std::vector<EdgePos_t *>(nextDoorData.devices.size());

  /**Pointers for each kernel type**/
  std::vector<EdgePos_t *> gridKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  std::vector<EdgePos_t *> dGridKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  std::vector<VertexID_t *> dGridKernelTransits = std::vector<VertexID_t *>(nextDoorData.devices.size());

  std::vector<EdgePos_t *> threadBlockKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  std::vector<EdgePos_t *> dThreadBlockKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  std::vector<VertexID_t *> dThreadBlockKernelTransits = std::vector<VertexID_t *>(nextDoorData.devices.size());

  std::vector<EdgePos_t *> subWarpKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  std::vector<EdgePos_t *> dSubWarpKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  std::vector<VertexID_t *> dSubWarpKernelTransits = std::vector<VertexID_t *>(nextDoorData.devices.size());

  std::vector<EdgePos_t *> identityKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  std::vector<EdgePos_t *> dIdentityKernelTransitsNum = std::vector<EdgePos_t *>(nextDoorData.devices.size());
  /**********************************/

  // /****Variables for Collective Transit Sampling***/
  // std::vector<EdgePos_t *> hSumNeighborhoodSizes = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);
  // std::vector<EdgePos_t *> dSumNeighborhoodSizes = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);
  // std::vector<EdgePos_t *> dSampleNeighborhoodPos = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);
  // std::vector<EdgePos_t *> dSampleNeighborhoodSizes = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);
  // std::vector<VertexID_t *> dCollectiveNeighborhoodCSRCols = std::vector<VertexID_t *>(nextDoorData.devices.size(), nullptr);
  // std::vector<EdgePos_t *> dCollectiveNeighborhoodCSRRows = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);

  std::vector<EdgePos_t *> dInvalidVertexStartPosInMap = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t *> invalidVertexStartPosInMap = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);

  /*Single Memory Location on both CPU and GPU for transferring
   *number of transits for all kernels */
  std::vector<EdgePos_t *> dKernelTransitNums = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t *> hKernelTransitNums = std::vector<EdgePos_t *>(nextDoorData.devices.size(), nullptr);
  const int NUM_KERNEL_TYPES = TransitKernelTypes::NumKernelTypes + 1;

  std::vector<int *> dKernelTypeForTransit = std::vector<int *>(nextDoorData.devices.size(), nullptr);

  for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
  {
    // auto device = nextDoorData.devices[deviceIdx];
    const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.sampleNum, deviceIdx, numDevices);
    // CHK_CU(cudaSetDevice(device));
    uniqueTransitNumRuns[deviceIdx] = new EdgePos_t;
    hKernelTransitNums[deviceIdx] = new EdgePos_t[NUM_KERNEL_TYPES];
    // CHK_CU(cudaMallocHost(&uniqueTransitNumRuns[deviceIdx], sizeof(EdgePos_t)));
    // CHK_CU(cudaMallocHost(&hKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t)));

    gridKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx];
    threadBlockKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 1;
    subWarpKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 2;
    identityKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 3;
    invalidVertexStartPosInMap[deviceIdx] = hKernelTransitNums[deviceIdx] + 4;
    // threadBlockKernelTransitsNum = hKernelTransitNums[3];

    dKernelTypeForTransit[deviceIdx] = static_cast<int *>(device->AllocWorkspace(ctx, sizeof(int) * nextDoorData.n_nodes));
    CUDA_CALL(cudaMemset(dKernelTypeForTransit[deviceIdx], 0, sizeof(int) * nextDoorData.n_nodes));
    dTransitPositions[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, sizeof(VertexID_t) * nextDoorData.n_nodes));
    dGridKernelTransits[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, sizeof(VertexID_t) * perDeviceNumSamples * maxNeighborsToSample));
    // std::cout << "perDeviceNumSamples " << perDeviceNumSamples << " maxNeighborsToSample " << maxNeighborsToSample << std::endl;
    if (useThreadBlockKernel)
    {
      dThreadBlockKernelTransits[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, sizeof(VertexID_t) * perDeviceNumSamples * maxNeighborsToSample));
    }

    if (useSubWarpKernel)
    {
      dSubWarpKernelTransits[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, sizeof(VertexID_t) * perDeviceNumSamples * maxNeighborsToSample));
    }

    dKernelTransitNums[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));

    CUDA_CALL(cudaMemset(dKernelTransitNums[deviceIdx], 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
    dGridKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx];
    dThreadBlockKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 1;
    dSubWarpKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 2;
    dIdentityKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 3;
    dInvalidVertexStartPosInMap[deviceIdx] = dKernelTransitNums[deviceIdx] + 4;

    // Check if the space runs out.
    // TODO: Use DoubleBuffer version that requires O(P) space.
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx],
                                    nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx],
                                    perDeviceNumSamples * maxNeighborsToSample));

    d_temp_storage[deviceIdx] = static_cast<VertexID_t*>(device->AllocWorkspace(ctx, temp_storage_bytes[deviceIdx]));
    
    d_num_selected_out[deviceIdx] = static_cast<VertexID_t*>(device->AllocWorkspace(ctx, sizeof(VertexID_t)));
    CUDA_CALL(cub::DeviceSelect::UniqueByKey(d_unique_temp_storage[deviceIdx], unique_temp_storage_bytes[deviceIdx],
                                    nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx],
                                    d_num_selected_out[deviceIdx], perDeviceNumSamples * maxNeighborsToSample));

    d_unique_temp_storage[deviceIdx] = static_cast<VertexID_t*>(device->AllocWorkspace(ctx, unique_temp_storage_bytes[deviceIdx]));

    CUDA_CALL(cudaMemset(nextDoorData.dSampleInsertionPositions[deviceIdx], 0, sizeof(VertexID_t) * perDeviceNumSamples));

    dUniqueTransits[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, (nextDoorData.n_nodes + 1) * sizeof(VertexID_t)));
    dUniqueTransitsCounts[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, (nextDoorData.n_nodes + 1) * sizeof(VertexID_t)));
    dUniqueTransitsNumRuns[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, sizeof(VertexID_t)));

    CUDA_CALL(cudaMemset(dUniqueTransitsCounts[deviceIdx], 0, (nextDoorData.n_nodes + 1) * sizeof(VertexID_t)));
    CUDA_CALL(cudaMemset(dUniqueTransitsNumRuns[deviceIdx], 0, sizeof(size_t)));
  }

  std::vector<size_t> totalTransits = std::vector<size_t>(nextDoorData.devices.size());

  size_t totNeighborsToSampleAtStep = CCGApp().initialSampleSize() * nextDoorData.sampleNum;
  size_t totEdgesToSampleAtStep = CCGApp().initialSampleSize() * nextDoorData.sampleNum;

  for (int step = 0; step < tot_step; ++step)
  {
    // std::cout<<"step "<<step<<std::endl;
    size_t numTransits = totNeighborsToSampleAtStep;
    std::vector<size_t> totalThreads = std::vector<size_t>(nextDoorData.devices.size());
    for (size_t i = 0; i < nextDoorData.devices.size(); i++)
    {
      // 暂时只在一个device上做
      totalThreads[i] = totEdgesToSampleAtStep;
      // const auto perDeviceNumSamples = PartDivisionSize(nextDoorData.sampleNum, i, numDevices);
      // totalThreads[i] = perDeviceNumSamples * neighborsToSampleAtStep;
    }

    if (tot_step == 1)
    {
      // FIXME: Currently a non-sorted Transit to Sample Map is passed to both TP and TP+LB.
      // Here, if there is only one step, a sorted map is passed.
      // Fix this to make sure a sorted map is always passed.
      for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
      {
        // Invert sample->transit map by sorting samples based on the transit vertices
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx],
                                        nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                        nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx],
                                        totalThreads[deviceIdx], 0, nextDoorData.maxBits));
        // CHK_CU(cudaGetLastError());
      }
      // CUDA_SYNC_DEVICE_ALL(nextDoorData);
    }

    totEdgesToSampleAtStep = totNeighborsToSampleAtStep * fanouts[step];
    // neighborsToSampleAtStep = neighborsToSampleAtStep * fanouts[step];
    const int64_t spl_len = totEdgesToSampleAtStep;

    CCGApp().initStepSample(spl_len, ctx);
    for (size_t i = 0; i < nextDoorData.devices.size(); i++)
    {
      // 暂时只在一个device上做
      totalThreads[i] = totEdgesToSampleAtStep;
      // const auto perDeviceNumSamples = PartDivisionSize(nextDoorData.sampleNum, i, numDevices);
      // totalThreads[i] = perDeviceNumSamples * totEdgesToSampleAtStep;
    }

    if ((step == 0 && tot_step > 1) || !enableLoadBalancing)
    {
      // auto _st = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
      // When not doing load balancing call baseline transit parallel
      for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
      {
        // auto device = nextDoorData.devices[deviceIdx];
        // CHK_CU(cudaSetDevice(device));
        // std::cout<<"totalThreads: "<<totalThreads[deviceIdx] << " trace_length: " << trace_length<<" tot_step: "<< tot_step<<std::endl;
        const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.sampleNum, deviceIdx, numDevices);
        for (unsigned long threadsExecuted = 0; threadsExecuted < totalThreads[deviceIdx]; threadsExecuted += nextDoorData.maxThreadsPerKernel[deviceIdx])
        {
          size_t currExecutionThreads = min((size_t)nextDoorData.maxThreadsPerKernel[deviceIdx], totalThreads[deviceIdx] - threadsExecuted);
          samplingKernel<CCGApp, TransitParallelMode::NextFuncExecution, 0><<<utils::thread_block_size(currExecutionThreads, N_THREADS), N_THREADS>>>(
              step, threadsExecuted, currExecutionThreads, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
              (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
              totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx], finalSampleSize,
              nullptr, nullptr, nullptr, nullptr, nextDoorData.dCurandStates[deviceIdx], (int)tot_step, d_fanouts, out_rows, out_cols, out_idxs, out_trace, trace_length);
          // CUDA_CALL(cudaGetLastError());
        }
      }
      // CUDA_SYNC_DEVICE_ALL(nextDoorData);
      device->StreamSync(ctx, stream);
      // auto _ed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
      // std::cout<<"finish samplingKernel "<< std::fixed << std::setprecision(7) << (double)((_ed.count() - _st.count()) * 0.000001) << std::endl;
    }
    else
    {
      if (0)
      {
        // for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        // {
        //   auto device = nextDoorData.devices[deviceIdx];
        //   CHK_CU(cudaSetDevice(device));
        //   CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions[deviceIdx], 0, sizeof(VertexID_t) * nextDoorData.sampleNum));
        //   CHK_CU(cudaMemset(dSumNeighborhoodSizes[deviceIdx], 0, sizeof(EdgePos_t)));
        //   CHK_CU(cudaGetLastError());
        //   CHK_CU(cudaDeviceSynchronize());
        //   // TODO: Neighborhood is edges of all transit vertices. Hence, neighborhood size is (# of Transit Vertices)/(|G.V|) * |G.E|
        //   CHK_CU(cudaMemcpy(hSumNeighborhoodSizes[deviceIdx], dSumNeighborhoodSizes[deviceIdx], sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
        //   // std::cout <<" hSumNeighborhoodSizes " << *hSumNeighborhoodSizes << std::endl;
        //   CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRCols[deviceIdx], sizeof(VertexID_t) * (*hSumNeighborhoodSizes[deviceIdx])));
        //   CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRRows[deviceIdx], sizeof(EdgePos_t) * CCGApp().initialSampleSize() * nextDoorData.sampleNum));
        // }
      }
      else
      {
        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          // auto device = nextDoorData.devices[deviceIdx];
          // CHK_CU(cudaSetDevice(device));
          CHK_CU(cudaMemset(dKernelTransitNums[deviceIdx], 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
          CHK_CU(cudaMemset(dInvalidVertexStartPosInMap[deviceIdx], 0xFF, sizeof(EdgePos_t)));
          // const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.sampleNum, deviceIdx, numDevices);
          totalTransits[deviceIdx] = numTransits;
          // totalTransits[deviceIdx] = perDeviceNumSamples * numTransits;

          // Find the index of first invalid transit vertex.
          // auto _st = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
          invalidVertexStartPos<<<DIVUP(totalTransits[deviceIdx], 256L), 256>>>(step, nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                                                                totalTransits[deviceIdx], nextDoorData.INVALID_VERTEX,
                                                                                dInvalidVertexStartPosInMap[deviceIdx]);
          device->StreamSync(ctx, stream);
          // auto _ed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
          // std::cout<<"finish invalidVertexStartPos "<< std::fixed << std::setprecision(7) << (double)((_ed.count() - _st.count()) * 0.000001) << std::endl;
          // CHK_CU(cudaGetLastError());
        }

        // CUDA_SYNC_DEVICE_ALL(nextDoorData);

        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          device->CopyDataFromTo(dInvalidVertexStartPosInMap[deviceIdx], 0, invalidVertexStartPosInMap[deviceIdx], 0, 1 * sizeof(EdgePos_t), ctx, DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
          // Now the number of threads launched are equal to number of valid transit vertices
          if (*invalidVertexStartPosInMap[deviceIdx] == -1)
          {
            *invalidVertexStartPosInMap[deviceIdx] = totalTransits[deviceIdx];
          }
          totalThreads[deviceIdx] = *invalidVertexStartPosInMap[deviceIdx];
        }
        // printf("invalidVertexStartPosInMap: %ld, totNeighborsToSampleAtStep: %ld, totEdgesToSampleAtStep: %ld, stepsamplesize: %ld\n", *invalidVertexStartPosInMap[0], totNeighborsToSampleAtStep, totEdgesToSampleAtStep, totalThreads[0] * subWarpSizeAtStep(fanouts[step]));
        
        // 压缩Tranits数组，可省去
        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          void *dRunLengthEncodeTmpStorage = nullptr;
          size_t dRunLengthEncodeTmpStorageSize = 0;
          // Find the number of transit vertices
          cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize,
                                             nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                             dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx],
                                             dUniqueTransitsNumRuns[deviceIdx], totalThreads[deviceIdx]);

          if (dRunLengthEncodeTmpStorageSize > temp_storage_bytes[deviceIdx])
          {
            temp_storage_bytes[deviceIdx] = dRunLengthEncodeTmpStorageSize;
            device->FreeDataSpace(ctx, d_temp_storage[deviceIdx]);
            d_temp_storage[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, temp_storage_bytes[deviceIdx]));
          }
          assert(dRunLengthEncodeTmpStorageSize <= temp_storage_bytes[deviceIdx]);
          dRunLengthEncodeTmpStorage = d_temp_storage[deviceIdx];
          cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize,
                                             nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                             dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx],
                                             dUniqueTransitsNumRuns[deviceIdx], totalThreads[deviceIdx]);
          device->StreamSync(ctx, stream);
          // CHK_CU(cudaGetLastError());
        }

        // CUDA_SYNC_DEVICE_ALL(nextDoorData);

        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          device->CopyDataFromTo(dUniqueTransitsNumRuns[deviceIdx], 0, uniqueTransitNumRuns[deviceIdx], 0, sizeof(*uniqueTransitNumRuns[deviceIdx]), ctx, DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
          void *dExclusiveSumTmpStorage = nullptr;
          size_t dExclusiveSumTmpStorageSize = 0;
          // Exclusive sum to obtain the start position of each transit (and its samples) in the map
          cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts[deviceIdx],
                                        dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);

          if (dExclusiveSumTmpStorageSize > temp_storage_bytes[deviceIdx])
          {
            temp_storage_bytes[deviceIdx] = dExclusiveSumTmpStorageSize;
            device->FreeDataSpace(ctx, d_temp_storage[deviceIdx]);
            d_temp_storage[deviceIdx] = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, temp_storage_bytes[deviceIdx]));
          }
          assert(dExclusiveSumTmpStorageSize <= temp_storage_bytes[deviceIdx]);
          dExclusiveSumTmpStorage = d_temp_storage[deviceIdx];
          cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts[deviceIdx],
                                        dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);

          // CHK_CU(cudaGetLastError());
          // std::cout<<"unique transits: "<<*uniqueTransitNumRuns[deviceIdx]<<std::endl;
          // std::cout<<"dUniqueTransits: "<<std::endl;
          // gpuprint<<<1,1>>>(dUniqueTransits[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);
          // device->StreamSync(ctx, stream);
          // std::cout<<"dUniqueTransitsCounts: "<<std::endl;
          // gpuprint<<<1,1>>>(dUniqueTransitsCounts[deviceIdx], *uniqueTransitNumRuns[deviceIdx], LoadBalancing::LoadBalancingThreshold::GridLevel);
          // device->StreamSync(ctx, stream);
          // std::cout<<"Transit position: "<<std::endl;
          // gpuprint<<<1,1>>>(dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);
          // device->StreamSync(ctx, stream);
        }

        // CUDA_SYNC_DEVICE_ALL(nextDoorData);

        int subWarpSize = subWarpSizeAtStep(fanouts[step]);

        // auto _st = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          // auto device = nextDoorData.devices[deviceIdx];
          // CHK_CU(cudaSetDevice(device));
          if (*uniqueTransitNumRuns[deviceIdx] == 0) continue;

          partitionTransitsInKernels<512, TransitKernelTypes::GridKernel, true><<<utils::thread_block_size((*uniqueTransitNumRuns[deviceIdx]), 512L), 512>>>(
            step, dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx],
            dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx], nextDoorData.INVALID_VERTEX, dGridKernelTransits[deviceIdx], dGridKernelTransitsNum[deviceIdx],
            dThreadBlockKernelTransits[deviceIdx], dThreadBlockKernelTransitsNum[deviceIdx], dSubWarpKernelTransits[deviceIdx], dSubWarpKernelTransitsNum[deviceIdx], nullptr,
            dIdentityKernelTransitsNum[deviceIdx], dKernelTypeForTransit[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], fanouts[step]);

          device->StreamSync(ctx, stream);
          // CHK_CU(cudaGetLastError());
        }
        // CUDA_SYNC_DEVICE_ALL(nextDoorData);

        if (useThreadBlockKernel and subWarpSize > 1)
        {
          for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
          {
            // auto device = nextDoorData.devices[deviceIdx];
            // CHK_CU(cudaSetDevice(device));
            if (*uniqueTransitNumRuns[deviceIdx] == 0) continue;
            partitionTransitsInKernels<512, TransitKernelTypes::ThreadBlockKernel, false><<<utils::thread_block_size((*uniqueTransitNumRuns[deviceIdx]), 512L), 512>>>(
              step, dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx],
              dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx], nextDoorData.INVALID_VERTEX, dGridKernelTransits[deviceIdx], dGridKernelTransitsNum[deviceIdx],
              dThreadBlockKernelTransits[deviceIdx], dThreadBlockKernelTransitsNum[deviceIdx], dSubWarpKernelTransits[deviceIdx], dSubWarpKernelTransitsNum[deviceIdx], nullptr,
              dIdentityKernelTransitsNum[deviceIdx], dKernelTypeForTransit[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], fanouts[step]);

            device->StreamSync(ctx, stream);
            // CHK_CU(cudaGetLastError());
          }
          // CUDA_SYNC_DEVICE_ALL(nextDoorData);
        }
        // auto _ed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
        // std::cout<<"finish identityKernel "<< std::fixed << std::setprecision(7) << (double)((_ed.count() - _st.count()) * 0.000001) << std::endl;

        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          // auto device = nextDoorData.devices[deviceIdx];
          // CHK_CU(cudaSetDevice(device));
          if (*uniqueTransitNumRuns[deviceIdx] == 0) continue;
          device->CopyDataFromTo(dKernelTransitNums[deviceIdx], 0, hKernelTransitNums[deviceIdx], 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t), ctx, DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
          // CHK_CU(cudaMemcpy(hKernelTransitNums[deviceIdx], dKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
          // From each Transit we sample stepSize(step) vertices
          totalThreads[deviceIdx] = totalThreads[deviceIdx] * subWarpSize;
        }

        bool noTransitsForAllDevices = true;
        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          // auto device = nextDoorData.devices[deviceIdx];
          if (*uniqueTransitNumRuns[deviceIdx] > 0)
          {
            noTransitsForAllDevices = false;
          }
        }

        if (noTransitsForAllDevices)
        {
          std::cout<<"No transits for all devices. Stop."<<std::endl;
          break; // End Sampling because no more transits exists
        }
        // std::cout << "gridKernelTransitsNum " << *(hKernelTransitNums[0]) << std::endl;
        // std::cout << "threadBlockKernelTransitsNum " << *(hKernelTransitNums[0] + 1) << std::endl;
        // std::cout << "subWarpKernelTransitsNum " << *(hKernelTransitNums[0] + 2) << std::endl;
        // std::cout << "identityKernelTransitsNum " << *(hKernelTransitNums[0] + 3) << std::endl;
        // std::cout << "invalidVertexStartPosInMap " << *(hKernelTransitNums[0] + 4) << std::endl;

        // std::cout<<" TransitToSample: "<<std::endl;
        // gpuprint<<<1,1>>>(nextDoorData.dTransitToSampleMapKeys[0], totEdgesToSampleAtStep);
        // device->StreamSync(ctx, stream);
        // gpuprint<<<1,1>>>(nextDoorData.dTransitToSampleMapValues[0], totEdgesToSampleAtStep);
        // device->StreamSync(ctx, stream);
        // std::cout<<" SamplesToTransitMap: "<<std::endl;
        // gpuprint<<<1,1>>>(nextDoorData.dSamplesToTransitMapKeys[0], totEdgesToSampleAtStep);
        // device->StreamSync(ctx, stream);
        // gpuprint<<<1,1>>>(nextDoorData.dSamplesToTransitMapValues[0], totEdgesToSampleAtStep);
        // device->StreamSync(ctx, stream);
        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          // auto device = nextDoorData.devices[deviceIdx];
          // CHK_CU(cudaSetDevice(device));
          if (*uniqueTransitNumRuns[deviceIdx] == 0) continue;
          const size_t maxThreadBlocksPerKernel = min(8192L, nextDoorData.maxThreadsPerKernel[deviceIdx] / 256L);
          const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.sampleNum, deviceIdx, numDevices);
          if (*identityKernelTransitsNum[deviceIdx] > 0)
          {
            // CHK_CU(cudaGetLastError());
            // _st = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
            // std::cout<<"\n**** before identityKernel: totEdgesToSampleAtStep: "<<totEdgesToSampleAtStep<<std::endl;
            if (CCGApp().hasExplicitTransits())
            {
              identityKernel<CCGApp, 256, true, true><<<maxThreadBlocksPerKernel, 256>>>(
                step,deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], fanouts[step],
                d_fanouts, tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
            }
            else
            {
              identityKernel<CCGApp, 256, true, false><<<maxThreadBlocksPerKernel, 256>>>(
                step,deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], fanouts[step],
                d_fanouts, tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
            }
            device->StreamSync(ctx, stream);
            // _ed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
            // std::cout<<"finish identityKernel "<< std::fixed << std::setprecision(7) << (double)((_ed.count() - _st.count()) * 0.000001) << std::endl;
            // std::cout<<"\n**** after identityKernel: totEdgesToSampleAtStep: "<<totEdgesToSampleAtStep<<std::endl;
          }
        }
        // CUDA_SYNC_DEVICE_ALL(nextDoorData);

        if (subWarpSize > 1)
        {
          // throw "subWarpSize > 1";
          EdgePos_t finalSampleSizeTillPreviousStep = 0;
          EdgePos_t neighborsToSampleAtStep = 1;
          for (int _s = 0; _s < step; _s++)
          {
            neighborsToSampleAtStep *= fanouts[_s];
            finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
          }

          for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
          {
            // auto device = nextDoorData.devices[deviceIdx];
            // CHK_CU(cudaSetDevice(device));

            // Process more than one thread blocks positions written in dGridKernelTransits per thread block.
            // Processing more can improve the locality if thread blocks have common transits.
            const int perThreadSamplesForThreadBlockKernel = 4; // Works best for KHop 8 ?
            const int tbSize = 256L;
            // 4096
            const size_t maxThreadBlocksPerKernel = min(8192L, nextDoorData.maxThreadsPerKernel[deviceIdx] / tbSize);
            const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.sampleNum, deviceIdx, numDevices);
            const size_t threadBlocks = DIVUP((DIVUP((*threadBlockKernelTransitsNum[deviceIdx] * LoadBalancing::LoadBalancingThreshold::BlockLevel), tbSize)), perThreadSamplesForThreadBlockKernel);

            if (useThreadBlockKernel && *threadBlockKernelTransitsNum[deviceIdx] > 0)
            { // && numberOfTransits(step) > 1) {
              // FIXME: A Bug in Grid Kernel prevents it from being used when numberOfTransits for a sample at step are 1.
              //  for (int threadBlocksExecuted = 0; threadBlocksExecuted < threadBlocks; threadBlocksExecuted += nextDoorData.maxThreadsPerKernel/256) {
              const bool CACHE_EDGES = true;
              const bool CACHE_WEIGHTS = false;
              const int CACHE_SIZE = (CACHE_EDGES || CACHE_WEIGHTS) ? 384 : 0;
              // printf("device %d threadBlockKernelTransitsNum %d threadBlocks %d\n", device, *threadBlockKernelTransitsNum[deviceIdx], threadBlocks);
              // std::cout<<"\n**** before threadBlockKernel: totEdgesToSampleAtStep: "<<totEdgesToSampleAtStep<<" totNeighborsToSampleAtStep: "<<totNeighborsToSampleAtStep<<std::endl;
              // _st = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
              // VertexID_t* vis = static_cast<VertexID_t *>(device->AllocWorkspace(ctx, sizeof(VertexID_t) * totNeighborsToSampleAtStep));
              // CUDA_CALL(cudaMemset(vis, 0, sizeof(VertexID_t) * totNeighborsToSampleAtStep));
              switch (subWarpSizeAtStep(fanouts[step]))
              {
              case 32:
                threadBlockKernel<CCGApp, tbSize, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, perThreadSamplesForThreadBlockKernel, false, 0, 32><<<maxThreadBlocksPerKernel, tbSize>>>(
                  step, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                  (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                  spl_len, nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                  nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                  nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                  nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx],
                  *threadBlockKernelTransitsNum[deviceIdx], (int)threadBlocks, (int)fanouts[step], finalSampleSizeTillPreviousStep,
                  d_fanouts, tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
                break;
                // case 16:
                //   threadBlockKernel<tbSize,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,perThreadSamplesForThreadBlockKernel,false,0,16><<<maxThreadBlocksPerKernel, tbSize>>>(step,
                //     deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                //     (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                //     totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                //     nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                //     nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                //     nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx], *threadBlockKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep, d_fanouts, tot_step, out_rows, out_cols, out_idxs, sampledSize);
                //     break;
                // case 8:
                //   threadBlockKernel<tbSize,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,perThreadSamplesForThreadBlockKernel,false,0,8><<<maxThreadBlocksPerKernel, tbSize>>>(step,
                //     deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                //     (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                //     totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                //     nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                //     nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                //     nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx], *threadBlockKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep, d_fanouts, tot_step, out_rows, out_cols, out_idxs, sampledSize);
                //     break;
                // case 4:
                //   threadBlockKernel<tbSize,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,perThreadSamplesForThreadBlockKernel,false,0,4><<<maxThreadBlocksPerKernel, tbSize>>>(step,
                //     deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                //     (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                //     totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                //     nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                //     nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                //     nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx], *threadBlockKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep, d_fanouts, tot_step, out_rows, out_cols, out_idxs, sampledSize);
                //     break;
                // case 2:
                //   threadBlockKernel<tbSize,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,perThreadSamplesForThreadBlockKernel,false,0,2><<<maxThreadBlocksPerKernel, tbSize>>>(step,
                //     deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                //     (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                //     totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                //     nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                //     nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                //     nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx], *threadBlockKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep, d_fanouts, tot_step, out_rows, out_cols, out_idxs, sampledSize);
                //     break;
                // case 1:
                //   threadBlockKernel<tbSize,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,perThreadSamplesForThreadBlockKernel,false,0,1><<<maxThreadBlocksPerKernel, tbSize>>>(step,
                //     deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                //     (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                //     totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                //     nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                //     nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                //     nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx], *threadBlockKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep, d_fanouts, tot_step, out_rows, out_cols, out_idxs, sampledSize);
                //     break;
              }
              // CHK_CU(cudaGetLastError());
              device->StreamSync(ctx, stream);
              // _ed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
              // std::cout<<"finish threadblockKernel "<< std::fixed << std::setprecision(7) << (double)((_ed.count() - _st.count()) * 0.000001) << std::endl;

              // gpucknull<<<1,1>>>(vis, totNeighborsToSampleAtStep);
              // device->FreeWorkspace(ctx, vis);
              // std::cout<<"\n**** after threadBlockKernel: totEdgesToSampleAtStep: "<<totEdgesToSampleAtStep<<std::endl;
              // CHK_CU(cudaDeviceSynchronize());
              // }
            }
          }

          // CUDA_SYNC_DEVICE_ALL(nextDoorData);

          for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
          {
            // auto device = nextDoorData.devices[deviceIdx];
            // CHK_CU(cudaSetDevice(device));

            // Process more than one thread blocks positions written in dGridKernelTransits per thread block.
            // Processing more can improve the locality if thread blocks have common transits.
            const int perThreadSamplesForGridKernel = 8;
            // const int perThreadSamplesForGridKernel = 16; // Works best for KHop

            const size_t maxThreadBlocksPerKernel = min(4096L, nextDoorData.maxThreadsPerKernel[deviceIdx] / 256L);
            const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.sampleNum, deviceIdx, numDevices);
            const size_t threadBlocks = DIVUP(*gridKernelTransitsNum[deviceIdx], perThreadSamplesForGridKernel);
            // printf("device %d gridTransitsNum %d threadBlocks %d\n", device, *gridKernelTransitsNum[deviceIdx], threadBlocks);

            if (useGridKernel && *gridKernelTransitsNum[deviceIdx] > 0)
            { // && numberOfTransits(step) > 1) {
              // FIXME: A Bug in Grid Kernel prevents it from being used when numberOfTransits for a sample at step are 1.
              const bool CACHE_EDGES = true;
              const bool CACHE_WEIGHTS = false;
              const int CACHE_SIZE = (CACHE_EDGES || CACHE_WEIGHTS) ? 3 * 1024 - 10 : 0;
              // std::cout<<"\n**** before gridKernel: totEdgesToSampleAtStep: "<<totEdgesToSampleAtStep<<std::endl;
              // _st = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
              switch (subWarpSizeAtStep(fanouts[step]))
              {
              case 32:
                gridKernel<CCGApp, 256, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, false, perThreadSamplesForGridKernel, true, false, 256, 32><<<maxThreadBlocksPerKernel, 256>>>(
                  step,deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                  (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                  totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                  nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                  nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                  nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx],
                  *gridKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep,
                  d_fanouts, (int)tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
                break;
              case 16:
                gridKernel<CCGApp, 256, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, false, perThreadSamplesForGridKernel, true, true, 256, 16><<<maxThreadBlocksPerKernel, 256>>>(
                  step, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                  (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                  totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                  nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                  nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                  nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx],
                  *gridKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep,
                  d_fanouts, (int)tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
                break;
              case 8:
                gridKernel<CCGApp, 256, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, false, perThreadSamplesForGridKernel, true, true, 256, 8><<<maxThreadBlocksPerKernel, 256>>>(
                  step, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                  (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                  totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                  nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                  nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                  nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx],
                  *gridKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep,
                  d_fanouts, (int)tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
                break;
              case 4:
                gridKernel<CCGApp, 256, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, false, perThreadSamplesForGridKernel, true, true, 256, 4><<<maxThreadBlocksPerKernel, 256>>>(
                  step, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                  (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                  totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                  nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                  nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                  nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx],
                  *gridKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep,
                  d_fanouts, (int)tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
                break;
              case 2:
                gridKernel<CCGApp, 256, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, false, perThreadSamplesForGridKernel, true, true, 256, 2><<<maxThreadBlocksPerKernel, 256>>>(
                  step, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                  (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                  totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                  nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                  nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                  nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx],
                  *gridKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep,
                  d_fanouts, (int)tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
                break;
              case 1:
                gridKernel<CCGApp, 256, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, false, perThreadSamplesForGridKernel, true, true, 256, 1><<<maxThreadBlocksPerKernel, 256>>>(
                  step, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                  (const VertexID_t *)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t *)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                  totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.sampleNum,
                  nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                  nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                  nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx],
                  *gridKernelTransitsNum[deviceIdx], threadBlocks, fanouts[step], finalSampleSizeTillPreviousStep,
                  d_fanouts, (int)tot_step, out_rows, out_cols, out_idxs, out_trace, trace_length);
                break;
              default:
                // TODO: Add others
                break;
              }
              device->StreamSync(ctx, stream);
              // _ed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
              // std::cout<<"finish gridKernel "<< std::fixed << std::setprecision(7) << (double)((_ed.count() - _st.count()) * 0.000001) << std::endl;
              // std::cout<<"\n**** after gridKernel: totEdgesToSampleAtStep: "<<totEdgesToSampleAtStep<<std::endl;
              // CHK_CU(cudaGetLastError());
              // }
            }
          }
          // CUDA_SYNC_DEVICE_ALL(nextDoorData);
        }
      }
    }

    if (step != tot_step - 1)
    {
      // add src nodes into dst
      if (CCGApp().samplingType() == SamplingType::NeighborSampling && addsrc)
      {
        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          // auto device=nextDoorData.devices[deviceIdx];
          // CHK_CU(cudaSetDevice(device));
          device->CopyDataFromTo(nextDoorData.dTransitToSampleMapValues[deviceIdx], 0, nextDoorData.dSamplesToTransitMapKeys[deviceIdx], totEdgesToSampleAtStep * sizeof(VertexID_t), sizeof(VertexID_t) * totNeighborsToSampleAtStep, ctx, ctx, DGLDataType{kDGLInt, 64, 1});
          device->CopyDataFromTo(nextDoorData.dTransitToSampleMapKeys[deviceIdx], 0, nextDoorData.dSamplesToTransitMapValues[deviceIdx], totEdgesToSampleAtStep * sizeof(VertexID_t), sizeof(VertexID_t) * totNeighborsToSampleAtStep, ctx, ctx, DGLDataType{kDGLInt, 64, 1});
          // CHK_CU(cudaMemcpy(nextDoorData.dSamplesToTransitMapKeys[deviceIdx]+totEdgesToSampleAtStep, nextDoorData.dTransitToSampleMapValues[deviceIdx], sizeof(VertexID_t) * totNeighborsToSampleAtStep, cudaMemcpyDeviceToDevice));
          // CHK_CU(cudaMemcpy(nextDoorData.dSamplesToTransitMapValues[deviceIdx]+totEdgesToSampleAtStep, nextDoorData.dTransitToSampleMapKeys[deviceIdx], sizeof(VertexID_t) * totNeighborsToSampleAtStep, cudaMemcpyDeviceToDevice));
          totEdgesToSampleAtStep = totNeighborsToSampleAtStep + totEdgesToSampleAtStep;
        }
      }

      for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
      {
        // auto device = nextDoorData.devices[deviceIdx];
        // CHK_CU(cudaSetDevice(device));
        // Invert sample->transit map by sorting samples based on the transit vertices
        // std::cout<<"**** totalThreads 1: "<<totalThreads[deviceIdx]<<std::endl;
        // gpuprint<<<1,1>>>(nextDoorData.dSamplesToTransitMapValues[deviceIdx], totalThreads[deviceIdx]);
        // gpuprint<<<1,1>>>(nextDoorData.dSamplesToTransitMapKeys[deviceIdx], totalThreads[deviceIdx]);
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx],
                                        nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                        nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx],
                                        totEdgesToSampleAtStep, 0, nextDoorData.maxBits));
        // std::cout<<"\n**** totalThreads 2: "<<totalThreads[deviceIdx]<<std::endl;
        // gpuprint<<<1,1>>>(nextDoorData.dTransitToSampleMapKeys[deviceIdx], totalThreads[deviceIdx]);
        // gpuprint<<<1,1>>>(nextDoorData.dTransitToSampleMapValues[deviceIdx], totalThreads[deviceIdx]);
        // cudaDeviceSynchronize();
        // ********* 修改 neighborsToSampleAtStep ***********
      }
      // CUDA_SYNC_DEVICE_ALL(nextDoorData);
      
      // get unique tranist for next step
      if (CCGApp().samplingType() == SamplingType::NeighborSampling && uniqueseeds)
      {
        for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
        {
          CUDA_CALL(cub::DeviceSelect::UniqueByKey(d_unique_temp_storage[deviceIdx], unique_temp_storage_bytes[deviceIdx],
                                      nextDoorData.dTransitToSampleMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx],
                                      nextDoorData.dTransitToSampleMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx],
                                      d_num_selected_out[deviceIdx], totEdgesToSampleAtStep));
          device->CopyDataFromTo(d_num_selected_out[deviceIdx], 0, &totEdgesToSampleAtStep, 0, sizeof(VertexID_t), ctx, DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
          // std::cout<<"\n**** num_selected_out: "<<totEdgesToSampleAtStep<<std::endl<<"TransitToSampleMap:\n";
        }
        // CUDA_SYNC_DEVICE_ALL(nextDoorData);
      }
      totNeighborsToSampleAtStep = totEdgesToSampleAtStep;

      // std::cout<<"\n**** end step " << step << ", totEdgesToSampleAtStep: "<<totEdgesToSampleAtStep<<std::endl;
      // std::cout<<" TransitToSample: "<<std::endl;
      // gpuprint<<<1,1>>>(nextDoorData.dTransitToSampleMapKeys[0], totEdgesToSampleAtStep);
      // device->StreamSync(ctx, stream);
      // gpuprint<<<1,1>>>(nextDoorData.dTransitToSampleMapValues[0], totEdgesToSampleAtStep);
      // device->StreamSync(ctx, stream);
      // std::cout<<" SamplesToTransitMap: "<<std::endl;
      // gpuprint<<<1,1>>>(nextDoorData.dSamplesToTransitMapKeys[0], totEdgesToSampleAtStep);
      // device->StreamSync(ctx, stream);
      // gpuprint<<<1,1>>>(nextDoorData.dSamplesToTransitMapValues[0], totEdgesToSampleAtStep);
      // device->StreamSync(ctx, stream);
    }

    CCGApp().procStepSample();

    // if (CCGApp().samplingType() == SamplingType::NeighborSampling)
    // {
      // std::cout << "end step. CCGSample COO Matrix: length: " << spl_len << std::endl;
      // gpuck<<<1,1>>>((VertexID_t *)picked_row->data, (VertexID_t *)picked_col->data, (VertexID_t *)picked_idx->data, spl_len, 4846609, 85702474);
      // device->StreamSync(ctx, stream);
      // gpuprint<<<1,1>>>((VertexID_t *)picked_row->data, spl_len);
      // device->StreamSync(ctx, stream);
      // gpuprint<<<1,1>>>((VertexID_t *)picked_col->data, spl_len);
      // device->StreamSync(ctx, stream);
      // gpuprint<<<1,1>>>((VertexID_t *)picked_idx->data, spl_len);
      // device->StreamSync(ctx, stream);
    // }
  }
  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] ed step_sample " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << "\n";

  device->FreeWorkspace(ctx, d_fanouts);
  for (size_t deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++)
  {    
    device->FreeWorkspace(ctx, d_temp_storage[deviceIdx]);
    device->FreeWorkspace(ctx, d_unique_temp_storage[deviceIdx]);
    device->FreeWorkspace(ctx, d_num_selected_out[deviceIdx]);
    device->FreeWorkspace(ctx, dUniqueTransits[deviceIdx]);
    device->FreeWorkspace(ctx, dUniqueTransitsCounts[deviceIdx]);
    device->FreeWorkspace(ctx, dUniqueTransitsNumRuns[deviceIdx]);
    device->FreeWorkspace(ctx, dKernelTypeForTransit[deviceIdx]);
    device->FreeWorkspace(ctx, dTransitPositions[deviceIdx]);
    device->FreeWorkspace(ctx, dGridKernelTransits[deviceIdx]);
    if (useThreadBlockKernel) device->FreeWorkspace(ctx, dThreadBlockKernelTransits[deviceIdx]);
    if (useSubWarpKernel) device->FreeWorkspace(ctx, dSubWarpKernelTransits[deviceIdx]);
    device->FreeWorkspace(ctx, dKernelTransitNums[deviceIdx]);
  }
  return;
}

__global__ void fullsamplingKernel(VertexID_t *d_seed_nodes, VertexID_t seed_node_size, EdgePos_t* degoffset, VertexID_t* out_rows, VertexID_t* out_cols, EdgePos_t* out_idxs)
{
  EdgePos_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  if (threadId >= seed_node_size)
    return;

  VertexID_t transit = d_seed_nodes[threadId];
  EdgePos_t beginPos = degoffset[threadId];
  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
  BCGVertex bcgv(transit, bcg->graph, bcg->offset[transit]);
  EdgePos_t numTransitEdges = bcgv.outd;
  VertexID_t neighbor;
  EdgePos_t neighbor_pos = bcg->degoffset[transit];

  for (EdgePos_t neighborIdx = 0; neighborIdx < numTransitEdges; neighborIdx++)
  {
    neighbor = bcgv.get_vertex(neighborIdx);
    out_rows[beginPos + neighborIdx] = transit;
    out_cols[beginPos + neighborIdx] = neighbor;
    out_idxs[beginPos + neighborIdx] = neighbor_pos + neighborIdx;
  }
  // if(threadId == seed_node_size - 1) {
  //   printf("Last position: %d\n", )
  // }
}

// std::vector<GPUBCGPartition> setCCGStorage(const NextDoorData& data, void* gpu_ccg) {
//   //Assume that whole graph can be stored in GPU Memory.
//   //Hence, only one Graph Partition is created.
//   std::vector<GPUBCGPartition> gpuBCGPartitions;
//   //Copy full graph to GPU
//   for (size_t device = 0; device < data.devices.size(); device++) {
//     GPUBCGPartition gb = *(GPUBCGPartition*)gpu_ccg;
//     gpuBCGPartitions.push_back(gb);
//   }
//   return gpuBCGPartitions;
// }

void setNextDoorData(NextDoorData *data, void *gpu_ccg, void *crs)
{
  for (size_t device = 0; device < data->devices.size(); device++)
  {
    // GPUBCGPartition gb = *(GPUBCGPartition *)gpu_ccg;
    // data->gpuBCGPartitions.push_back(gb);

    data->maxThreadsPerKernel[device] = CCGCurandNum;
    data->dCurandStates[device] = (curandState *)crs;
  }
  return;
}

std::vector<dgl::aten::COOMatrix> CCGSampleNeighbors(uint64_t n_nodes, void *gpu_ccg, void *crs, NextDoorData *nextDoorData, IdArray &seed_nodes_arr, const std::vector<int64_t> &fanouts, bool loadBalacing = true)
{
  const auto& ctx = seed_nodes_arr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  CCGNeighborApp().init(n_nodes, fanouts, ctx);

  // size_t free, free1, tot;
  // CUDA_CALL(cudaMemGetInfo(&free, &tot));
  
  // std::cout<<"CCGSampleNeighbors: "<<"seed_node length "<<seed_nodes_arr.NumElements()<<" device: "seed_nodes_arr->ctx.device_type<<" "<<kDGLCUDA<<std::endl;
  // gpuprint<<<1,1>>>(static_cast<VertexID_t*> (seed_nodes_arr->data), seed_nodes_arr.NumElements());
  // cudaDeviceSynchronize();

  // auto _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] bg cu_initspl " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << "\n";
  initializeNextDoorSample<CCGNeighborApp>(*nextDoorData, seed_nodes_arr, fanouts.size());
  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] ed cu_initspl " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << "\n";

  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] bg cu_dospl " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << "\n";
  // printf("seeds number: %ld\n", seed_nodes_arr.NumElements());
  doTransitParallelSampling<CCGNeighborApp>(*nextDoorData, ctx, loadBalacing);
  // CUDA_CALL(cudaMemGetInfo(&free1, &tot));
  // std::cout << "[PF] stat CCGSampleNeighbors "<< (free-free1)/1024/1024 << std::endl;
  // _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  // std::cout << "[PF] ed cu_dospl " << std::fixed << std::setprecision(7) << (double)(_outt.count() * 0.000001) << "\n";
  // for(int i=0;i<vecCOO.size();i++) {
  //   std::cout<<i<<" "<<vecCOO[i].row.use_count()<<std::endl;
  //   ret[i] = std::move(vecCOO[i]);
  // }
  // freeDeviceData(nextDoorData);
  // const DGLDataType& dtype = DGLDataType{kDGLInt, 64, 1};
  // return std::vector<dgl::aten::COOMatrix>(fanouts.size(), dgl::aten::COOMatrix(0, 0, aten::NullArray(dtype, ctx), aten::NullArray(dtype, ctx)));
  return std::move(vecCOO);
}


////////////////////////////////
// Random walk sampling
////////////////////////////////

IdArray CCGRandomWalk(uint64_t n_nodes, void *gpu_ccg, NextDoorData *nextDoorData, IdArray seeds, int64_t length, bool loadBalacing = true)
{
  const auto& ctx = seeds->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  // std::cout<<ctx.device_type<<" "<<kDGLCUDA<<std::endl;
  
  CCGRandomWalkApp().init(n_nodes, seeds, length, ctx);
  setFristNode<<<utils::thread_block_size((unsigned long)seeds.NumElements(), 256UL), 256UL>>>(out_trace, seeds.Ptr<VertexID_t>(), trace_length);
  initializeNextDoorSample<CCGRandomWalkApp>(*nextDoorData, seeds, (int)trace_length - 1);
  // std::cout<< "seeds size: " << seeds.NumElements() << " trace length: " << trace_length << std::endl;
  doTransitParallelSampling<CCGRandomWalkApp>(*nextDoorData, ctx, loadBalacing);
  // std::cout<<"use count: " << traces.use_count() << std::endl;
  return std::move(traces);
}


////////////////////////////////
// Full Layer sampling
////////////////////////////////

__global__ void initializeFullLayersSample(VertexID_t *seed_nodes, VertexID_t seed_num, VertexID_t n_nodes, EdgePos_t* ccg_deg, EdgePos_t* deg)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= seed_num)
    return;
  deg[thread_id] = ccg_deg[seed_nodes[thread_id]];
}

std::vector<dgl::aten::COOMatrix> CCGSampleFullLayers(uint64_t n_nodes, void *gpu_ccg, IdArray &seed_nodes_arr, int64_t num_layers)
{
  GPUBCGPartition *d_ccg = (GPUBCGPartition *)gpu_ccg;
  std::vector<dgl::aten::COOMatrix> ret;
  VertexID_t seed_node_size = seed_nodes_arr.NumElements();
  VertexID_t *d_seed_nodes;
  std::vector<VertexID_t> h_new_seed_nodes;

  CHK_CU(cudaMalloc(&d_seed_nodes, sizeof(VertexID_t) * n_nodes));
  CHK_CU(cudaMemcpy(d_seed_nodes, seed_nodes_arr->data, sizeof(VertexID_t) * seed_node_size, cudaMemcpyDeviceToDevice));
  for (int64_t step = 0; step < num_layers; step++)
  {
    // TO-DO: malloc first
    FullLayersData *fullLayerData = new FullLayersData();
    CHK_CU(cudaMalloc(&(fullLayerData->d_deg), sizeof(EdgePos_t) * (seed_node_size + 1)));
    CHK_CU(cudaMalloc(&(fullLayerData->d_degoffset), sizeof(EdgePos_t) * (seed_node_size + 1)));

    // std::cout << "seed_node_size: " << seed_node_size << " n_nodes: " << n_nodes << std::endl;
    // gpuprint<<<1,1>>>(d_seed_nodes, seed_node_size);
    initializeFullLayersSample<<<utils::thread_block_size((size_t)seed_node_size, N_THREADS), N_THREADS>>>(d_seed_nodes, seed_node_size, n_nodes, d_ccg->d_deg, fullLayerData->d_deg);
    
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, fullLayerData->d_deg, fullLayerData->d_degoffset, seed_node_size + 1);
    CHK_CU(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, fullLayerData->d_deg, fullLayerData->d_degoffset, seed_node_size + 1);

    EdgePos_t spl_len;
    CHK_CU(cudaMemcpy(&spl_len, fullLayerData->d_degoffset + seed_node_size, sizeof(EdgePos_t), cudaMemcpyDeviceToHost));

    // std::cout << "spl_len: " << spl_len << std::endl;
    fullLayerData->setSampleNum(spl_len);

    fullsamplingKernel<<<utils::thread_block_size((size_t)seed_node_size, N_THREADS), N_THREADS>>>(d_seed_nodes, seed_node_size, fullLayerData->d_degoffset, fullLayerData->d_out_rows, fullLayerData->d_out_cols, fullLayerData->d_out_idxs);
    if(step != num_layers - 1) {
      // prepare new seed_node_size & d_seed_nodes
      VertexID_t* h_seed_nodes = static_cast<VertexID_t *>(malloc(sizeof(VertexID_t) * spl_len));
      CHK_CU(cudaMemcpy(h_seed_nodes, fullLayerData->d_out_cols, sizeof(VertexID_t) * spl_len, cudaMemcpyDeviceToHost));

      std::unordered_set<VertexID_t> new_seed_set;
      for(VertexID_t idx = 0; idx < spl_len; idx++) {
        new_seed_set.insert(h_seed_nodes[idx]);
      }
      seed_node_size = new_seed_set.size();
      h_new_seed_nodes.clear();
      h_new_seed_nodes.reserve(seed_node_size);
      h_new_seed_nodes.insert(h_new_seed_nodes.end(), new_seed_set.begin(), new_seed_set.end());
      CHK_CU(cudaMemcpy(d_seed_nodes, h_new_seed_nodes.data(), sizeof(VertexID_t) * seed_node_size, cudaMemcpyHostToDevice));
      free(h_seed_nodes);
    }

    auto _picked_row = fullLayerData->d_picked_row.CreateView({spl_len}, fullLayerData->d_picked_row->dtype);
    auto _picked_col = fullLayerData->d_picked_col.CreateView({spl_len}, fullLayerData->d_picked_col->dtype);
    auto _picked_idx = fullLayerData->d_picked_idx.CreateView({spl_len}, fullLayerData->d_picked_idx->dtype);
    // std::cout << "CCGSampleFullLayers COO Matrix:" << std::endl;
    // std::cout << spl_len << " " << _picked_row << std::endl;
    // std::cout << spl_len << " " << _picked_col << std::endl;
    // std::cout << spl_len << " " << _picked_idx << std::endl;
    ret.push_back(dgl::aten::COOMatrix(n_nodes, n_nodes, _picked_col, _picked_row, _picked_idx));

    CHK_CU(cudaFree(fullLayerData->d_deg));
    CHK_CU(cudaFree(fullLayerData->d_degoffset));
    CHK_CU(cudaFree(d_temp_storage));
  }
  // CHK_CU(cudaDeviceSynchronize());
  return ret;
}

// } // namespace sampling

////////////////////////////////
// Labor sampling
////////////////////////////////

using namespace dgl::aten::cuda;

constexpr int BLOCK_SIZE = 128;
constexpr int CTA_SIZE = 128;
constexpr double eps = 0.0001;

namespace {

template <typename IdType>
struct TransformOp {
  const IdType* idx_coo;
  const IdType* rows;
  const IdType* subindptr;
  const IdType* data_arr;
  __host__ __device__ auto operator()(IdType idx) {
    const auto in_row = idx_coo[idx];
    auto row = rows[in_row];
    IdType rofs = idx - subindptr[in_row];
    BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
    const auto in_idx = bcg->degoffset[row] + rofs;
    BCGVertex bcgv(row, bcg->graph, bcg->offset[row]);
    const auto u = bcgv.get_vertex(rofs);
    const auto data = data_arr ? data_arr[in_idx] : in_idx;
    return thrust::make_tuple(row, u, data);
  }
};

template <
    typename IdType, typename FloatType, typename probs_t, typename A_t,
    typename B_t>
struct TransformOpImp {
  probs_t probs;
  A_t A;
  B_t B;
  const IdType* idx_coo;
  const IdType* rows;
  const FloatType* cs;
  const IdType* subindptr;
  const IdType* data_arr;
  __host__ __device__ auto operator()(IdType idx) {
    const auto ps = probs[idx];
    const auto in_row = idx_coo[idx];
    const auto c = cs[in_row];
    auto row = rows[in_row];
    IdType rofs = idx - subindptr[in_row];
    BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
    const auto in_idx = bcg->degoffset[row] + rofs;
    BCGVertex bcgv(row, bcg->graph, bcg->offset[row]);
    const auto u = bcgv.get_vertex(rofs);
    const auto w = A[in_idx];
    const auto w2 = B[in_idx];
    const auto data = data_arr ? data_arr[in_idx] : in_idx;
    return thrust::make_tuple(
        in_row, row, u, data, w / min((FloatType)1, c * w2 * ps));
  }
};

template <typename FloatType>
struct StencilOp {
  const FloatType* cs;
  template <typename IdType>
  __host__ __device__ auto operator()(
      IdType in_row, FloatType ps, FloatType rnd) {
    return rnd <= cs[in_row] * ps;
  }
};

template <typename IdType, typename FloatType, typename ps_t, typename A_t>
struct StencilOpFused {
  const uint64_t rand_seed;
  const IdType* idx_coo;
  const FloatType* cs;
  const ps_t probs;
  const A_t A;
  const IdType* subindptr;
  const IdType* rows;
  const IdType* nids;
  __device__ auto operator()(IdType idx) {
    const auto in_row = idx_coo[idx];
    const auto ps = probs[idx];
    IdType rofs = idx - subindptr[in_row];
    IdType row = rows[in_row];
    BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
    const auto in_idx = bcg->degoffset[row] + rofs;
    BCGVertex bcgv(row, bcg->graph, bcg->offset[row]);
    const auto u = bcgv.get_vertex(rofs);
    const auto t = nids ? nids[u] : u;  // t in the paper
    curandStatePhilox4_32_10_t rng;
    // rolled random number r_t is a function of the random_seed and t
    curand_init(123123, rand_seed, t, &rng);
    const float rnd = curand_uniform(&rng);
    return rnd <= cs[in_row] * A[in_idx] * ps;
  }
};

template <typename IdType, typename FloatType>
struct TransformOpMean {
  const IdType* ds;
  const FloatType* ws;
  __host__ __device__ auto operator()(IdType idx, FloatType ps) {
    return ps * ds[idx] / ws[idx];
  }
};

struct TransformOpMinWith1 {
  template <typename FloatType>
  __host__ __device__ auto operator()(FloatType x) {
    return min((FloatType)1, x);
  }
};

template <typename IdType>
struct IndptrFunc {
  __host__ __device__ auto operator()(IdType row) { 
    BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
    return bcg->degoffset[row];
  }
};

template <typename FloatType>
struct SquareFunc {
  __host__ __device__ auto operator()(FloatType x) {
    return thrust::make_tuple(x, x * x);
  }
};

struct TupleSum {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return thrust::make_tuple(
        thrust::get<0>(a) + thrust::get<0>(b),
        thrust::get<1>(a) + thrust::get<1>(b));
  }
};

template <typename IdType, typename FloatType>
struct DegreeFunc {
  const IdType num_picks;
  const IdType* rows;
  const FloatType* ds;
  IdType* in_deg;
  FloatType* cs;
  __host__ __device__ auto operator()(IdType tIdx) {
    BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];
    const auto out_row = rows[tIdx];
    const auto d = bcg->degoffset[out_row + 1] - bcg->degoffset[out_row];
    in_deg[tIdx] = d;
    cs[tIdx] = num_picks / (ds ? ds[tIdx] : (FloatType)d);
  }
};

template <typename IdType, typename FloatType>
__global__ void _CCGRowWiseOneHopExtractorKernel(
    const uint64_t rand_seed, const IdType hop_size, const IdType* const rows,
    const IdType* const subindptr,
    const IdType* const idx_coo,
    const IdType* const nids, const FloatType* const A, FloatType* const rands,
    IdType* const hop, FloatType* const A_l) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  curandStatePhilox4_32_10_t rng;

  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];

  while (tx < hop_size) {
    IdType rpos = idx_coo[tx];
    IdType rofs = tx - subindptr[rpos];
    IdType row = rows[rpos];
    const auto in_idx = bcg->degoffset[row] + rofs;
    BCGVertex bcgv(row, bcg->graph, bcg->offset[row]);
    const auto u = bcgv.get_vertex(rofs);
    hop[tx] = u;
    const auto v = nids ? nids[u] : u;
    // 123123 is just a number with no significance.
    curand_init(123123, rand_seed, v, &rng);
    const float rnd = curand_uniform(&rng);
    if (A) A_l[tx] = A[in_idx];
    rands[tx] = (FloatType)rnd;
    tx += stride_x;
  }
}

template <typename IdType, typename FloatType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CCGRowWiseLayerSampleDegreeKernel(
    const IdType num_picks, const IdType num_rows, const IdType* const rows,
    FloatType* const cs, const FloatType* const ds, const FloatType* const d2s,
    const FloatType* const probs,
    const FloatType* const A, const IdType* const subindptr) {
  typedef cub::BlockReduce<FloatType, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ FloatType var_1_bcast[BLOCK_CTAS];

  // we assign one warp per row
  assert(blockDim.x == CTA_SIZE);
  assert(blockDim.y == BLOCK_CTAS);

  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const auto last_row =
      min(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  constexpr FloatType ONE = 1;

  BCGPartition *bcg = (BCGPartition *)&bcgPartitionBuff[0];

  while (out_row < last_row) {
    const auto row = rows[out_row];

    const auto in_row_start = bcg->degoffset[row];
    const auto out_row_start = subindptr[out_row];

    const IdType degree = bcg->degoffset[row + 1] - in_row_start;

    if (degree > 0) {
      // stands for k in in arXiv:2210.13339, i.e. fanout
      const auto k = min(num_picks, degree);
      // slightly better than NS
      const FloatType d_ = ds ? ds[row] : degree;
      // stands for right handside of Equation (22) in arXiv:2210.13339
      FloatType var_target =
          d_ * d_ / k + (ds ? d2s[row] - d_ * d_ / degree : 0);

      auto c = cs[out_row];
      const int num_valid = min(degree, (IdType)CTA_SIZE);
      // stands for left handside of Equation (22) in arXiv:2210.13339
      FloatType var_1;
      do {
        var_1 = 0;
        if (A) {
          for (int idx = threadIdx.x; idx < degree; idx += CTA_SIZE) {
            const auto w = A[in_row_start + idx];
            const auto ps = probs ? probs[out_row_start + idx] : w;
            var_1 += w * w / min(ONE, c * ps);
          }
        } else {
          for (int idx = threadIdx.x; idx < degree; idx += CTA_SIZE) {
            const auto ps = probs[out_row_start + idx];
            var_1 += 1 / min(ONE, c * ps);
          }
        }
        var_1 = BlockReduce(temp_storage).Sum(var_1, num_valid);
        if (threadIdx.x == 0) var_1_bcast[threadIdx.y] = var_1;
        __syncthreads();
        var_1 = var_1_bcast[threadIdx.y];

        c *= var_1 / var_target;
      } while (min(var_1, var_target) / max(var_1, var_target) < 1 - eps);

      if (threadIdx.x == 0) cs[out_row] = c;
    }

    out_row += BLOCK_CTAS;
  }
}


}  // namespace

template <typename IdType, typename FloatType, typename exec_policy_t>
void compute_importance_sampling_probabilities(
    VertexID_t n_nodes, const IdType hop_size, cudaStream_t stream,
    const uint64_t random_seed, const IdType num_rows, const IdType* rows,
    const IdType* subindptr,
    IdArray idx_coo_arr, const IdType* nids,
    FloatArray cs_arr,  // holds the computed cs values, has size num_rows
    const bool weighted, const FloatType* A, const FloatType* ds,
    const FloatType* d2s, const IdType num_picks, DGLContext ctx,
    const runtime::CUDAWorkspaceAllocator& allocator,
    const exec_policy_t& exec_policy, const int importance_sampling,
    IdType* hop_1,  // holds the contiguous one-hop neighborhood, has size |E|
    FloatType* rands,  // holds the rolled random numbers r_t for each edge, has
                       // size |E|
    FloatType* probs_found) {  // holds the computed pi_t values for each edge,
                               // has size |E|
  auto device = runtime::DeviceAPI::Get(ctx);
  auto idx_coo = idx_coo_arr.Ptr<IdType>();
  auto cs = cs_arr.Ptr<FloatType>();
  FloatArray A_l_arr = weighted
                           ? NewFloatArray(hop_size, ctx, sizeof(FloatType) * 8)
                           : NullArray();
  auto A_l = A_l_arr.Ptr<FloatType>();

  const uint64_t max_log_num_vertices = [&]() -> int {
    for (int i = 0; i < static_cast<int>(sizeof(IdType)) * 8; i++)
      if (n_nodes <= ((IdType)1) << i) return i;
    return sizeof(IdType) * 8;
  }();

  {  // extracts the onehop neighborhood cols to a contiguous range into hop_1
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((hop_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    CUDA_KERNEL_CALL(
        (_CCGRowWiseOneHopExtractorKernel<IdType, FloatType>), grid, block, 0,
        stream, random_seed, hop_size, rows, subindptr,
        idx_coo, nids, weighted ? A : nullptr, rands, hop_1, A_l);
  }
  int64_t hop_uniq_size = 0;
  IdArray hop_new_arr = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  auto hop_new = hop_new_arr.Ptr<IdType>();
  auto hop_unique = allocator.alloc_unique<IdType>(hop_size);
  // After this block, hop_unique holds the unique set of one-hop neighborhood
  // and hop_new holds the relabeled hop_1, idx_coo already holds relabeled
  // destination. hop_unique[hop_new] == hop_1 holds
  {
    auto hop_2 = allocator.alloc_unique<IdType>(hop_size);
    auto hop_3 = allocator.alloc_unique<IdType>(hop_size);

    device->CopyDataFromTo(
        hop_1, 0, hop_2.get(), 0, sizeof(IdType) * hop_size, ctx, ctx,
        DGLDataType{kDGLInt, 64, 1});

    cub::DoubleBuffer<IdType> hop_b(hop_2.get(), hop_3.get());

    {
      std::size_t temp_storage_bytes = 0;
      CUDA_CALL(cub::DeviceRadixSort::SortKeys(
          nullptr, temp_storage_bytes, hop_b, hop_size, 0, max_log_num_vertices,
          stream));

      auto temp = allocator.alloc_unique<char>(temp_storage_bytes);

      CUDA_CALL(cub::DeviceRadixSort::SortKeys(
          temp.get(), temp_storage_bytes, hop_b, hop_size, 0,
          max_log_num_vertices, stream));
    }

    auto hop_counts = allocator.alloc_unique<IdType>(hop_size + 1);
    auto hop_unique_size = allocator.alloc_unique<int64_t>(1);

    {
      std::size_t temp_storage_bytes = 0;
      CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
          nullptr, temp_storage_bytes, hop_b.Current(), hop_unique.get(),
          hop_counts.get(), hop_unique_size.get(), hop_size, stream));

      auto temp = allocator.alloc_unique<char>(temp_storage_bytes);

      CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
          temp.get(), temp_storage_bytes, hop_b.Current(), hop_unique.get(),
          hop_counts.get(), hop_unique_size.get(), hop_size, stream));

      device->CopyDataFromTo(
          hop_unique_size.get(), 0, &hop_uniq_size, 0, sizeof(hop_uniq_size),
          ctx, DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
    }

    thrust::lower_bound(
        exec_policy, hop_unique.get(), hop_unique.get() + hop_uniq_size, hop_1,
        hop_1 + hop_size, hop_new);
  }

  // @todo Consider creating a CSC because the SpMV will be done multiple times.
  COOMatrix rmat(
      num_rows, hop_uniq_size, idx_coo_arr, hop_new_arr, NullArray(), true,
      false);

  BcastOff bcast_off;
  bcast_off.use_bcast = false;
  bcast_off.out_len = 1;
  bcast_off.lhs_len = 1;
  bcast_off.rhs_len = 1;

  FloatArray probs_arr =
      NewFloatArray(hop_uniq_size, ctx, sizeof(FloatType) * 8);
  auto probs_1 = probs_arr.Ptr<FloatType>();
  FloatArray probs_arr_2 =
      NewFloatArray(hop_uniq_size, ctx, sizeof(FloatType) * 8);
  auto probs = probs_arr_2.Ptr<FloatType>();
  auto arg_u = NewIdArray(hop_uniq_size, ctx, sizeof(IdType) * 8);
  auto arg_e = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);

  double prev_ex_nodes = hop_uniq_size;

  for (int iters = 0; iters < importance_sampling || importance_sampling < 0;
       iters++) {
    if (weighted && iters == 0) {
      aten::cuda::SpMMCoo<
          IdType, FloatType, aten::cuda::binary::Mul<FloatType>,
          aten::cuda::reduce::Max<IdType, FloatType, true>>(
          bcast_off, rmat, cs_arr, A_l_arr, probs_arr_2, arg_u, arg_e);
    } else {
      aten::cuda::SpMMCoo<
          IdType, FloatType, aten::cuda::binary::CopyLhs<FloatType>,
          aten::cuda::reduce::Max<IdType, FloatType, true>>(
          bcast_off, rmat, cs_arr, NullArray(), iters ? probs_arr : probs_arr_2,
          arg_u, arg_e);
    }

    if (iters)
      thrust::transform(
          exec_policy, probs_1, probs_1 + hop_uniq_size, probs, probs,
          thrust::multiplies<FloatType>{});

    thrust::gather(
        exec_policy, hop_new, hop_new + hop_size, probs, probs_found);

    {
      constexpr int BLOCK_CTAS = BLOCK_SIZE / CTA_SIZE;
      // the number of rows each thread block will cover
      constexpr int TILE_SIZE = BLOCK_CTAS;
      const dim3 block(CTA_SIZE, BLOCK_CTAS);
      const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
      CUDA_KERNEL_CALL(
          (_CCGRowWiseLayerSampleDegreeKernel<
              IdType, FloatType, BLOCK_CTAS, TILE_SIZE>),
          grid, block, 0, stream, (IdType)num_picks, num_rows, rows, cs,
          weighted ? ds : nullptr, weighted ? d2s : nullptr,
          probs_found, A, subindptr);
    }

    {
      auto probs_min_1 =
          thrust::make_transform_iterator(probs, TransformOpMinWith1{});
      const double cur_ex_nodes = thrust::reduce(
          exec_policy, probs_min_1, probs_min_1 + hop_uniq_size, 0.0);
      if (cur_ex_nodes / prev_ex_nodes >= 1 - eps) break;
      prev_ex_nodes = cur_ex_nodes;
    }
  }
}

/////////////////////////////// CCG ///////////////////////////////

template <DGLDeviceType XPU, typename IdType, typename FloatType>
std::pair<COOMatrix, FloatArray> CCGLaborSampling(
    void *gpu_ccg, VertexID_t n_nodes, IdArray rows_arr, const int64_t num_picks,
    FloatArray prob_arr, const int importance_sampling, IdArray random_seed_arr,
    IdArray NIDs) {
  const bool weighted = !IsNullArray(prob_arr);

  const auto& ctx = rows_arr->ctx;

  runtime::CUDAWorkspaceAllocator allocator(ctx);

  const auto stream = runtime::getCurrentCUDAStream();
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);

  auto device = runtime::DeviceAPI::Get(ctx);

  const IdType num_rows = rows_arr->shape[0];
  IdType* const rows = rows_arr.Ptr<IdType>();
  IdType* const nids = IsNullArray(NIDs) ? nullptr : NIDs.Ptr<IdType>();
  FloatType* const A = prob_arr.Ptr<FloatType>();

//   IdType* const indptr = mat.indptr.Ptr<IdType>();
//   IdType* const indices = mat.indices.Ptr<IdType>();
  IdType* const data = nullptr;

  // compute in-degrees
  auto in_deg = allocator.alloc_unique<IdType>(num_rows + 1);
  // cs stands for c_s in arXiv:2210.13339
  FloatArray cs_arr = NewFloatArray(num_rows, ctx, sizeof(FloatType) * 8);
  auto cs = cs_arr.Ptr<FloatType>();
  // ds stands for A_{*s} in arXiv:2210.13339
  FloatArray ds_arr = weighted
                          ? NewFloatArray(num_rows, ctx, sizeof(FloatType) * 8)
                          : NullArray();
  auto ds = ds_arr.Ptr<FloatType>();
  // d2s stands for (A^2)_{*s} in arXiv:2210.13339, ^2 is elementwise.
  FloatArray d2s_arr = weighted
                           ? NewFloatArray(num_rows, ctx, sizeof(FloatType) * 8)
                           : NullArray();
  auto d2s = d2s_arr.Ptr<FloatType>();

  if (weighted) {
    auto b_offsets =
        thrust::make_transform_iterator(rows, IndptrFunc<IdType>{});
    auto e_offsets =
        thrust::make_transform_iterator(rows, IndptrFunc<IdType>{});

    auto A_A2 = thrust::make_transform_iterator(A, SquareFunc<FloatType>{});
    auto ds_d2s = thrust::make_zip_iterator(ds, d2s);

    size_t prefix_temp_size = 0;
    CUDA_CALL(cub::DeviceSegmentedReduce::Reduce(
        nullptr, prefix_temp_size, A_A2, ds_d2s, num_rows, b_offsets, e_offsets,
        TupleSum{}, thrust::make_tuple((FloatType)0, (FloatType)0), stream));
    auto temp = allocator.alloc_unique<char>(prefix_temp_size);
    CUDA_CALL(cub::DeviceSegmentedReduce::Reduce(
        temp.get(), prefix_temp_size, A_A2, ds_d2s, num_rows, b_offsets,
        e_offsets, TupleSum{}, thrust::make_tuple((FloatType)0, (FloatType)0),
        stream));
  }

  thrust::counting_iterator<IdType> iota(0);
  thrust::for_each(
      exec_policy, iota, iota + num_rows,
      DegreeFunc<IdType, FloatType>{
          (IdType)num_picks, rows, weighted ? ds : nullptr,
          in_deg.get(), cs});

  // fill subindptr
  IdArray subindptr_arr = NewIdArray(num_rows + 1, ctx, sizeof(IdType) * 8);
  auto subindptr = subindptr_arr.Ptr<IdType>();

  IdType hop_size;
  {
    size_t prefix_temp_size = 0;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, prefix_temp_size, in_deg.get(), subindptr, num_rows + 1,
        stream));
    auto temp = allocator.alloc_unique<char>(prefix_temp_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        temp.get(), prefix_temp_size, in_deg.get(), subindptr, num_rows + 1,
        stream));
    device->CopyDataFromTo(
        subindptr, num_rows * sizeof(hop_size), &hop_size, 0, sizeof(hop_size),
        ctx, DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
  }

  IdArray hop_arr = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  CSRMatrix smat(
      num_rows, n_nodes, subindptr_arr, hop_arr, NullArray(), false);
  // @todo Consider fusing CCGToCOO into StencilOpFused kernel
  auto smatcoo = CSRToCOO(smat, false);

  auto idx_coo_arr = smatcoo.row;
  auto idx_coo = idx_coo_arr.Ptr<IdType>();

  auto hop_1 = hop_arr.Ptr<IdType>();
  auto rands =
      allocator.alloc_unique<FloatType>(importance_sampling ? hop_size : 1);
  auto probs_found =
      allocator.alloc_unique<FloatType>(importance_sampling ? hop_size : 1);

  if (weighted) {
    // Recompute c for weighted graphs.
    constexpr int BLOCK_CTAS = BLOCK_SIZE / CTA_SIZE;
    // the number of rows each thread block will cover
    constexpr int TILE_SIZE = BLOCK_CTAS;
    const dim3 block(CTA_SIZE, BLOCK_CTAS);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CCGRowWiseLayerSampleDegreeKernel<
            IdType, FloatType, BLOCK_CTAS, TILE_SIZE>),
        grid, block, 0, stream, (IdType)num_picks, num_rows, rows, cs, ds, d2s,
        nullptr, A, subindptr);
  }

  const uint64_t random_seed =
      IsNullArray(random_seed_arr)
          ? RandomEngine::ThreadLocal()->RandInt(1000000000)
          : random_seed_arr.Ptr<int64_t>()[0];

  if (importance_sampling)
    compute_importance_sampling_probabilities<
        IdType, FloatType, decltype(exec_policy)>(
        n_nodes, hop_size, stream, random_seed, num_rows, rows, subindptr,
        idx_coo_arr, nids, cs_arr, weighted, A, ds, d2s,
        (IdType)num_picks, ctx, allocator, exec_policy, importance_sampling,
        hop_1, rands.get(), probs_found.get());

  IdArray picked_row = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  IdArray picked_col = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  IdArray picked_idx = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  FloatArray picked_imp =
      importance_sampling || weighted
          ? NewFloatArray(hop_size, ctx, sizeof(FloatType) * 8)
          : NullArray();

  IdType* const picked_row_data = picked_row.Ptr<IdType>();
  IdType* const picked_col_data = picked_col.Ptr<IdType>();
  IdType* const picked_idx_data = picked_idx.Ptr<IdType>();
  FloatType* const picked_imp_data = picked_imp.Ptr<FloatType>();

  auto picked_inrow = allocator.alloc_unique<IdType>(
      importance_sampling || weighted ? hop_size : 1);

  // Sample edges here
  IdType num_edges;
  {
    thrust::constant_iterator<FloatType> one(1);
    if (importance_sampling) {
      auto output = thrust::make_zip_iterator(
          picked_inrow.get(), picked_row_data, picked_col_data, picked_idx_data,
          picked_imp_data);
      if (weighted) {
        auto transformed_output = thrust::make_transform_output_iterator(
            output,
            TransformOpImp<
                IdType, FloatType, FloatType*, FloatType*, decltype(one)>{
                probs_found.get(), A, one, idx_coo, rows, cs, subindptr,
                data});
        auto stencil =
            thrust::make_zip_iterator(idx_coo, probs_found.get(), rands.get());
        num_edges =
            thrust::copy_if(
                exec_policy, iota, iota + hop_size, stencil, transformed_output,
                thrust::make_zip_function(StencilOp<FloatType>{cs})) -
            transformed_output;
      } else {
        auto transformed_output = thrust::make_transform_output_iterator(
            output,
            TransformOpImp<
                IdType, FloatType, FloatType*, decltype(one), decltype(one)>{
                probs_found.get(), one, one, idx_coo, rows, cs,
                subindptr, data});
        auto stencil =
            thrust::make_zip_iterator(idx_coo, probs_found.get(), rands.get());
        num_edges =
            thrust::copy_if(
                exec_policy, iota, iota + hop_size, stencil, transformed_output,
                thrust::make_zip_function(StencilOp<FloatType>{cs})) -
            transformed_output;
      }
    } else {
      if (weighted) {
        auto output = thrust::make_zip_iterator(
            picked_inrow.get(), picked_row_data, picked_col_data,
            picked_idx_data, picked_imp_data);
        auto transformed_output = thrust::make_transform_output_iterator(
            output,
            TransformOpImp<
                IdType, FloatType, decltype(one), FloatType*, FloatType*>{
                one, A, A, idx_coo, rows, cs, subindptr, data});
        const auto pred =
            StencilOpFused<IdType, FloatType, decltype(one), FloatType*>{
                random_seed, idx_coo, cs,     one,     A,
                subindptr,   rows,    nids};
        num_edges = thrust::copy_if(
                        exec_policy, iota, iota + hop_size, iota,
                        transformed_output, pred) -
                    transformed_output;
      } else {
        auto output = thrust::make_zip_iterator(
            picked_row_data, picked_col_data, picked_idx_data);
        auto transformed_output = thrust::make_transform_output_iterator(
            output, TransformOp<IdType>{
                        idx_coo, rows, subindptr, data});
        const auto pred =
            StencilOpFused<IdType, FloatType, decltype(one), decltype(one)>{
                random_seed, idx_coo, cs,     one,     one,
                subindptr,   rows,    nids};
        num_edges = thrust::copy_if(
                        exec_policy, iota, iota + hop_size, iota,
                        transformed_output, pred) -
                    transformed_output;
      }
    }
  }

  // Normalize edge weights here
  if (importance_sampling || weighted) {
    thrust::constant_iterator<IdType> one(1);
    // contains degree information
    auto ds = allocator.alloc_unique<IdType>(num_rows);
    // contains sum of edge weights
    auto ws = allocator.alloc_unique<FloatType>(num_rows);
    // contains degree information only for vertices with nonzero degree
    auto ds_2 = allocator.alloc_unique<IdType>(num_rows);
    // contains sum of edge weights only for vertices with nonzero degree
    auto ws_2 = allocator.alloc_unique<FloatType>(num_rows);
    auto output_ = thrust::make_zip_iterator(ds.get(), ws.get());
    // contains row ids only for vertices with nonzero degree
    auto keys = allocator.alloc_unique<IdType>(num_rows);
    auto input = thrust::make_zip_iterator(one, picked_imp_data);
    auto new_end = thrust::reduce_by_key(
        exec_policy, picked_inrow.get(), picked_inrow.get() + num_edges, input,
        keys.get(), output_, thrust::equal_to<IdType>{}, TupleSum{});

    {
      thrust::constant_iterator<IdType> zero_int(0);
      thrust::constant_iterator<FloatType> zero_float(0);
      auto input = thrust::make_zip_iterator(zero_int, zero_float);
      auto output = thrust::make_zip_iterator(ds_2.get(), ws_2.get());
      thrust::copy(exec_policy, input, input + num_rows, output);
      {
        const auto num_rows_2 = new_end.first - keys.get();
        thrust::scatter(
            exec_policy, output_, output_ + num_rows_2, keys.get(), output);
      }
    }

    {
      auto input =
          thrust::make_zip_iterator(picked_inrow.get(), picked_imp_data);
      auto transformed_input = thrust::make_transform_iterator(
          input, thrust::make_zip_function(TransformOpMean<IdType, FloatType>{
                     ds_2.get(), ws_2.get()}));
      thrust::copy(
          exec_policy, transformed_input, transformed_input + num_edges,
          picked_imp_data);
    }
  }

  picked_row = picked_row.CreateView({num_edges}, picked_row->dtype);
  picked_col = picked_col.CreateView({num_edges}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({num_edges}, picked_idx->dtype);
  if (importance_sampling || weighted)
    picked_imp = picked_imp.CreateView({num_edges}, picked_imp->dtype);

  // std::cout << "CCGSample COO Matrix: length: " << num_edges << std::endl;
  // gpuprint<<<1,1>>>(picked_row_data, num_edges);
  // cudaDeviceSynchronize();
  // gpuprint<<<1,1>>>(picked_col_data, num_edges);
  // cudaDeviceSynchronize();
  // gpuprint<<<1,1>>>(picked_idx_data, num_edges);
  // cudaDeviceSynchronize();
  // device->StreamSync(ctx, stream);

  return std::make_pair(
      COOMatrix(n_nodes, n_nodes, picked_col, picked_row, picked_idx),
      picked_imp);
}

template std::pair<COOMatrix, FloatArray>
CCGLaborSampling<kDGLCUDA, int64_t, float>(
    void *, VertexID_t, IdArray, int64_t, FloatArray, int, IdArray, IdArray);
template std::pair<COOMatrix, FloatArray>
CCGLaborSampling<kDGLCUDA, int64_t, double>(
    void *, VertexID_t, IdArray, int64_t, FloatArray, int, IdArray, IdArray);

} // namespace tcpdgl

}  // namespace dgl
