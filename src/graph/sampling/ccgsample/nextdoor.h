#ifndef NEXTDOOR_H
#define NEXTDOOR_H

#include "ccg.cuh"
#include <curand_kernel.h>
#include <vector>

namespace dgl
{

namespace tcpdgl
{

class CCGSample {};

class NextDoorData {
public:
  std::vector<CCGSample> samples;
  std::vector<VertexID_t> initialContents;
  std::vector<VertexID_t> initialTransitToSampleValues;
  std::vector<int> devices;

  //Per Device Data.
  std::vector<CCGSample*> dOutputSamples;
  std::vector<VertexID_t*> dSamplesToTransitMapKeys;
  std::vector<VertexID_t*> dSamplesToTransitMapValues;
  std::vector<VertexID_t*> dTransitToSampleMapKeys;
  std::vector<VertexID_t*> dTransitToSampleMapValues;
  std::vector<EdgePos_t*> dSampleInsertionPositions;
  std::vector<EdgePos_t*> dNeighborhoodSizes;
  std::vector<curandState*> dCurandStates;
  std::vector<size_t> maxThreadsPerKernel;
  std::vector<VertexID_t*> dFinalSamples;
  // std::vector<VertexID_t*> dInitialSamples;
  VertexID_t INVALID_VERTEX;
  int maxBits;
  // std::vector<GPUBCGPartition> gpuBCGPartitions;

  VertexID_t n_nodes;
  EdgePos_t n_edges;
  VertexID_t sampleNum;
  int maxNeighborsToSample;
  int finalSampleSize;
  size_t totalMaxNeighbor;
  size_t totalFinalSample; 

  void setNumber(VertexID_t _n_nodes, VertexID_t _sampleNum, const std::vector<int> &fanouts) {
    n_nodes = _n_nodes;
    sampleNum = _sampleNum;
    maxNeighborsToSample = 1;
    finalSampleSize = 0;
    for (auto fanout : fanouts) {
      maxNeighborsToSample *= (fanout + 1); // save src nodes in result
      finalSampleSize += maxNeighborsToSample;
    }
    totalMaxNeighbor = 1ll * sampleNum * maxNeighborsToSample;
    totalFinalSample = 1ll * sampleNum * finalSampleSize;
    return;
  }

  NextDoorData() = default;
};

}
}
#endif