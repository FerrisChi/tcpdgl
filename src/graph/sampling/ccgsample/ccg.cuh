#ifndef __CCG_HPP__
#define __CCG_HPP__

typedef int64_t VertexID_t;
typedef int64_t EdgePos_t;
typedef uint32_t Graph_t;
typedef uint64_t Offset_t;

#define GRAPH_LEN 32

struct BCGPartition {
    ~BCGPartition() {}
//   const VertexID first_vertex_id;
//   const VertexID last_vertex_id;
    const Graph_t *graph;
    const Offset_t *offset;
    const VertexID_t n_nodes;
    const EdgePos_t n_edges;
    // prefix sum of deg
    const EdgePos_t *degoffset;

    __host__
    BCGPartition (const Graph_t *_graph, const Offset_t *_offset, const EdgePos_t *_degoffset, const VertexID_t _n_nodes, const EdgePos_t _n_edges) : 
                                graph(_graph), offset(_offset), n_nodes(_n_nodes), n_edges(_n_edges), degoffset(_degoffset) {}

};

struct GPUBCGPartition {
    BCGPartition* d_bcg;
    Graph_t* d_graph;
    Offset_t* d_offset;
    // prefix sum of deg
    EdgePos_t* d_degoffset;
    EdgePos_t* d_deg;
};

#endif
