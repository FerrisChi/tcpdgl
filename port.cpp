Vertex next(Sample s, Array<Vertex> transits, Array<Edge> srcEdges, int step);
int steps();
int sampleSize(int step);
bool unique(int step);
enum SamplingType {NeighborSampling, RandomWalkSampling, Collective};
SamplingType samplinType();
void initStepSample();
void procStepSample();


// layer
Vertex next(s, transits, srcEdges, step) {
    for(trn: transits)
        for(v: trn.out_nodes) {
            float rnd = randFloat(0,1);
            if(rnd <= trn.weight * weight(trn, v)) s.addEdge(step, trn, v);}
    return s;}
int steps() {return 2;}
int sampleSize(int step) {return (step==0)?25:10;}
bool unique(int step) {return false;}
SamplingType samplingType() {return SamplingType::Collective;}
void initStepSample() {step_COO = new step_COO();}
void procStepSample() {vecCOO.push(step_COO);}


// graphsage
Vertex next(s, transits, srcEdges, step) {
    int idx = randInt(0, srcEdges.size());
    return srcEdges[idx];}
int steps() {return 2;}
int sampleSize(int step) {return (step==0)?25:10;}
bool unique(int step) {return true;}
SamplingType samplingType() {return SamplingType::NeighborSampling;}
void initStepSample() {step_COO = new step_COO();}
void procStepSample() {vecCOO.push(step_COO);}

// deepwalk
Vertex next(s, transits, srcEdges, step) {
    int idx = randInt(0, srcEdges.size());
    return srcEdges[idx];}
int steps() {return 80;}
int sampleSize(int step) {return 1;}
bool unique(int step) {return false;}
SamplingType samplingType() {return SamplingType::RandomWalkSampling;}
void initStepSample() {}
void procStepSample() {}

