#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

struct Edge {
    int u, v;
};

inline bool cmp(Edge a, Edge b) {
    if (a.u == b.u) return a.v < b.v;
    return a.u < b.u;
}
int main(int argc,char *argv[]) {
    if (argc < 3) {
        printf("incorrect arguments.\n");
        printf("<input_path> <output_path> [<skip_line>] [w(eight)]\n");
        abort();
    }

    int sl = 0;
    bool out_weight = 1;
    if (argc >= 4) sl = atoi(argv[3]);
    if (argc == 5) out_weight = argv[3][0] == 'w';

    std::string input_path(argv[1]);
    std::string output_path(argv[2]);
    Edge edge;
    vector<Edge> edges;
    ifstream fin;
    
    fin.open(input_path);
    printf("Skip %d Lines\n", sl);
    while (sl--) {
        char buf[500];
        fin.getline(buf, 500);
    }
    while (fin >> edge.u >> edge.v)
        if (edge.u != edge.v) edges.push_back(edge);

    fin.close();

    ofstream fout;
    float weight = 1.0;
    fout.open(output_path, std::ofstream::binary);
    for (auto e : edges) {
        fout.write(reinterpret_cast<const char*>(&e.u), sizeof(int));
        fout.write(reinterpret_cast<const char*>(&e.v), sizeof(int));
        if (out_weight) fout.write(reinterpret_cast<const char*>(&weight), sizeof(float));
    }
    fout.close();

    return 0;
}