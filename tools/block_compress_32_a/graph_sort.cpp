#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

struct Edge {
    int u, v;

    bool operator==(const Edge i) {
        return (i.u == this->u) && (i.v == this->v);
    }
};

inline bool cmp(Edge a, Edge b) {
    if (a.u == b.u) return a.v < b.v;
    return a.u < b.u;
}

int main(int argc,char *argv[]) {
    if (argc < 5) {
        printf("incorrect arguments.\n");
        printf("<input_path> <output_path> <u/d> <map_path> [<skip_line>]\n");
        printf("u means to add inverse edge to edge list.\n");
        abort();
    }

    int sl = 0;
    bool directed = argv[3][0] == 'd';
    if (argc == 6) sl = atoi(argv[5]);

    std::string input_path(argv[1]);
    std::string output_path(argv[2]);
    std::string map_path(argv[4]);

    // map origin id to now id
    map<int, int> o2n;
    int tot = 0;
    Edge edge;
    vector<Edge> edges;
    ifstream fin;
    char ch;
    
    fin.open(input_path);
    printf("Skip %d Lines\n", sl);
    while (sl--) {
        char buf[500];
        fin.getline(buf, 500);
    }
    printf("Reading file from %s\n", input_path.c_str());
    while (fin >> edge.u >> edge.v) {
        // printf("%d %d\n", edge.u, edge.v);
        if (edge.u == edge.v) continue;
        if (o2n.count(edge.u) == 0) o2n[edge.u] = tot++;
        if (o2n.count(edge.v) == 0) o2n[edge.v] = tot++;
        edge.u = o2n[edge.u];
        edge.v = o2n[edge.v];
        if (!directed) {
            if (edge.u > edge.v) swap(edge.u, edge.v);
        }
        edges.emplace_back(edge);
        // printf("%d %d\n", edge.u, edge.v);
    }
    fin.close();

    printf("Sorting %ld edges.\n", edges.size());
    sort(edges.begin(), edges.end(), cmp);
    if (!directed) {
        printf("Adding inverse edges.");
        edges.erase(unique(edges.begin(), edges.end()), edges.end());
        int len = edges.size();
        for (int i = 0; i < len; ++i) {
            edge = edges[i];
            swap(edge.u, edge.v);
            edges.emplace_back(edge);
        }
        sort(edges.begin(), edges.end(), cmp);
    }

    printf("Output to %s", output_path.c_str());
    ofstream fout;
    fout.open(output_path);
    for (auto e : edges) fout << e.u << ' ' << e.v << '\n';
    fout.close();

    fout.open(map_path);
    for(auto &[bef, now]: o2n) fout << bef << ' ' << now << '\n';
    fout.close();

    std::cout << "Graph has " << tot << " vertices, " << edges.size() << " edges.\n";
    
    return 0;
}
