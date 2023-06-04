#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdio>

#define MIN_LEN 1

class node {
    std::vector <uint32_t> neighbor, resv;
    std::vector <std::pair<uint32_t, uint32_t>> ints;

public:
    inline void add(uint32_t v) {
        neighbor.push_back(v);
    }

    void filter(void) {
        auto iter = neighbor.begin();
        uint32_t st = 0;
        while (st < neighbor.size()) {
            auto ed = st + 1;
            while (ed < neighbor.size() && neighbor[ed - 1] + 1 == neighbor[ed]) ++ed;
            auto len = ed - st;
            if (len > MIN_LEN) ints.push_back(std::make_pair(neighbor[st], len));
            else resv.push_back(neighbor[st]);
            st = ed;
        }
    }

    void output_v(uint32_t v, std::ofstream& fout_int, std::ofstream& fout_res) {
        if (!ints.empty()) {
            fout_int << v << ' ' << ints.size() << ' ';
            for (auto i : ints) fout_int << i.first << ' ' << i.second << ' ';
            fout_int << std::endl;
        }
        if (!resv.empty()) {
            // fout_res << v << ' ' << resv.size() << ' ';
            // for (auto v : resv) fout_res << v << ' ';
            // fout_res << std::endl;
            for (auto _v : resv) fout_res << v << ' ' << _v << std::endl;
        }
        return;
    }
};

int main(int argc,char *argv[]) {
    if (argc != 3) {
        printf("incorrect arguments.\n");
        printf("<input_path> <output_path>\n");
        abort();
    }

    std::string input_path(argv[1]);
    std::string output_path(argv[2]);

    uint32_t u, v, v_num;
    std::vector<node> vertices;
    std::ifstream fin;
    
    fin.open(input_path);

    while (fin >> u >> v) {
        if (vertices.size() < u + 1) vertices.resize(u + 1);
        vertices[u].add(v);
    }
    v_num = u + 1;

    fin.close();
    printf("Begin Filter.\n");

#pragma omp parallel for
    for (uint32_t i = 0; i < v_num; ++i) {
        vertices[i].filter();
    }

    printf("Begin Write.\n");
    
    std::ofstream fout_int, fout_res;
    fout_int.open(output_path + "_int.txt");
    fout_res.open(output_path + "_res.txt");
    for (uint32_t i = 0; i < v_num; ++i) vertices[i].output_v(i, fout_int, fout_res);
    fout_int.close();
    fout_res.close();

    return 0;
}
