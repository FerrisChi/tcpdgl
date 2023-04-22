#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdio>

#define MIN_LEN 1

typedef vtype uint32_t;

class EncodeHelper {
    const int PRE_ENCODE_NUM = 16 * 1024 * 1024;
    std::vector<bits> gamma_code;

public:
    template <typename type>
    int get_bit_num(type num) {
        if (num == 0) return 1;
        int ret = 0;
        while (num) {num >>= 1; ++ret;}
        return ret;
    }

    void encode(bits &bit_array, size_type x, int len) {
        for (int i = len - 1; i >= 0; i--) {
            bit_array.emplace_back((x >> i) & 1L);
        }
    }

    void encode_gamma(bits &bit_array, size_type x) {
        x++;
        assert(x >= 0);
        int len = this->get_significent_bit(x);
        this->encode(bit_array, 1, len + 1);
        this->encode(bit_array, x, len);
    }

    void append_gamma(bits &bit_array, size_type x) {
        if (x < this->PRE_ENCODE_NUM) {
            bit_array.insert(bit_array.end(), this->gamma_code[x].begin(), this->gamma_code[x].end());
        } else {
            encode_gamma(bit_array, x);
        }
    }

    void append_bit(bits &bit_array, size_type x, int max_bit) {
        assert(x >= 0);
        auto bit_cnt = get_bit_num(x);
        for (int i = bit_cnt; i < max_bit; ++i) bit_array.emplace_back(0);
        if (x == 0) bit_array.emplace_back(0);
        else {
            size_type mask = 1 << (bit_cnt - 1);
            while (mask) {
                bit_array.emplace_back((mask & x) != 0);
                mask >>= 1;
            }
        }
        return;
    }

    int get_significent_bit(size_type x) {
        assert(x > 0);
        int ret = 0;
        while (x > 1) x >>= 1, ret++;
        return ret;
    }

    void pre_encoding() {
        this->gamma_code.clear();
        this->gamma_code.resize(this->PRE_ENCODE_NUM);
#pragma omp parallel for
        for (size_type i = 0; i < this->PRE_ENCODE_NUM; i++) {
            encode_gamma(this->gamma_code[i], i);
        }
    }
}eh;

class Vertex {
    vtype v, n_int;
    std::vector<std::pair<vtype, vtype>> ints;

    struct CPInt {
        vtype v1, v2, len;

        void init(vtype _v1, vtype _v2, vtype _len, vtype L) {
            assert(v1 != v2);
            if (len > (L >> 1)) {
                swap(v1, v2);
                _len = L - _len;
            }
            assert(_len <= (L >> 1));
            v1 = _v1;
            v2 = _v2;
            len = _len;
            return;
        }
    };
    std::vector<CPInt> cpints;

public:
    void init(vtype _v, vtype _n, ifstream& fin) : v(_v) {
        vtype head, len;
        for (int i = 0; i < _n; ++i) {
            fin >> head >> len;
            if (len >= T_LEN)
                ints.push_back(std::make_pair(head, len));
        }
        return;
    }

    void compress(void) {
        n_int = ints.size();
        vtype L = 0, light = -1, heavy = -1;
        for (auto i : ints) {
            L += i.second;
            i.second *= n_int;
        }

        for (auto i : ints) {
            if (i.second <= L) break;
            ++light;
        }

        for (auto i : ints) {
            if (i.second > L) break;
            ++heavy;
        }

        auto _len = ints[heavy].second;

        while (heavy < n_int) {
            CPInt cpint;
            if (_len > L) {
                auto lint = ints[light];
                cpint.init(lint.first, lint.second, ints[heavy].first, L);
                _len = _len - (L - lint.second);
                ints[heavy].first += (L - lint.second);
                while (ints[++light].second > L);
            }
            else {
                auto lint = ints[heavy];
                while (ints[++heavy].second <= L);
                cpint.init(lint.first, lint.second, ints[heavy].first, L);
                _len = ints[heavy].second - (L - ints[light].second);
                ints[heavy].first += (L - lint.second);
            }
            cpints.push_back(cpint);
        }

        return;
    }

    void write(ofstream &fout) {
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

    uint32_t u, n;
    Vertex vertex;
    std::vector<node> vertices;
    std::ifstream fin;
    
    fin.open(input_path);

    while (fin >> u >> n) {
        vertex.init(u, n, fin);
        vertices.push_back(vertex);
        ++v_num;
    }

    fin.close();
    printf("Begin Compress.\n");

#pragma omp parallel for
    for (uint32_t i = 0; i < v_num; ++i) {
        vertices[i].compress();
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
