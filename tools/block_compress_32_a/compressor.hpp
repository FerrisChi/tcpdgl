#ifndef CGR_COMPRESSOR_HPP
#define CGR_COMPRESSOR_HPP

#include <string>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cmath>
#include <deque>
#include <list>

using size_type = int64_t;
using bits = std::vector<bool>;

int T_D = 10;
int T_k = 32;
int T_h = -1;

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

    // append fix-length code x(length = max_bit) to bit_array
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

    // get significent bit position of x
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

enum bit_stat {
    E_OUTD, E_DESC, E_H, E_K, E_PPOS, E_BODY, E_TOT
};


class Compressor {

    size_type num_node;
    size_type num_edge;
    std::vector<std::pair<size_type, size_type>> edges;
    std::vector<std::vector<size_type>> adjlist;

    class NodeCompress {
public:
        size_type node;
        size_type outd;
        size_type data_len;
        int h, k;
        // first positive position
        int ppos;
        bits len_arr, data_arr;

        size_type bit_cnt[10];
  
        class LayerCompress {
            std::vector<int> layer_bit;
            std::vector<std::vector<size_type>> layer_cnode;
        
public:
            void init(int h) {
                layer_bit.clear();
                layer_bit.resize(h);
                layer_cnode.clear();
                layer_cnode.resize(h);
            }

            void push(int h, size_type v) {
                assert(h >= 1);
                layer_cnode[h - 1].push_back(v);
                layer_bit[h - 1] = std::max(layer_bit[h - 1], eh.get_bit_num(v));
            }

            void encode_len(bits &data) {
                for (auto b : layer_bit) {
                    assert(b <= 32);
                    eh.append_bit(data, b - 1, 5);
                }
            }

            void encode_data(bits &data) {
                for (int h = 0; h < layer_bit.size(); ++h) {
                    for (auto v : layer_cnode[h]) {
                        eh.append_bit(data, v, layer_bit[h]);
                    }
                }
            }

            void print(void) {
                for (int i = 0; i < layer_bit.size(); ++i) {
                    std::cout << i << " : (" << layer_bit[i] << ") ";
                    for (auto v : layer_cnode[i]) std::cout << v << ' ';
                    std::cout << std::endl;
                }
            }

        };
        LayerCompress lc;

        void set_tk(void) {
            if (outd < T_D) {
                h = 1; k = outd;
                return;
            }
            auto cal_nd = [](int _k, int _h) {
                return (pow(_k, _h + 1) - _k) / (_k - 1);
            };
            if (T_h >= 0) {
                h = T_h;
                // decltype(outd) sum = T_k * T_k + T_k, layer_num = T_k * T_k;
                // while (h < T_h && sum < outd) {
                //     layer_num *= T_k;
                //     sum += layer_num;
                //     ++h;
                // }
                k = pow(outd, 1.0 / h);
                if (k < 2) {
                    k = 2;
                    h = 1;
                    while (cal_nd(k, h) < outd) ++h;
                    
                    auto tmp = k * (1 - pow(k, h)) / (1 - k);
                    if (tmp < outd) {
                        std::cout << outd << ' ' << k << ' ' << h << std::endl;
                        std::cout << cal_nd(k, h) << ' ' << cal_nd(k, h + 1) << std::endl;
                    }
                    assert(tmp >= outd);
                }
                else {
                    while (cal_nd(k, h) < outd) ++k;
                    while (cal_nd(k - 1, h) >= outd) --k;
                    while (cal_nd(k, h - 1) >= outd) --h;
                    assert(k * (1 - pow(k, h)) / (1 - k) >= outd);
                    // if ((k - 1) * (1 - pow(k - 1, h)) / (2 - k) >= outd) std::cout << outd << ' ' << k << ' ' << h << std::endl;
                    if (k > 2) assert((k - 1) * (1 - pow(k - 1, h)) / (2 - k) < outd);
                    assert(cal_nd(k, h - 1) < outd);
                    assert(k > 1);
                }
            }            
            else {
                h = 2;
                decltype(outd) sum = T_k * T_k + T_k, layer_num = T_k * T_k;
                while (sum < outd) {
                    layer_num *= T_k;
                    sum += layer_num;
                    ++h;
                }
                // binary search suitable k
                int left = 1, right = T_k;
                while (left < right) {
                    auto x = (left + right) >> 1;
                    // k+k^2+...+k^h = k(k^h-1)/(k-1)
                    if (pow(x, h + 1) - x >= outd * (x - 1)) right = x; // < outd
                    else left = x + 1;
                }
                k = right;
                assert(k * (1 - pow(k, h)) / (1 - k) >= outd);
                assert(k > 1);
                assert((k - 1) * (1 - pow(k - 1, h)) / (2 - k) < outd);
            }
            // if (outd > T_k * T_k * T_k + T_k * T_k + T_k) std::cout << outd << ' ' << h << ' ' << k << std::endl;
            return;
        }

        void init(size_type _node, size_type _outd) {
            node = _node;
            outd = _outd;
            ppos = 0;
            set_tk();
            lc.init(h);
            return;
        }

        // sub_len: size without last layer nodes
        // leaf_len: maximum possible leaves
        void dfs(int cur_h, size_type x, size_type sub_len, size_type leaf_len, std::list<size_type> neighbors) {
            // std::cout << x << ' ' << cur_h << std::endl;
            auto tot_len = neighbors.size();
            if (tot_len <= k) {
                for (auto i : neighbors) lc.push(cur_h, abs(i - x) - 1);
                if (cur_h == 1) {
                    auto iter = neighbors.begin();
                    for (int i = 0; i < neighbors.size(); ++i, ++iter) {
                        if ((*iter) < x) ppos = i + 1;
                    }
                }
                return;
            }

            size_type res_len = tot_len - k * sub_len;
            auto iter = neighbors.begin();
            for (int i = 0; i < k; ++i) {
                auto iter_s = iter;
                // subtree size
                auto len = sub_len + std::min(res_len, leaf_len);
                size_type cur;
                size_type x_pos;
                if (k & 1) {
                    x_pos = (size_type)((pow(k, h - cur_h + 1) - k) / (k - 1) + h - cur_h) >> 1;
                }
                else x_pos = (sub_len + leaf_len) >> 1;
                // left leaves less than half of possible leave len, make sure it has actual node
                if (auto half_len = (leaf_len + (k & 1)) >> 1; res_len < half_len) {
                    x_pos -= half_len - res_len;
                }
                res_len = std::max(res_len - leaf_len, (size_type)0);
                if (x_pos < 0) {
                    std::cout << outd << ' ' << k << ' ' << h << " --- " << x_pos << std::endl;
                }
                assert(x_pos >= 0);
                assert(x_pos < len);
                for (int j = 0; j < len; ++j) {
                    if (j == x_pos) {
                        cur = *iter;
                        lc.push(cur_h, abs(cur - x) - 1);
                        if (cur_h == 1) {
                            if (cur < x) ppos = i + 1;
                        }
                        else {
                            if (i <= ((k - 1) >> 1)) {
                                if (cur >= x) {
                                    std::cout << this->node << ' ' << outd << ' ' << h << ',' << k << std::endl;
                                    std::cout << cur_h << ' ' << len << std::endl;
                                    std::cout << sub_len << "---" << leaf_len << std::endl;
                                    std::cout << x << ' ' << cur << std::endl;
                                    std::cout << i << ' ' << ((k - 1) >> 1) << std::endl;
                                    std::cout << x_pos << std::endl;
                                }
                                assert(cur < x);
                            }
                            else assert(cur > x);
                        }
                        iter = neighbors.erase(iter);
                    }
                    else ++iter;
                }
                if (len > 1)
                    dfs(cur_h + 1, cur, (sub_len - 1) / k, leaf_len / k, std::list<size_type>(iter_s, iter));
            }
        }

        void encode_desc(void) {
            assert(data_arr.empty());
            eh.append_gamma(data_arr, outd);
            bit_cnt[E_OUTD] = data_arr.size();
            
            if (T_h >= 0) {
                assert(k >= 1);
                eh.append_gamma(data_arr, k);
                bit_cnt[E_K] = data_arr.size() - bit_cnt[E_OUTD];
                assert(h >= 1);
                assert(h <= T_h);
                eh.append_bit(data_arr, h - 1, eh.get_bit_num(T_h - 1));
                bit_cnt[E_H] = eh.get_bit_num(T_h - 1);
            }
            else {
                assert(k >= 1);
                assert(k <= T_k);
                eh.append_bit(data_arr, k - 1, eh.get_bit_num(T_k - 1));
                bit_cnt[E_K] = eh.get_bit_num(T_k - 1);
                assert(h >= 1);
                assert(h <= -T_h);
                eh.append_bit(data_arr, h - 1, eh.get_bit_num(-1 - T_h));
                bit_cnt[E_K] = eh.get_bit_num(-1 - T_h);
            }

            assert(ppos >= 0);
            assert(ppos <= k);
            eh.append_bit(data_arr, ppos, eh.get_bit_num(k));
            bit_cnt[E_PPOS] = eh.get_bit_num(k);
            bit_cnt[E_DESC] = data_arr.size();
            lc.encode_len(data_arr);
            return;
        }

        void encode_data(void) {
            lc.encode_data(data_arr);
            bit_cnt[E_TOT] = data_arr.size();
            bit_cnt[E_BODY] = bit_cnt[E_TOT] - bit_cnt[E_DESC];
            return;
        }

        void encode_offset(void) {
            data_len = data_arr.size();
            eh.append_bit(len_arr, data_len, eh.get_bit_num(data_len));
        }

        void encode_this(void) {
            len_arr.clear();
            data_arr.clear();
            encode_desc();
            encode_data();
            encode_offset();
        }
    };
    std::vector<NodeCompress> cnodes;

public:
    Compressor() : num_node(0), num_edge(0) {}
    Compressor(int _K, int _H) : num_node(0), num_edge(0) {
        if (_K > 0) T_k = _K;
        if (_H > 0) {
            T_h = _H;
            // T_D = pow(2, T_h) - 1;
        }
    }

    bool load_graph(const std::string &file_path) {
        FILE *f = fopen(file_path.c_str(), "r");
        if (f == 0) {
            std::cout << "file cannot open!" << std::endl;
            abort();
        }
        size_type u = 0, v = 0;
        this->num_node = 0;
        while (fscanf(f, "%ld %ld", &u, &v) > 0) {
            assert(u >= 0);
            assert(v >= 0);
            this->num_node = std::max(this->num_node, u + 1);
            this->num_node = std::max(this->num_node, v + 1);
            if (!this->edges.empty()) {
                assert(this->edges.back().first <= u);
                if (this->edges.back().first == u) {
                    if (this->edges.back().second == v) continue;
                    assert(this->edges.back().second < v);
                }
            }
            this->edges.emplace_back(std::pair<size_type, size_type>(u, v));
        }
        this->num_edge = this->edges.size();
        this->adjlist.resize(this->num_node);
        for (auto edge : this->edges) {
            this->adjlist[edge.first].emplace_back(edge.second);
        }
        fclose(f);
        return true;
    }

    bool write_cgr(const std::string &dir_path) {
        bits graph;

        FILE *of_graph = fopen((dir_path + ".graph").c_str(), "w");

        if (of_graph == 0) {
            std::cout << "graph file cannot create!" << std::endl;
            abort();
        }

        this->wrtie_data(of_graph);
        fclose(of_graph);
    
        FILE *of_offset = fopen((dir_path + ".offset").c_str(), "w");

        if (of_offset == 0) {
            std::cout << "graph file cannot create!" << std::endl;
            abort();
        }

        this->write_offset3(of_offset);
        fclose(of_offset);

        return true;
    }

    void wrtie_data(FILE* &of) {
        std::vector<unsigned char> buf;

        unsigned char cur = 0;
        int bit_count = 0;

        for (size_type i = 0; i < this->num_node; i++) {
            for (auto bit : this->cnodes[i].data_arr) {
                cur <<= 1;
                if (bit) cur++;
                bit_count++;
                if (bit_count == 8) {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }
        }

        if (bit_count) {
            while (bit_count < 8) cur <<= 1, bit_count++;
            buf.emplace_back(cur);
        }

        fwrite(buf.data(), sizeof(unsigned char), buf.size(), of);
    }

    void encode_node_s1(const size_type node) {
        auto &cnode = this->cnodes[node];
        auto &_neighbors = this->adjlist[node];
        std::list<size_type> neighbors(_neighbors.begin(), _neighbors.end());  // std::vector<std::vector<size_type>> adjlist
        cnode.init(node, neighbors.size());
        if (cnode.outd == 0) return;
        // std::cout << node << "-------" << cnode.outd << std::endl;
        cnode.dfs(1, node, (pow(cnode.k, cnode.h - 1) - 1) / (cnode.k - 1), pow(cnode.k, cnode.h - 1), neighbors);
        cnode.encode_this();
        return;
    }

    void compress(void) {
        std::cout << "Thresold_K = " << T_k << ", Thresold_H = " << T_h << std::endl;
        eh.pre_encoding();
        this->cnodes.clear();
        this->cnodes.resize(this->num_node);

// #pragma omp parallel for
        for (size_type i = 0; i < this->num_node; i++) {
            encode_node_s1(i);
        }

        size_type sum_bit[10] = {0};
        for (size_type i = 0; i < this->num_node; i++) {
            sum_bit[E_OUTD] += cnodes[i].bit_cnt[E_OUTD];
            sum_bit[E_K] += cnodes[i].bit_cnt[E_K];
            sum_bit[E_H] += cnodes[i].bit_cnt[E_H];
            sum_bit[E_PPOS] += cnodes[i].bit_cnt[E_PPOS];
            sum_bit[E_DESC] += cnodes[i].bit_cnt[E_DESC];
            sum_bit[E_BODY] += cnodes[i].bit_cnt[E_BODY];
            sum_bit[E_TOT] += cnodes[i].bit_cnt[E_TOT];
        }

        std::cout << "-------- STAT INFO --------" << std::endl;
        std::cout << "OUTD BIT\t" << sum_bit[E_OUTD] << "\t" << 1.0 * sum_bit[E_OUTD] / sum_bit[E_TOT] << std::endl;
        std::cout << "   K BIT\t" << sum_bit[E_K] << "\t" << 1.0 * sum_bit[E_K] / sum_bit[E_TOT] << std::endl;
        std::cout << "   H BIT\t" << sum_bit[E_H] << "\t" << 1.0 * sum_bit[E_H] / sum_bit[E_TOT] << std::endl;
        std::cout << "PPOS BIT\t" << sum_bit[E_PPOS] << "\t" << 1.0 * sum_bit[E_PPOS] / sum_bit[E_TOT] << std::endl;
        std::cout << "DESC BIT\t" << sum_bit[E_DESC] << "\t" << 1.0 * sum_bit[E_DESC] / sum_bit[E_TOT] << std::endl;
        std::cout << "BODY BIT\t" << sum_bit[E_BODY] << "\t" << 1.0 * sum_bit[E_BODY] / sum_bit[E_TOT] << std::endl;
        std::cout << " TOT BIT\t" << sum_bit[E_TOT] << std::endl;

        return;
    }

    void write_offset3(FILE* &of) {
        // binary + min_len + sum
        size_type last_offset = 0;

        std::vector<unsigned char> buf;
        unsigned char cur = 0;
        int bit_count = 0;
        int max_len = 0;

        for (size_type i = 0; i < this->num_node; i++) {
            if (this->cnodes[i].data_len > max_len)
                max_len = this->cnodes[i].data_len;
        }
        // std::cout << max_len << ' ' << eh.get_bit_num(max_len) << std::endl;
        max_len = eh.get_bit_num(max_len);

        bits len_bit;
        eh.append_bit(len_bit, max_len, 8);
        for (auto bit : len_bit) {
            cur <<= 1;
            if (bit) cur++;
            bit_count++;
            if (bit_count == 8) {
                buf.emplace_back(cur);
                cur = 0;
                bit_count = 0;
            }
        }
        
        for (size_type i = 0; i < this->num_node; i++) {
            // if (i < 32) std::cout << i << ' ' << this->cnodes[i].data_len << std::endl;
            auto len = this->cnodes[i].data_len;
            auto bit_cnt = eh.get_bit_num(len);

            for (int i = bit_cnt; i < max_len; ++i) {
                cur <<= 1;
                bit_count++;
                if (bit_count == 8) {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }

            for (auto bit : this->cnodes[i].len_arr) {
                cur <<= 1;
                if (bit) cur++;
                bit_count++;
                if (bit_count == 8) {
                    buf.emplace_back(cur);
                    cur = 0;
                    bit_count = 0;
                }
            }
        }

        if (bit_count) {
            while (bit_count < 8) cur <<= 1, bit_count++;
            buf.emplace_back(cur);
        }

        std::cout << buf.size() << std::endl;
        fwrite(buf.data(), sizeof(unsigned char), buf.size(), of);
    }
};

#endif /* CGR_COMPRESSOR_HPP */