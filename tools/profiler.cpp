/*
运行： 
  ~/ananconda3/envs/cpdgl/bin/python3 ~/Program/cpdgl/example/pytorch/graphsage/python3 test.py | ~/Program/cpdgl/tools/profiler d s
  d 表示输出详细内容（details），s 表示输出总结内容（summary），包括均值、最大、最小。

代码内计时：
  [PF] start/stop time_stamp
  启动与停止，可以没有。会在[max(0, start_stamp), min(stop_stamp, +oo)]之间的内容。start、stop暂时只能使用一次。记录多个区间你可以修改一下。
  
  [PF] bg/ed/it name time_stamp
  [PF]是前缀；bg/ed代表开始/结束计时，会统计name相同的两个时间戳之间的时间；it表示记录迭代，会在details中输出该时间戳。

  Python Example
    print('[PF] bg py_call_sampling', time.time())
    ...
    print('[PF] ed py_call_sampling', time.time())

  Cpp Example
    auto _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "[PF] bg call_sampling " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
    ...
    _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "[PF] ed call_sampling " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
*/


#include <string>
#include <iostream>
#include <deque>
#include <algorithm>
#include <iomanip>
#include <map>
#include <cassert>

enum {
    EMPTY, BEGIN, END, ITER, PERIOD
};

struct statmp{
    std::string name;
    double stat;
    int count;

    statmp() {
        name = "";
        stat = 0;
        count = 0;
    }

    statmp(const std::string &_name, const double &_stat, int _count = 0) : name(_name), stat(_stat), count(_count) {}

};

struct statq{
    std::string name;
    std::deque<double> stats;
    int count;
    
    statq() {
        name = "";
        stats.clear();
        count = 0;
    }

    statq(const statmp &_statmp) : name(_statmp.name), count(_statmp.count){
        stats.clear();
        stats.push_back(_statmp.stat);
    }

    void print() {
        std::cout << (this->name) << " called " << (this->count) << " times\n[";
        for (auto &stat: stats) {
            std::cout << std::fixed << std::setprecision(7) << stat << " ";
        }
        std::cout << "]\n";
    }
};

struct stamp {
    std::string name;
    double tsp;
    int label;

    stamp() {
        name = "";
        tsp = -1.0;
        label = EMPTY;
    }

    stamp(const std::string& _name, const std::string &_tsp, const int _label) : name(_name), label(_label) {
        tsp = std::stod(_tsp);
        // printf("\t%lf\n",tsp);
    }

    stamp(const std::string& _it) {
        name = _it;
        label = ITER;
    }
};

struct period {
    std::string name;
    double tsp_bg, tsp_ed;
    int label;

    period(stamp it) : name(it.name), tsp_bg(it.tsp), label(ITER) {}
    period(stamp stp1, stamp stp2) {
        if (stp1.tsp > stp2.tsp) std::swap(stp1, stp2);

        this->name = stp1.name;
        this->tsp_bg = stp1.tsp;
        this->tsp_ed = stp2.tsp;
        this->label = PERIOD;
    }

    void print(void) {
        if (this->label == ITER) {
            std::cout << "---------- ITER - " << (this->name) << " ----------" << "\n";
        }

        if (this->label == PERIOD) {
            std::cout << (this->name) << " : [" << std::fixed << std::setprecision(7) << (this->tsp_bg) << " , " << std::fixed << std::setprecision(7) << (this->tsp_ed) << "] | LastSeconds = " << std::fixed << std::setprecision(7) << (this->tsp_ed) - (this->tsp_bg) << "\n"; 
        }
    }
};

inline bool cmp_s(stamp a, stamp b) {
    return a.tsp < b.tsp;
}

inline bool cmp_op(period a, period b) {
    return a.tsp_bg < b.tsp_bg;
}

inline bool cmp_stat(statmp a, statmp b) {
    return a.name < b.name;
}

std::deque<std::string> get_split(std::string str) {
    std::deque<std::string> ret;
    std::string substr = "";
    auto len = str.size();
    for (auto c : str) {
        if (c == ' ') {
            ret.push_back(substr);
            substr = "";
        }
        else {
            substr = substr + c;
        }
    }
    if (substr != "") ret.push_back(substr);
    return ret;
}

int main(int argc, char** argv) {
    bool OP_DETAILS = 0, OP_SUMMARY = 0, OP_ANALIZE = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == 'd') OP_DETAILS = 1;
        if (argv[i][0] == 's') OP_SUMMARY = 1;
        if (argv[i][0] == 'a') OP_ANALIZE = 1;
    }

    if (OP_DETAILS == 0 && OP_SUMMARY == 0) {
        std::cout << "No output, profiler exits." << "\n";
        std::cout << "[Usage] profile [d : output_details] [s : output_summary]" << "\n";
        return 0;
    }

    std::string str;
    std::deque<stamp> timps;
    std::deque<statmp> statmps;
    double l_pd = -1.0, r_pd = -1.0; // TODO: Extend to more periods.
    double stat;
    int run_flag = 1;
    while (std::getline(std::cin, str)) {
        auto strs = get_split(str);
        if (strs[0] != "[PF]") {
            // std::cout << str << "\n";
            if (strs[0] == "end") break;
            continue;
        }
        if (strs[1] == "start") {l_pd = std::stod(strs[2]); run_flag = 1; continue;}
        if (strs[1] == "stop") {r_pd = std::stod(strs[2]); run_flag = 0; continue;}
        if (!run_flag) continue;

        const std::string cur_name = strs[2];
        if (strs[1] == "it")
            timps.push_back(stamp(cur_name, strs[3], ITER));
        
        else if (strs[1] == "bg") 
            timps.push_back(stamp(cur_name, strs[3], BEGIN));

        else if (strs[1] == "ed") 
            timps.push_back(stamp(cur_name, strs[3], END));

        else if (strs[1] == "stat"){
            stat = std::stod(strs[3]);
            statmps.push_back(statmp(cur_name, stat, 1));
        }
    }

// ============================ANALYSIS============================

    std::sort(timps.begin(), timps.end(), cmp_s);
    std::deque<period> outq;

    // while (l_pd >= 0 && timps.front().tsp < l_pd) timps.pop_front();
    
    for (int i = 0;  i < timps.size(); ++i) {
        const auto &_stamp = timps[i];
        // if (l_pd >= 0 && _stamp.tsp < l_pd) continue;
        // if (r_pd >= 0 && _stamp.tsp > r_pd ) break;

        if (_stamp.label == ITER) {
            outq.push_back(period(_stamp));
        }

        if (_stamp.label == END) {
            for (int j = i - 1; j >= 0; --j) {
                if (timps[j].name == _stamp.name && timps[j].label == BEGIN) {
                    // std::deque<std::pair<std::string, double>> desb;
                    // desb.push_back(std::make_pair("Begin at", timps[j].tsp));
                    // desb.push_back(std::make_pair("Last", _stamp.tsp - timps[j].tsp));
                    // desb.push_back(std::make_pair("End at", _stamp.tsp));
                    // outq.push_back(period(_stamp.name, timps[j].tsp, desb));
                    outq.push_back(period(timps[j], _stamp));
                    timps[j].label = EMPTY;
                }
            }
        }
    }

    std::sort(statmps.begin(), statmps.end(), cmp_stat);
    std::deque<statq> outstatqs;
    for (int i = 0; i < statmps.size();) {
        auto &nowstatmp = statmps[i];
        statq nowstatq(nowstatmp);
        for(i++; i < statmps.size(); i++) {
            if(statmps[i].name != nowstatmp.name) break;
            nowstatq.stats.push_back(statmps[i].stat);
            nowstatq.count ++;
        }
        outstatqs.push_back(nowstatq);
    }

    if (outq.empty() && outstatqs.empty()) {
        std::cout << "Nothing to profile." << "\n";
        return 0;
    }
    
    std::sort(outq.begin(), outq.end(), cmp_op);
    // printf("outq: %ld timps, %ld periods\n", timps.size(), outq.size());
    // printf("statmps: %ld statmps, %ld statqs\n", statmps.size(), outstatqs.size());

// ============================PRINT============================
    if (OP_DETAILS) {
        std::cout << "===== DETAILS =====" << "\n";
        for (auto op : outq) op.print();
        std::cout << "\n";

        for (auto op : outstatqs) op.print();
        std::cout << "\n";
    }

    if (OP_SUMMARY) {
        std::map<std::string, std::deque<period>> op_map;
        for (auto op : outq) {
            if (op.label != PERIOD) continue;
            
            if (op_map.find(op.name) == op_map.end()) {
                std::deque<period> _ops = {op};
                op_map[op.name] = _ops;
            }
            else {
                op_map[op.name].push_back(op);
            }
        }

        if (op_map.empty() && outstatqs.empty()) {
            std::cout << "Nothing to summary." << "\n";
            return 0;
        }

        std::cout << "===== SUMMARY =====" << "\n";
        if(!op_map.empty()) {
            std::cout << "===== {TIME} =====" << "\n";
            for (auto op_pair : op_map) {
                auto ops = op_pair.second;
                double avgt = 0, maxt = 0, mint = -1;
                for (auto op : op_pair.second) {
                    double tmp = op.tsp_ed - op.tsp_bg;
                    avgt += tmp;
                    if (mint == -1) maxt = mint = tmp;
                    else {
                        maxt = std::max(maxt, tmp);
                        mint = std::min(mint, tmp);
                    }
                }
                // std::cout << op_pair.first << " : num = " << ops.size() << ", avg_time = " << std::fixed << std::setprecision(7) << avgt / ops.size() << " ,[" << std::fixed << std::setprecision(7) << mint << " , " << std::fixed << std::setprecision(7) << maxt << "]" << "\n";
                avgt *= 1e3;
                mint *= 1e3;
                maxt *= 1e3;
                std::cout << op_pair.first << " : num = " << ops.size() << ", avg_time = " << std::fixed << std::setprecision(4) << avgt / ops.size();
                if(OP_ANALIZE) std::cout << "  [" << std::fixed << std::setprecision(4) << mint << " , " << std::fixed << std::setprecision(4) << maxt << "]";
                std::cout << "\n";
            }
            std::cout << "\n";
        }
            
        if(!outstatqs.empty()) {
            std::cout << "===== {STAT} =====" << "\n";
            for(auto &opstatq: outstatqs) {
                double sum = 0, mx=-1e15, mi=1e15;
                for (auto &stat: opstatq.stats) {
                    sum += stat;
                    if (stat > mx) mx = stat;
                    if (stat < mi) mi = stat;
                }
                std::cout << opstatq.name << " : num = " << opstatq.count << " , avg = " << std::fixed << std::setprecision(2) << sum / opstatq.count;
                if(OP_ANALIZE) std::cout << "  [ " << std::fixed << std::setprecision(2) << mi << " , "<< std::fixed << std::setprecision(2) << mx;
                std::cout << " ]\n";
            }
            std::cout << "\n";
        }

    }

    return 0;
}
