import os
from os.path import join as osj
import subprocess
from statistics import mean
import datetime

PROJECT_PATH = '/home/chijj/tcpdgl'
PROGRAM_PATH = osj(PROJECT_PATH, 'examples', 'pytorch', 'tcpdgl')
RESULT_PATH = osj(PROJECT_PATH, 'result')
SUMMARY_PATH = osj(RESULT_PATH, 'summary')
LOG_PATH=osj(PROJECT_PATH, 'log')

# 'cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1'
DATASET = ['cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1']
K = [32]
NAME = "tc1_k32_h2"

# False, True
PIN = [False]
LOAD_BALANCE = [False, True]

EPOCH = 50
# 'bm_prof.py', 'cp_prof.py'
PROGRAM = ['cp.py']

COUNT = 5
NUM_FEAT = 10
mp={'load_graph': 4, 'to_dvs': 5,
    'sample.capi': 0, 'model_compution': 3,
    'epoch':1, 'end2end': 2}

TIMEOUT = 300

# Build log/
if not os.path.isdir(LOG_PATH):
    os.makedirs(LOG_PATH)
# for file in os.listdir(LOG_PATH):
#     os.remove(os.path.join(LOG_PATH, file))

# Build summary result/summary/
if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)
# for file in os.listdir(SUMMARY_PATH):
#     open(os.path.join(SUMMARY_PATH, file), 'w').close()

profile_cmd = f"{os.path.join(PROJECT_PATH, 'tools', 'profiler')} s"

for cnt in range(COUNT):
    for pin in PIN:
        for prog in PROGRAM:
            for d in DATASET:
                for bal in LOAD_BALANCE:
                    name = f"{prog.split('.')[0]}_{d}_{'pin' if pin else 'gpu'}{'_bal' if bal else ''}"
                    out_path = osj(RESULT_PATH, name)
                    log_path = osj(LOG_PATH, f"{name}_{cnt}")

                    cp_cmd = f"python {os.path.join(PROGRAM_PATH, prog)} --graph_name {d} --n_feat {NUM_FEAT} --n_epoch {EPOCH} --sampler sage --log_path {log_path}{' --pin' if pin else ''}{' --load_balance' if bal else ''} "
                    echo_cmd = f'echo "========{cnt}========" >> {out_path}'
                    os.system(echo_cmd)
                    print(f'{cnt} : {cp_cmd}')

                    p1 = subprocess.Popen(cp_cmd.split())
                    try:
                        print('Running in process', p1.pid)
                        p1.wait(TIMEOUT)
                    except subprocess.TimeoutExpired:
                        print('Timed out - killing', p1.pid)
                        p1.kill()
                    print("Done")

                    profile_p = subprocess.Popen(profile_cmd.split(), stdin=open(log_path, mode='r+'), stdout=open(out_path, mode='a+'))

current_time = datetime.datetime.now()

for prog in PROGRAM:
    for dataset in DATASET:
        for pin in PIN:
            for bal in LOAD_BALANCE:
                name = f"{prog.split('.')[0]}_{dataset}_{'pin' if pin else 'gpu'}{'_bal' if bal else ''}"
                out_path = osj(RESULT_PATH, name) 

                d = {}
                f = open(out_path, 'r')
                for line in f.readlines():
                    line = line.split(' ')
                    if (not 'avg_time' in line) and (not 'avg' in line):
                        continue
                    stat_name = line[0]
                    if d.get(stat_name) == None:
                        d[stat_name] = []
                    for i in range(1, len(line)):
                        if line[i] == 'avg' or line[i] == 'avg_time':
                            stat = float(line[i+2].strip(','))
                            d[stat_name].append(stat)
                            break
                f.close()
                f = open(osj(SUMMARY_PATH, f'{current_time}.txt'), 'a+')
                f.write(f"###  {name} ###\n")
                d_result=sorted(d.items(), key=lambda x: mp.get(x[0], 100), reverse=False)

                for (key,q) in d_result:
                    if mp.get(key, 100) == 100:
                        f.write('  ')
                    mn = round(mean(q), 3)
                    f.write(key+' '+str(mn)+'\n')
                f.write('\n\n')
                f.close()
            