import os
import subprocess
from statistics import mean

PROJECT_PATH = '/home/chijj/tcpdgl'
PROGRAM_PATH = os.path.join(PROJECT_PATH, 'examples', 'pytorch', 'tcpdgl')
RESULT_PATH = os.path.join(PROJECT_PATH, 'result')
SUMMARY_PATH = os.path.join(RESULT_PATH, 'summary')
LOG_PATH=os.path.join(PROJECT_PATH, 'log')

# 'cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1', 'ogbl_ppa', 'ogbl_ddi'
DATASET = ['cora', 'Reddit', 'ogbl_ppa', 'ogbl_ddi']
K = [32]
NAME = "tc1_k32_h2"

# False, True
LOAD_BALANCE = [False]

# 'bm.py', 'cp.py', 'bm_dist_disk.py', 'cp_dist_disk.py', 'dw_bm.py', 'dw_cp.py'
PROGRAM = ['dw_cp.py']
BATCH_SIZE = [2048, 4096, 5120, 6400]
COUNT = 1
NUM_FEAT = 10
TIMEOUT = 15

mp={'load_graph': 0, 'to_dvs': 1,
    'sample.capi': 2, 'dataloader.__next__': 3, 
    'epoch': 4, 'end2end': 5}


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

# execute deepwalk programs
for cnt in range(COUNT):
    for bal in LOAD_BALANCE:
        for prog in PROGRAM:
            for d in DATASET:
                for size in BATCH_SIZE:
                    name = f"{prog.split('.')[0]}_{d}_{size}{'_bal' if bal else ''}"
                    out_path = os.path.join(RESULT_PATH, name)
                    log_path = os.path.join(LOG_PATH, f"name_{cnt}")

                    cp_cmd = f"python {os.path.join(PROGRAM_PATH, prog)} --graph_name {d} --cnt {cnt} --batch_size {size} --log_path {log_path}{' --load_from_ogb' if d == 'ogbl-ppa' or d == 'ogbl-ddi' else ''}{' --load_balance' if prog == 'dw_cp.py' else ''}"
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

# Write deepwalk summary
for prog in PROGRAM:
    for dataset in DATASET:
        for size in BATCH_SIZE:
            for bal in LOAD_BALANCE:
                d = {}
                name = f"{prog.split('.')[0]}_{dataset}_{size}{'_bal' if bal else ''}"
                out_path = os.path.join(RESULT_PATH, name)
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
                f = open(os.path.join(SUMMARY_PATH, f'{prog.split(".")[0]}.txt'), 'a+')
                f.write(f"###  {dataset}_{size}{'_bal' if bal else ''}  ###\n")
                d_result=sorted(d.items(), key=lambda x: mp.get(x[0], 100), reverse=False)

                for (key,q) in d_result:
                    if mp.get(key, 100) == 100:
                        f.write('  ')
                    mn = round(mean(q), 3)
                    f.write(key+' '+str(mn)+'\n')
                f.write('\n\n')
                f.close()