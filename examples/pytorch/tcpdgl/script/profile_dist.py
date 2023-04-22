import os
import subprocess
from statistics import mean

PROJECT_PATH = '/home/chijj/tcpdgl'
PROGRAM_PATH = os.path.join(PROJECT_PATH, 'examples', 'pytorch', 'tcpdgl')
RESULT_PATH = os.path.join(PROJECT_PATH, 'result')
SUMMARY_PATH = os.path.join(RESULT_PATH, 'summary')
LOG_PATH=os.path.join(PROJECT_PATH, 'log')

# 'cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1'
DATASET = ['cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1']
K = [32]
NAME = "tc1_k32_h2"
# False, True
PIN = [False]
EPOCH = 10
PROGRAM = ['bm.py', 'cp.py', 'bm_dist_disk.py', 'cp_dist_disk.py']
COUNT = 3
NUM_FEAT = 10
# 'labor', 'sage'
SAMPLER = ['labor', 'sage']
DIST = [4]
TIMEOUT = 600
mp={'load_graph': 0, 'to_dvs': 1,
    'sample.capi': 2, 'dataloader.__next__': 3, 
    'epoch': 4, 'end2end': 5}


# Build log/
if not os.path.isdir(LOG_PATH):
    os.makedirs(LOG_PATH)
for file in os.listdir(LOG_PATH):
    os.remove(os.path.join(LOG_PATH, file))

# Build result/
for pin in PIN:
    for prog in PROGRAM:
        for s in SAMPLER:
            for d in DATASET:
                out_dir_path = os.path.join(RESULT_PATH, prog.split('.')[0], str(d))
                out_path = f"{out_dir_path}/{'pin' if pin else 'gpu'}_feat{NUM_FEAT}_{s}"
                if not os.path.exists(out_dir_path):
                    os.makedirs(out_dir_path)
                f=open(out_path, mode='w')
                f.close()

# Build summary result/summary/
if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)
for file in os.listdir(SUMMARY_PATH):
    open(os.path.join(SUMMARY_PATH, file), 'w').close()

# execute programs
profile_cmd = f"{os.path.join(PROJECT_PATH, 'tools', 'profiler')} s"
for cnt in range(COUNT):
    for pin in PIN:
        for prog in PROGRAM:
            for s in SAMPLER:
                for d in DATASET:
                    out_dir_path = os.path.join(RESULT_PATH, prog.split('.')[0], str(d))
                    out_path = f"{out_dir_path}/{'pin' if pin else 'gpu'}_feat{NUM_FEAT}_{s}"
                    log_path = os.path.join(LOG_PATH, f"{prog.split('.')[0]}_{d}_{cnt}")
                    cp_cmd = f"python {os.path.join(PROGRAM_PATH, prog)} --graph_name {d} {'--pin ' if pin else ''}--n_feat {NUM_FEAT} --n_epoch {EPOCH} --sampler {s} --cnt {cnt}{' --gpu 0,1,2,3' if 'dist' in prog else ''}"
                    print(f'{cnt} : {cp_cmd}')
                    echo_cmd = f'echo "========{cnt}========" >> {out_path}'
                    os.system(echo_cmd)

                    p1 = subprocess.Popen(cp_cmd.split())
                    try:
                        print('Running in process', p1.pid)
                        p1.wait(TIMEOUT)
                    except subprocess.TimeoutExpired:
                        print('Timed out - killing', p1.pid)
                        p1.kill()
                    print("Done")

                    profile_p = subprocess.Popen(profile_cmd.split(), stdin=open(log_path, mode='r+'), stdout=open(out_path, mode='a+'))
                                        
# Write summary
for prog in PROGRAM:
    for s in SAMPLER:
        for dataset in DATASET:
            for pin in PIN:
                d = {}
                file_path = os.path.join(RESULT_PATH, prog.split('.')[0], dataset, ('pin' if pin else 'gpu') + f'_feat{NUM_FEAT}_{s}')
                f = open(file_path, 'r')
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
                f.write(f"###  {dataset}_feat{NUM_FEAT}_{s} <{'pin' if pin else 'gpu'}>  ###\n")
                d_result=sorted(d.items(), key=lambda x: mp.get(x[0], 100), reverse=False)

                for (key,q) in d_result:
                    if mp.get(key, 100) == 100:
                        f.write('  ')
                    mn = round(mean(q), 3)
                    f.write(key+' '+str(mn)+'\n')
                f.write('\n\n')
                f.close()
