import os
import subprocess
from statistics import mean

PATH = '/home/chijj/cpdgl_1.0/examples/pytorch/graphsage/'
SUMMARY_PATH = '/home/chijj/cpdgl_1.0/result/summary'
RESULT_PATH='/home/chijj/cpdgl_1.0/result'

# 'cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1'
DATASET = ['cora', 'ogbn_products', 'Reddit', 'orkut', 'patents', 'lj1']
K = [32]
NAME = "tc1_k32_h2"
# False, True
PIN = [False]
EPOCH = 50
# 'bm_prof.py', 'cp_prof.py'
PROGRAM = ['bm_prof.py', 'cp_prof.py']
COUNT = 5
NUM_FEAT = 10
# 'labor', 'sage'
SAMPLER = ['labor', 'sage']
mp={'load_graph': 0, 'to_dvs': 1,
                    'sample.capi': 2, 'dataloader.__next__': 3, 
                    'model_compution': 4, 'epoch': 5}

TIMEOUT = 120

for pin in PIN:
    for prog in PROGRAM:
        for s in SAMPLER:
            for d in DATASET:
                out_dir_path = f"/home/chijj/cpdgl_1.0/result/{prog.split('.')[0]}/{d}"
                out_path = f"{out_dir_path}/{'pin' if pin else 'gpu'}_feat{NUM_FEAT}_{s}"
                if not os.path.exists(out_dir_path):
                    os.makedirs(out_dir_path)
                f=open(out_path, mode='w')
                f.close()

for i in range(COUNT):
    for pin in PIN:
        for prog in PROGRAM:
            for s in SAMPLER:
                for d in DATASET:
                    out_dir_path = f"/home/chijj/cpdgl_1.0/result/{prog.split('.')[0]}/{d}"
                    out_path = f"{out_dir_path}/{'pin' if pin else 'gpu'}_feat{NUM_FEAT}_{s}"
                    cp_cmd = f"python {PATH+prog} --graph_name {d} {'--pin ' if pin else ''}--n_feat {NUM_FEAT} --n_epoch {EPOCH} --sampler {s}"
                    profile_cmd = f"/home/chijj/cpdgl/tools/profiler s"
                    echo_cmd = f'echo "========{i}========" >> {out_path}'
                    os.system(echo_cmd)
                    print(f'{i} : {cp_cmd}')
                    # continue
                    # os.system(cp_cmd)
                    p1 = subprocess.Popen(cp_cmd.split(), stdout=subprocess.PIPE)
                    p2 = subprocess.Popen(profile_cmd.split(), stdin=p1.stdout, stdout=open(out_path, mode='a+'))
                    try:
                        print('Running in process', p1.pid, p2.pid)
                        p1.wait(TIMEOUT)
                    except subprocess.TimeoutExpired:
                        print('Timed out - killing', p1.pid)
                        p1.kill()
                        p2.kill()
                    print("Done")


if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)
for file in os.listdir(SUMMARY_PATH):
    open(os.path.join(SUMMARY_PATH, file), 'w').close()

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
        