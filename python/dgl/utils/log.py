"""Log config"""
import logging
import time
import sys

prog_name = 'unknown'
graph_name = 'unknown'
cnt = ''
prog_name = sys.argv[0].split('/')[-1].split('.')[0]
load_balance = False
log_path = ''
batch_size = ''

for id, argc in enumerate(sys.argv):
    if argc == '--graph_name':
        graph_name = sys.argv[id+1]
    elif argc == '--cnt':
        cnt = sys.argv[id+1]
    elif argc == '--load_balance':
        load_balance = True
    elif argc == '--log_path':
        log_path = sys.argv[id+1]
    elif argc == '--batch_size':
        batch_size = sys.argv[id+1]
    

log_tt = int(time.time())

pflogger = logging.getLogger('PFlogger')
pflogger.propagate = False
pflogger.setLevel(logging.INFO)

formatter = logging.Formatter('[PF] %(message)s')

if log_path == '':
    log_path = f'/home/chijj/tcpdgl/log/{prog_name}_{graph_name}{"_"+cnt if cnt else ""}{"_bal" if load_balance else ""}{"_"+batch_size if batch_size else ""}'

fh = logging.FileHandler(log_path, mode='w+', encoding='utf-8')
fh.setFormatter(formatter)
pflogger.addHandler(fh)

# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# pflogger.addHandler(ch)
