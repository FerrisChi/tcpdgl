"""Log config"""
import logging
import time
import sys

prog_name = 'unknown'
graph_name = 'unknown'
cnt = '-1'
prog_name = sys.argv[0].split('/')[-1].split('.')[0]

for id, argc in enumerate(sys.argv):
    if argc == '--graph_name':
        graph_name = sys.argv[id+1]
    if argc == '--cnt':
        cnt = sys.argv[id+1]
    

log_tt = int(time.time())

pflogger = logging.getLogger('PFlogger')
pflogger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[PF] %(message)s')

filename = f'/home/chijj/tcpdgl/log/{prog_name}_{graph_name}_{cnt}'
fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
ch = logging.StreamHandler()
fh.setFormatter(formatter)
ch.setFormatter(formatter)
pflogger.addHandler(fh)
# pflogger.addHandler(ch)
