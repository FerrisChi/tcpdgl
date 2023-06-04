import subprocess
import os

TIMEOUT = 9999999

if __name__ == '__main__':
    pf_balance_cmd = f'python /home/chijj/tcpdgl/examples/pytorch/tcpdgl/script/pf_balance.py'
    pf_dw_cmd = f'python /home/chijj/tcpdgl/examples/pytorch/tcpdgl/script/pf_dw.py'
    pf_cmd = [pf_balance_cmd, pf_dw_cmd]

    for cmd in pf_cmd:
        p = subprocess.Popen(cmd.split())
        try:
            print('Running in process', p.pid)
            p.wait(TIMEOUT)
        except subprocess.TimeoutExpired:
            print('Timed out - killing', p.pid)
            p.kill()
    print("Done")