# -*- coding: utf-8 -*-
# @Time    : 2019/8/13 3:11 PM
# @Author  : weiziyang
# @FileName: auto_run.py
# @Software: PyCharm
import re
import os
import sys
import time
from datetime import datetime


if __name__ == "__main__":
    depth = sys.argv[1]
    epoch = sys.argv[2]
    want_GPU = sys.argv[3]
    while True:
        time.sleep(5)
        print(datetime.today(), 'try...')
        GPU_state_string = os.popen('nvidia-smi').read()
        GPU_stats = [int(each) for each in re.findall(r'(\d+)MiB / 11178', GPU_state_string)]
        available_GPU = []
        for n, GPU_memory in enumerate(GPU_stats):
            if GPU_memory < 6000:
                available_GPU.append(n)
        if len(available_GPU) >= int(want_GPU):
            print(datetime.today(), 'GPU number is enough, num:', len(available_GPU))
            break
    GPU_num = len(available_GPU)
    GPU_no = ''.join([str(each) for each in available_GPU])
    batch_num = GPU_num * 2
    print('start!')
    cmd = f'python3 train.py --GPU_num {GPU_num} --GPU_no {GPU_no} --batch_size {batch_num}'
    os.system(cmd)
    print('end!')
