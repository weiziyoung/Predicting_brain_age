# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 11:25 PM
# @Author  : weiziyang
# @FileName: test.py
# @Software: PyCharm
import os
import time

if __name__ == "__main__":
    def compress(path):
        import gzip
        with open(path, 'rb')as f:
            content = f.read()
        output_path = os.path.join('compress', os.path.join(*path.split('/')[-2:]) + '.gz')
        if not os.path.exists(os.path.join(*output_path.split('/')[:-1])):
            os.mkdir(os.path.join(*output_path.split('/')[:-1]))
        with gzip.open(output_path, 'wb') as f:
            f.write(content)
    compress('ADNI.nii')