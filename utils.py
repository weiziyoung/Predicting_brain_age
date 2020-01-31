# -*- coding: utf-8 -*-
# @Time    : 2019/6/24 5:51 PM
# @Author  : weiziyang
# @FileName: utils.py
# @Software: PyCharm
import os
import pandas
import numpy as np

def get_sample_count():
    result = list(os.walk('n4_bias/'))
    count = 0
    for i in range(1, len(result)):
        temp = result[i]
        count += len(temp[2])
    return count


def compress(path):
    import gzip
    with open(path, 'rb')as f:
        content = f.read()
    output_path = os.path.join('compress', os.path.join(*path.split('/')[-2:]) +'.gz')
    if not os.path.exists(os.path.join(*output_path.split('/')[:-1])):
        os.mkdir(os.path.join(*output_path.split('/')[:-1]))
    with gzip.open(output_path, 'wb') as f:
        f.write(content)


def concat_table(name_list, output_name):
    csv_list = []
    for name in name_list:
        full_path = os.path.join('preprocessed', name)
        temp_csv = pandas.read_csv(full_path)
        csv_list.append(temp_csv)
    csv_result = pandas.concat(csv_list)
    csv_result = csv_result.drop(columns=['Unnamed: 0'])
    csv_result.reset_index(drop=True, inplace=True)
    csv_result['origin'] = output_name
    csv_result.to_csv(os.path.join('preprocessed', output_name + '.csv'))


def append_format():
    dir_names = [each for each in os.listdir('preprocessed') if not each.startswith('.')
                 and not each.endswith('.csv') and not each.endswith('.ipynb')]
    for name in dir_names:
        temp_path = os.path.join('preprocessed', name)
        file_names = [each for each in os.listdir(temp_path) if not each.startswith('.')]
        for file_name in file_names:
            full_name = os.path.join(temp_path, file_name)
            if not full_name.endswith('nii'):
                os.rename(full_name, full_name + '.nii')


def check():
    ZSCORE = np.load('training_set/ZSCORE/y.npy')
    GMM = np.load('training_set/GMM/y.npy')
    for a, b in zip(ZSCORE, GMM):
        assert  a == b
    print('all correct')

if __name__ == "__main__":
    check()


