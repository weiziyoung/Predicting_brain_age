# -*- coding: utf-8 -*-
# @Time    : 2019/7/21 5:07 PM
# @Author  : weiziyang
# @FileName: template_registration.py
# @Software: PyCharm
import os
import sys
from tqdm import tqdm

f = open('enviorment.txt', 'r')
config_file = f.read()
f.close()

from_dir = 'n4_bias'
config_dict = dict()
for line in config_file.split('\n'):
    try:
        left, right = line.split('=')
        config_dict[left] = right
    except Exception:
        pass
os.environ.update(config_dict)
if not os.path.exists('registration'):
    os.mkdir('registration/')
if not os.path.exists('affine_matrix'):
    os.mkdir('affine_matrix/')
template_file = 'MNI152_T1_1mm_brain.nii.gz'

if len(sys.argv) == 1:
    categories = [each for each in os.listdir('n4_bias') if '.' not in each]
else:
    categories = sys.argv[1:]

for category in categories:
    category_path = os.path.join(from_dir, category)
    files = [each for each in os.listdir(category_path) if each.endswith('.gz')]

    if not os.path.exists(os.path.join('registration', category)):
        os.mkdir(os.path.join('registration', category))
    if not os.path.exists(os.path.join('affine_matrix', category)):
        os.mkdir(os.path.join('affine_matrix', category))
    for file in tqdm(files):
        full_path = os.path.join(category_path, file)
        output_path = os.path.join('registration', category, file.split('.')[0])
        matrix_path = os.path.join('affine_matrix', category, '.'.join([file.split('.')[0], '.mat']))
        os.system("flirt -ref {} -in {}  -out {} -omat {}".format(template_file, full_path, output_path, matrix_path))
