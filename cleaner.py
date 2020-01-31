# -*- coding: utf-8 -*-
# @Time    : 2019/6/24 3:11 PM
# @Author  : weiziyang
# @FileName: cleaner.py
# @Software: PyCharm
import os
import gzip

import numpy as np
import pandas
from tqdm import tqdm

pandas.set_option('display.max_columns', None)


def unzip(file_name, to_path):
    g_file = gzip.GzipFile(file_name)
    open(to_path, "wb").write(g_file.read())
    g_file.close()


class Cleaner(object):
    def __init__(self, name):
        self.name = name
        self.from_dir_name = 'data/' + self.name
        self.to_dir_name = 'preprocessed/' + self.name
        if not os.path.exists(self.to_dir_name):
            os.makedirs(self.to_dir_name)

    def clean(self):
        file_names = [each for each in os.listdir(self.from_dir_name) if each.startswith('sub')]
        for name in tqdm(file_names):
            full_path = os.path.join(self.from_dir_name, name, 'anat', 'mprage_skullstripped.nii.gz')
            to_path = os.path.join(self.to_dir_name, name)
            unzip(full_path, to_path)

    def get_demo_data(self):
        demo_info_path = os.path.join(self.from_dir_name, self.name + '_demographics.txt')
        with open(demo_info_path, 'r') as f:
            name_list = []
            age_list = []
            gender_list = []
            lines = f.readlines()
            for line in lines:
                info = line.split()
                name = info[0]
                age = info[2]
                gender = info[3]
                name_list.append(name)
                age_list.append(age)
                gender_list.append(gender)
            dataframe = pandas.DataFrame({
                'name': name_list,
                'age': np.array(age_list, dtype='float64'),
                'gender': pandas.Categorical(gender_list),
                'origin': self.name
            })
            dataframe.to_csv(self.to_dir_name+'.csv')

    def start(self):
        self.clean()
        self.get_demo_data()


class IXICleaner(object):
    def __init__(self):
        self.from_dir_name = 'data/IXI/IXI-T1'
        self.to_dir_name = 'preprocessed/IXI'
        if not os.path.exists(self.to_dir_name):
            os.makedirs(self.to_dir_name)

    def clean(self):
        file_names = [each for each in os.listdir(self.from_dir_name) if not each.startswith('.')]
        for name in tqdm(file_names):
            full_path = os.path.join(self.from_dir_name, name)
            to_path = os.path.join(self.to_dir_name, name)
            unzip(full_path, to_path)

    def get_demo_data(self):
        info_path = os.path.join('data/IXI/IXI.xls')
        data = pandas.read_excel(info_path)
        # filter the individual who has no age
        data = data[data.AGE > 0]
        # filter the item that is in dataset

        def is_in_dataset(obj):
            id = obj['IXI_ID']
            file_name = 'preprocessed/IXI/IXI' + str(id).zfill(3)
            if os.path.exists(file_name):
                return True
            return False
        data = data[data.apply(is_in_dataset, axis=1)]
        # drop duplicated items
        data = data.drop_duplicates(['IXI_ID'])
        # get the column we want
        data = data.loc[:, ['IXI_ID', 'SEX_ID (1=m, 2=f)', 'AGE']]
        data.columns = ['name', 'gender', 'age']
        data = data[['name', 'age', 'gender']]
        data['origin'] = 'IXI'
        data.loc[data['gender'] == 1, 'gender'] = 'm'
        data.loc[data['gender'] == 2, 'gender'] = 'f'
        data.to_csv(self.to_dir_name + '.csv')


class MultiCleaner(Cleaner):
    def __init__(self, name):
        super().__init__(name)

    def clean(self):
        dir_names = [each for each in os.listdir(self.from_dir_name) if not each.startswith('.')
                     and not each.endswith('.txt')]
        for dir_name in tqdm(dir_names):
            path = os.path.join(self.from_dir_name, dir_name)
            for file_name in os.listdir(path):
                if not file_name.startswith('.'):
                    full_path = os.path.join(path, file_name, 'anat', 'mprage_skullstripped.nii.gz')
                    to_path = os.path.join(self.to_dir_name, file_name)
                    unzip(full_path, to_path)


if __name__ == "__main__":
    cleaner = Cleaner('Pittsburgh')
    cleaner.start()