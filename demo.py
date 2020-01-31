# -*- coding: utf-8 -*-
# @Time    : 2019/6/24 12:36 PM
# @Author  : weiziyang
# @FileName: demo.py
# @Software: PyCharm
import os
import random

import numpy as np
import nibabel as nib
from mayavi import mlab
from nilearn.plotting import plot_roi


class Demonstrator:
    def __init__(self, name):
        self.package_name = f'registration/{name}'
        self.filenames = [each for each in os.listdir(self.package_name) if not each.startswith('.')]

    def random_select(self, keyword=None):
        file_list = []
        if keyword:
            for name in self.filenames:
                if keyword in name:
                    file_list.append(name)
        else:
            file_list = self.filenames
        rand_num = random.randint(0, len(file_list)-1)
        rand_file = file_list[rand_num]
        file_full_name = os.path.join(self.package_name, rand_file)
        data = np.array(nib.load(file_full_name).get_data())
        return data

    @staticmethod
    def scaled(data):
        print('max', np.max(data), 'min', np.min(data))
        rang = np.max(data) - np.min(data)
        after_scaled = data/rang * 255
        return after_scaled

    def show_3d_contour(self, data):
        mlab.contour3d(data)
        mlab.show()

    def show_internal(self, data):
        mlab.pipeline.volume(mlab.pipeline.scalar_field(data), vmin=0, vmax=0.8)
        mlab.show()

    def show_slice(self, data):
        mlab.volume_slice(data, plane_orientation='z_axes', slice_index=50)
        mlab.show()


if __name__ == "__main__":
    baltimore = Demonstrator('AnnArbor')
    data = baltimore.random_select()
    # data = baltimore.scaled(data)
    baltimore.show_slice(data)
