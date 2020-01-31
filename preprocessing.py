# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 12:42 PM
# @Author  : weiziyang
# @FileName: preprocessing.py
# @Software: PyCharm
import os
import copy

import deepbrain
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import nibabel.processing
from skimage import morphology
from intensity_normalization.normalize import gmm


class Processor(object):
    def __init__(self, file_name, skull_strip=False):
        self.template_path = 'MNI152_T1_1mm_brain.nii.gz'
        self.environment_path = 'enviorment.txt'
        self.file_name = file_name
        self._skull_strip = skull_strip
        self.obj_name = self.file_name.split('/')[-1]
        self.agent = nib.load(self.file_name)
        self.n4_bias_path = '-'.join(['n4_bias', self.obj_name])
        self.registration_path = '-'.join(['registration', self.obj_name])
        self.matrix_path = '-'.join(['registration', self.obj_name, '.mat'])
        self.segmentation_path = '-'.join(['segmentation', self.obj_name])
        self.normalisation_path = '-'.join(['normalisation', self.obj_name])

    def init_env(self):
        f = open(self.environment_path, 'r')
        config_file = f.read()
        f.close()

        config_dict = dict()
        for line in config_file.split('\n'):
            left, right = line.split('=')
            config_dict[left] = right
        os.environ.update(config_dict)

    def resample(self, resolution=(1, 1, 1)):
        self.agent = nib.processing.resample_to_output(self.agent, resolution)

    def n4_bias_correction(self):
        inputImage = sitk.ReadImage(self.file_name)
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output = corrector.Execute(inputImage, maskImage)
        sitk.WriteImage(output, self.n4_bias_path)

    def skull_strip(self):
        if self._skull_strip:
            copy_img = copy.copy(np.array(self.agent.dataobj))
            extractor = deepbrain.Extractor()
            prob = extractor.run(copy_img)
            copy_img[prob < 0.5] = 0
            binary = copy.copy(copy_img)
            binary[binary > 0] = 1
            labels = morphology.label(binary)
            labels_num = [len(labels[labels == each]) for each in np.unique(labels)]
            rank = np.argsort(np.argsort(labels_num))
            index = list(rank).index(len(rank) - 2)
            new_img = copy.copy(copy_img)
            new_img[labels != index] = 0
            self.agent.dataobj = new_img

    def template_registration(self):
        os.system(f"flirt -ref {self.template_path} -in {self.n4_bias_path}"
                  f"  -out {self.registration_path} -omat {self.matrix_path}")

    def gmm_normalisation(self):
        agent = nib.load(self.registration_path)
        new_agent = gmm.gmm_normalize(agent)
        nib.save(new_agent, self.normalisation_path)

    def auto_segmentation(self):
        os.system(f"/usr/local/fsl/bin/fast {self.registration_path}")

    def clean(self):
        pass

    def start(self):
        self.init_env()
        self.resample()
        self.skull_strip()
        self.n4_bias_correction()
        self.template_registration()
        self.gmm_normalisation()
        self.auto_segmentation()
        self.clean()


if __name__ == "__main__":
        Processor('.')
