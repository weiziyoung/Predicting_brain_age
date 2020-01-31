# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 8:25 PM
# @Author  : weiziyang
# @FileName: correction.py
# @Software: PyCharm
import SimpleITK as sitk
import os
import sys

if len(sys.argv) == 1:
        categories = [each for each in os.listdir('compress') if '.' not in each]
else:
        categories = sys.argv[1:]
for category in categories:
        print('start', category)
        category_path = os.path.join('compress', category)
        print(category_path)
        file_names = [each for each in os.listdir(category_path) if each.endswith('.gz')]
        output_category_path = os.path.join('n4_bias',category)
        if not os.path.exists(output_category_path):
                os.mkdir(output_category_path)
        for n, file_name in enumerate(file_names):
                percent = n / len(file_names)
                print('percent:', percent)
                output_path = os.path.join('n4_bias', category, file_name)
                if not os.path.exists(output_path):
                    try:
                        full_path = os.path.join('compress', category, file_name)
                        inputImage = sitk.ReadImage(full_path)
                        maskImage = sitk.OtsuThreshold( inputImage, 0, 1, 200 )
                        inputImage = sitk.Cast( inputImage, sitk.sitkFloat32 )
                        corrector = sitk.N4BiasFieldCorrectionImageFilter()
                        output = corrector.Execute( inputImage, maskImage )
                        sitk.WriteImage(output, output_path)
                    except Exception as e:
                        print(output_path, 'error')