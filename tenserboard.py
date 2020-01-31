# -*- coding: utf-8 -*-
# @Time    : 2019/8/28 2:15 AM
# @Author  : weiziyang
# @FileName: tenserboard.py
# @Software: PyCharm

import Resnet3D
import torch
import numpy as np
import nibabel as nib
from tensorboardX import SummaryWriter


net = Resnet3D.resnet34()
data = np.array(nib.load('training_set/GMM/X/1.nii.gz').dataobj)
writer = SummaryWriter(log_dir='logs')
data = torch.Tensor(data[np.newaxis,np.newaxis,:,:,:])

writer.add_graph(net, input_to_model=(data,))
writer.close()
