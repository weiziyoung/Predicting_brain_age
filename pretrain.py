# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 1:27 PM
# @Author  : weiziyang
# @FileName: pretrain.py
# @Software: PyCharm
import torch
from Resnet3D import resnet10



pretrained_model_obj = torch.load('MedicalNet/pretrain/resnet_10.pth', map_location='cpu')
new_dict = {}
for key,value in pretrained_model_obj['state_dict'].items():
    new_key = key.replace('module.', '')
    new_dict[new_key] = value

new_model = resnet10()
new_model_dict = new_model.state_dict()
pretrained_dict = {key: value for key, value in new_dict.items() if key in new_model_dict.keys()}
new_model_dict.update(pretrained_dict)
new_model.load_state_dict(new_model_dict)
pass