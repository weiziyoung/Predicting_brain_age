# -*- coding: utf-8 -*-
# @Time    : 2019/7/27 3:48 PM
# @Author  : weiziyang
# @FileName: 3D_CNN.py
# @Software: PyCharm

import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import nibabel as nib
from torch import optim
from datetime import datetime
from Resnet3D import resnet10,resnet18, resnet34, resnet50, resnet101
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_num', help="Input the amount of GPU you need", default=0, type=int)
    parser.add_argument('--GPU_no', help="Input the No of GPU you want", default='', type=str)
    parser.add_argument('--batch_size', help="Input the batch number", default=2, type=int)
    parser.add_argument('--epoch', help="Input the number of epoch number", default=40, type=int)
    parser.add_argument('--pretrain', help="Input if you need a pre trained model", default=False, type=bool)
    parser.add_argument('--model_depth', help="Input your resnet depth", default=10, type=int)
    parser.add_argument('--lr', help='Input learning rate', default=0.0001, type=float)
    parser.add_argument('--norm', help='Input your normalisation method', default='GMM', type=str)
    parser.add_argument('--flag', default='', type=str)
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.y = np.load(os.path.join('training_set', dir_name, 'y.npy'))

    def __getitem__(self, index):
        self.agent = nib.load(os.path.join('training_set', self.dir_name, 'X', str(index)+'.nii.gz'))
        data = np.array(self.agent.dataobj)
        data = data[np.newaxis, :, :, :]
        return torch.from_numpy(data), torch.tensor(self.y[index])

    def __len__(self):
        return self.y.shape[0]


if __name__ == "__main__":
    args = init()
    para_name = '-'.join([key + ':' + str(value) for key, value in args.__dict__.items()])
    if not os.path.exists('logging'):
        os.mkdir('logging')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='logging/'+para_name+'.log')
    logger = logging.getLogger(__name__)
    logging.info('Start training, parameter:'+para_name)

    assert args.GPU_num <= torch.cuda.device_count(), 'GPU exceed the maximum num'
    if torch.cuda.is_available():
        if args.GPU_no:
            device = torch.device("cuda:"+args.GPU_no[0])
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    model = eval('resnet'+str(args.model_depth)+'()').to(device)

    if args.pretrain:
        device = torch.device('cpu')
        old_model = torch.load('pretrain/resnet34.pkl', map_location=None if torch.cuda.is_available() else 'cpu')
        old_model_dict = {key.replace('module.', ''): value for key, value in old_model.state_dict().items()}
        model_dict = {key: value for key, value in old_model_dict.items() if key in model.state_dict().keys()}
        model.load_state_dict(model_dict)
        model.to(device)

    if device.type == 'cuda' and args.GPU_num > 1:
        if args.GPU_no:
            assert len(args.GPU_no) == args.GPU_num
            model = nn.DataParallel(model, [int(each) for each in args.GPU_no])
        else:
            model = nn.DataParallel(model, list(range(args.GPU_num)))

    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    dataset = MyDataset(args.norm)
    train_num = int(len(dataset)*0.9)
    test_num = len(dataset) - train_num
    training_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    loss_function = nn.MSELoss()
    train_loss_list = []
    test_loss_list = []
    for epoch in range(args.epoch):
        logging.info('Epoch '+str(epoch) + 'start training')
        running_loss = 0.0
        real_loss = 0.0
        model.train()
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimiser.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            real_loss += torch.sum(torch.abs(labels - outputs))
        logging.info("Epoch %d, training loss %4.2f" % (epoch, real_loss/train_num))
        train_loss_list.append(real_loss/train_num)
        model.eval()
        with torch.no_grad():
            test_loss = 0
            predict_list = []
            true_list = []
            for data in tqdm(test_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.unsqueeze(1).to(device)
                outputs = model(inputs)
                predict_list.extend(list(outputs.detach().cpu().squeeze().numpy()))
                true_list.extend(list(labels.detach().cpu().squeeze().numpy()))

                test_loss += torch.sum(torch.abs(labels - outputs))
            logging.info('Epoch %d, true:' % epoch + str(true_list) + 'predict:' + str(predict_list))
        logging.info('Epoch %d Test loss %4.2f' % (epoch, test_loss/test_num))
        test_loss_list.append(test_loss/test_num)
    print('**** Finished Training ****')

    now = str(datetime.today())
    plt.figure(figsize=(20, 10))
    plt.plot(train_loss_list, 'r--', label='train')
    plt.plot(test_loss_list, 'b--', label='valid')
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss of  Resnet{} model with learning rate {} (pretrained:{}, normalisation:{}):'.format(args.model_depth,
                                                                                                        args.lr,
                                                                                                        args.pretrain,
                                                                                                        args.norm))
    plt.savefig("output/loss_history" + para_name + ".png")

    model_name = 'output/model' + para_name + '.pkl'
    torch.save(model, model_name)


