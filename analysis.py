# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 3:03 PM
# @Author  : weiziyang
# @FileName: analysis.py
# @Software: PyCharm
from tabulate import tabulate
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re


def plot(*file_list):
    for file_name in file_list:
        f = open(file_name)
        text = f.read()
        f.close()
        training_losses = [float(each) for each in re.findall(r'training loss (\d+\.\d+)',text)]
        test_losseses = [float(each) for each in re.findall(r'Test loss (\d+\.\d+)',text)]
        plt.plot(training_losses, label=file_name+' training_loss')
        plt.plot(test_losseses, label=file_name+' test_loss')
        plt.legend()
    plt.show()


def true_and_predict(string):
    result = re.search(r'true:(.+)predict:(.+)',string)
    true, predict = result.group(1), result.group(2)
    test_array = np.array(eval(true))
    predict_array = np.array(eval(predict))
    plt.figure(figsize=(10, 10))
    p_value = pearsonr(test_array, predict_array)
    rmse = np.sqrt(mean_squared_error(test_array, predict_array))
    r2 = r2_score(test_array, predict_array)
    plt.scatter(test_array, predict_array, c='k')
    plt.xlabel('Ground truth Age(years)')
    plt.ylabel('Predicted Age(years)')
    plt.title('Ground-truth Age versus Predict Age using \n \
                Resnet34 with Image normalised by GMM method')
    plt.plot(np.linspace(15, 80, 100), np.linspace(15, 80, 100), c='r', label='Expected prediction line')
    offset = 0
    plt.text(10, 80 + offset, f'Mean Absolute Error={round(np.sum(np.abs(predict_array - test_array))/len(predict_array),3)}', fontsize=14)
    plt.text(10, 75 + offset, f'Pearson correlation coefficient:{round(p_value[0],3)}', fontsize=14)
    plt.text(10, 70 + offset, f'R Squared:{round(r2,3)}', fontsize=14)
    plt.text(10, 65 + offset, f'RMSE:{round(rmse,3)}', fontsize=14)
    plt.legend()
    plt.show()


def get_training_time(*file_list):
    for file in file_list:
        f = open(file)
        text = f.read()
        f.close()
        lines = text.split('\n')
        first, last = lines[0], lines[-1]
        for each in lines[::-1]:
            if each.strip():
                last = each.strip()
                break
        start_time = datetime.strptime(first.split(',')[0], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(last.split(',')[0], '%Y-%m-%d %H:%M:%S')
        hours = (end_time - start_time).seconds//(60*60)
        print(file, hours, 'h')


if __name__ == "__main__":
    plot('logging/resnet10-40-lr0.01.log','logging/resnet10-40-lr0.001.log')