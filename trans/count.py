import os

import torch
import numpy as np

import matplotlib.pyplot as plt


def param_count(pth_path):
    """for the global quantization"""
    pth = torch.load(pth_path)
    param_dict = pth['param_dict']
    maxi = pth['max']
    mini = pth['min']
    upper_lim = pth['upper_lim']
    print(maxi, mini, upper_lim)

    param_table = {'num_params': 0, 'table': [0] * (upper_lim + 1)}

    for name, param in param_dict.items():
        if not isinstance(param, torch.Tensor):
            count(param, param_table)
            print('done', name)

    distribution(param_table, maxi, mini, upper_lim)
    return param_table


def count(param, param_table):
    for elem in np.nditer(param):
        elem = elem[()]
        param_table['table'][elem] += 1
        param_table['num_params'] += 1


def distribution(param_table, maxi, mini, upper_lim):
    x = np.arange(upper_lim + 1)
    y = param_table['table']

    plt.bar(x, y, width=0.9)
    plt.show()


def compute_mean_std(pth_path):
    param = torch.load(pth_path)
    means = []
    stds = []
    num_layer = 0
    for name, param in param.items():
        if not param.shape == torch.Size([]):
            mean = param.mean().item()
            std = param.std().item()
            print(f"{name} | mean: {mean} | std: {std}")
            means.append(mean)
            stds.append(std)
            num_layer += 1

    # 绘图std
    x = np.arange(num_layer)
    plt.bar(x, stds)
    plt.show()


if __name__ == '__main__':
    table = param_count('../weights/uint8_VGG16_cifar10_8904.pth')
    # pth = '../weights/VGG16_cifar10_8931.pth'




