import os
import collections

import numpy as np
import torch

from models import VGG16
from data import data
from train import evaluate


class GlobalDeQuantizationTool:
    def __init__(self, pth_path):
        pth = torch.load(pth_path)
        self.maxi = pth['max']
        self.mini = pth['min']
        self.param = pth['param_dict']
        self.lower_lim = pth['lower_lim']
        self.upper_lim = pth['upper_lim']
        self.num_layers = pth['layers']

    def de_quantize(self):
        de_qt_param = collections.OrderedDict()
        for name, param in self.param.items():
            # if param.shape == torch.Size([]):
            if isinstance(param, torch.Tensor):
                # Batch Norm
                de_qt_param[name] = param
            else:
                de_qt_param[name] = self.trans_to_float(param)

        return de_qt_param

    def trans_to_float(self, param):
        param = param.astype(np.float32)
        param /= (self.upper_lim + 1)
        param *= (self.maxi - self.mini)
        param += self.mini

        return torch.from_numpy(param)


class LayerDeQuantizationTool:
    def __init__(self, pth_path):
        self.pth = torch.load(pth_path)
        self.lower_lim = self.pth.pop('lower_lim')
        self.upper_lim = self.pth.pop('upper_lim')
        self.num_layers = self.pth.pop('layers')

    def de_quantize(self):
        de_qt_param = collections.OrderedDict()
        for name, param in self.pth.items():
            # if param.shape == torch.Size([]):
            if isinstance(param, torch.Tensor):
                # Batch Norm
                de_qt_param[name] = param
            else:
                de_qt_param[name] = self.trans_to_float(param)

        return de_qt_param

    def trans_to_float(self, param):
        """{max, min, param}"""

        maxi = param["max"]
        mini = param["min"]
        param = param["param"]

        param = param.astype(np.float32)
        param /= (self.upper_lim + 1)
        param *= (maxi - mini)
        param += mini

        return torch.from_numpy(param)


if __name__ == '__main__':
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)


