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

    qt_path = "../weights/uint8_VGG16_cifar10_8931.pth"
    pth_path = "../weights/VGG16_cifar10_8931.pth"
    model = VGG16.form_model()
    _, test_dataset = data.form_datasets()
    test_loader = data.form_dataloader(
        test_dataset,
        batch_size=64,
        test=True
    )

    # print(evaluate(model, test_loader, 'cuda', pth_path))
    de_qt = LayerDeQuantizationTool(qt_path)

    model.load_state_dict(de_qt.de_quantize())
    print(evaluate(model, test_loader, 'cuda'))
    # print(de_qt.de_quantize())
