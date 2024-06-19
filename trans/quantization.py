import collections
import os

import numpy as np
import torch


class QuantizationTool:
    def __init__(self, param_dict: str):
        self.file_name = os.path.basename(param_dict)
        self.param_dict = torch.load(param_dict)


class GlobalQuantizationTool(QuantizationTool):
    def __init__(self, param_dict: str):
        super().__init__(param_dict)
        self.maxi, self.mini, self.param_range, self.num_layers = self.get_extrema()
        print(f'The range of parameters is {round(self.mini, 5)} ~ {round(self.maxi, 5)}, {self.num_layers} layers.')

    def get_extrema(self):
        """get the maximum and the minimum of the parameters"""
        maxi = float('-inf')
        mini = float('inf')
        layers = 0
        for name, param in self.param_dict.items():
            layers += 1
            if param.shape == torch.Size([]):
                # Batch_Norm 跳过
                continue
            param_max = param.max().item()
            param_min = param.min().item()

            # print(f'max={param_max}, min={param_min} | {name} | {param.shape}')

            if param_max > maxi:
                maxi = param_max
            if param_min < mini:
                mini = param_min
        param_range = maxi - mini
        return maxi, mini, param_range, layers

    def trans_to_int(self, weight: torch.Tensor, int_type):
        type_info = np.iinfo(int_type)
        lower_lim = type_info.min  # 0
        upper_lim = type_info.max  # 255(uint8)

        if weight.is_cuda:
            weight = weight.to('cpu')
        param_array = weight.numpy()

        # 最小值降为0
        param_array -= self.mini
        # 范围缩小至0~1
        param_array /= self.param_range
        # 映射到目标范围 此时maxi->upper+1
        param_array *= upper_lim + 1
        # 防止越界
        param_array = np.where(param_array >= upper_lim + 1, upper_lim + 0.5, param_array)

        quantized = int_type(param_array)
        return quantized

    def __call__(self, int_type=np.uint8):
        quantized_dict = collections.OrderedDict()
        layer = 0
        for name, param in self.param_dict.items():
            if param.shape == torch.Size([]):
                # Batch_Norm 不变
                quantized = param
            else:
                quantized = self.trans_to_int(param, int_type)
            quantized_dict[name] = quantized
            layer += 1
            print(f"Quantized layer {layer}/{self.num_layers}")

        type_info = np.iinfo(int_type)
        lower_lim = type_info.min
        upper_lim = type_info.max
        save_data = {
            "param_dict": quantized_dict,
            "max": self.maxi,
            "min": self.mini,
            "lower_lim": lower_lim,
            "upper_lim": upper_lim,
            "layers": self.num_layers
        }
        file_name = int_type.__name__ + '_' + self.file_name
        file_path = f"../weights/{file_name}"
        torch.save(save_data, file_path)
        print(f"Saved as {os.path.abspath(file_path)}")


class LayerQuantizationTool(QuantizationTool):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.num_layers = len(self.param_dict)

    @staticmethod
    def trans_to_int(weight: torch.Tensor, int_type):
        """
        :return:{
            max: layer_maximum,
            min: layer_minimum,
            param
        }
        """
        type_info = np.iinfo(int_type)
        lower_lim = type_info.min  # 0
        upper_lim = type_info.max  # 255(uint8)

        param_max = weight.max().item()
        param_min = weight.min().item()

        if weight.is_cuda:
            weight = weight.to('cpu')
            param_array = weight.numpy()

            # 最小值降为0
            param_array -= param_min
            # 范围缩小至0~1
            param_array /= param_max - param_min
            # 映射到目标范围 此时maxi->upper+1
            param_array *= upper_lim + 1
            # 防止越界
            param_array = np.where(param_array >= upper_lim + 1, upper_lim + 0.5, param_array)

            quantized = int_type(param_array)
            return {
                "max": param_max,
                "min": param_min,
                "param": quantized
            }

    def __call__(self, int_type=np.uint8):
        quantized_dict = collections.OrderedDict()
        layer = 0
        for name, param in self.param_dict.items():
            if param.shape == torch.Size([]):
                # Batch_Norm 不变
                quantized = param
            else:
                quantized = self.trans_to_int(param, int_type)
            quantized_dict[name] = quantized
            layer += 1
            if not isinstance(quantized, torch.Tensor):
                print(f"Quantized layer {layer}/{self.num_layers} | range: ({quantized['min']} ~ {quantized['max']})")

        type_info = np.iinfo(int_type)
        lower_lim = type_info.min
        upper_lim = type_info.max
        quantized_dict["lower_lim"] = lower_lim
        quantized_dict["upper_lim"] = upper_lim
        quantized_dict["layers"] = self.num_layers

        file_name = int_type.__name__ + '_' + self.file_name
        file_path = f"../weights/{file_name}"
        torch.save(quantized_dict, file_path)
        print(f"Saved as {os.path.abspath(file_path)}")


if __name__ == '__main__':
    pth_file = "../weights/VGG16_cifar10_8904.pth"
    qt = GlobalQuantizationTool(pth_file)
    qt()
