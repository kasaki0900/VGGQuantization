import torch


def clip_cut(pth_path, threshold_percent):
    """original model"""
    params = torch.load(pth_path)
    for name, param in params.items():
        if not param.shape == torch.Size([]):
            low_value = torch.quantile(param, threshold_percent / 100)
            high_value = torch.quantile(param, 1 - (threshold_percent / 100))
            params[name] = torch.clamp(param, max=high_value, min=low_value)

    return params

