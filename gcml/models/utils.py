from typing import Union

import torch
import numpy as np


def np_dict_to_tensor_dict(np_dict: dict, device):
    tensor_dict = {}
    for k, v in np_dict.items():
        if isinstance(v, dict):
            sub_tensor_dict = np_dict_to_tensor_dict(v, device)
            tensor_dict[k] = sub_tensor_dict
        elif isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, torch.Tensor):
            tensor_dict[k] = v.to(device)
        else:
            raise ValueError(f"Unknown type {type(v)}")
    return tensor_dict


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Code borrowed from stable-baselines3
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device
