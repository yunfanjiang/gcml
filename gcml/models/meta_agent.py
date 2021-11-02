from typing import Dict, Callable, Union

import torch
import numpy as np
import torch.nn.functional as F

from ..algorithms.common.utils import np_dict_to_tensor_dict


class MetaGoalReachAgent(object):
    def __init__(
        self, n_layers: int, activation: Callable[[torch.Tensor], torch.Tensor], device,
    ):
        self._n_layers = n_layers
        self._activation = activation
        self._device = device

    def act(
        self,
        input_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        parameters: Dict[str, Union[torch.Tensor, np.ndarray]],
        noise_coeff: float,
        greedy: bool,
    ):
        input_dict = np_dict_to_tensor_dict(input_dict, self._device)
        inputs = torch.cat(
            [
                input_dict["observation"],
                input_dict["achieved_goal"],
                input_dict["achieved_state_goal"],
                input_dict["desired_goal"],
                input_dict["desired_state_goal"],
            ],
            dim=-1,
        )
        x = inputs
        for i in range(self._n_layers - 1):
            x = F.linear(input=x, weight=parameters[f"w{i}"], bias=parameters[f"b{i}"],)
            x = self._activation(x)
        logits = F.linear(
            input=x,
            weight=parameters[f"w{self._n_layers - 1}"],
            bias=parameters[f"b{self._n_layers - 1}"],
        )
        logits *= 1 - noise_coeff
        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits)
        return action
