from typing import Dict, Callable, Union

import torch
import numpy as np
import torch.nn.functional as F

from .utils import np_dict_to_tensor_dict


class MetaGoalReachAgentBase(object):
    """
    Base class for meta goal-reaching agent.
    It implements a method to get logits, which is further processed for discrete outputs or continuous outputs.
    """

    def __init__(
        self,
        n_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        device,
        use_task_config: bool = False,
    ):
        self._n_layers = n_layers
        self._activation = activation
        self._device = device
        self._use_task_config = use_task_config

    def _get_logits(
        self,
        input_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        parameters: Dict[str, Union[torch.Tensor, np.ndarray]],
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
        if self._use_task_config:
            inputs = torch.cat([inputs, input_dict["task_config"].float()], dim=-1)
        x = inputs
        for i in range(self._n_layers - 1):
            x = F.linear(input=x, weight=parameters[f"w{i}"], bias=parameters[f"b{i}"],)
            x = self._activation(x)
        logits = F.linear(
            input=x,
            weight=parameters[f"w{self._n_layers - 1}"],
            bias=parameters[f"b{self._n_layers - 1}"],
        )
        self._logits = logits

    def act(
        self,
        input_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        parameters: Dict[str, Union[torch.Tensor, np.ndarray]],
        greedy: bool,
    ):
        """
        First call `self._get_logits()`, then process the logits to get corresponding types of outputs.
        Note that in general we do not propagate gradients from actions (we propagate gradients back from logits).
        """
        raise NotImplementedError

    @property
    def logits(self):
        return self._logits


class MetaGoalReachAgentDiscrete(MetaGoalReachAgentBase):
    def act(
        self,
        input_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        parameters: Dict[str, Union[torch.Tensor, np.ndarray]],
        greedy: bool,
    ):
        self._get_logits(
            input_dict=input_dict, parameters=parameters,
        )
        if greedy:
            action = torch.argmax(self.logits, dim=-1)
        else:
            action = torch.distributions.Categorical(logits=self.logits).sample()
        return action


class MetaGoalReachAgentContinuous(MetaGoalReachAgentBase):
    def __init__(
        self,
        n_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        device,
        std: float,
        use_task_config: bool = False,
    ):
        super(MetaGoalReachAgentContinuous, self).__init__(
            n_layers=n_layers,
            activation=activation,
            device=device,
            use_task_config=use_task_config,
        )
        self._std = std

    def act(
        self,
        input_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        parameters: Dict[str, Union[torch.Tensor, np.ndarray]],
        greedy: bool,
    ):
        self._get_logits(
            input_dict=input_dict, parameters=parameters,
        )
        if greedy:
            # if greedy, return the regressed action
            action = self.logits
        else:
            # if not greedy, sample from Gaussian distributions centered at logits
            action = torch.distributions.Normal(
                loc=self.logits, scale=self._std
            ).sample()
        return action
