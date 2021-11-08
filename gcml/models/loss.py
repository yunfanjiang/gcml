import torch
import numpy as np


def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return ((input-target)**2).mean()