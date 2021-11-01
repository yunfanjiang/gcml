import torch.nn as nn


class BaseGoalReachAgent(nn.Module):
    def step(self, *args, **kwargs):
        raise NotImplementedError
