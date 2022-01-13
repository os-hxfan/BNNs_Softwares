from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from software.models.dropout import BernoulliDropout
from software.utils import Flatten


@dataclass
class MinMax:
    min: float
    max: float
    def delta(self):
        return self.max - self.min

# IVIMNET Parameters.
bounds_D  = MinMax(0.0, 0.005) # Dt
bounds_f  = MinMax(0.0, 0.7)   # Fp
bounds_Dp = MinMax(0.005, 0.2) # Ds
bounds_f0 = MinMax(0.0, 2.0)   # S0


class IVIMNET(nn.Module):
    """ Simple linear regression model. """
    def __init__(self, bvalues: np.array, args):
        super(IVIMNET, self).__init__()
        self.args = args
        self.bvalues = bvalues
        self.fc_layers = nn.ModuleList([self._make_fc_layer() for _ in range(4)])
        self.encoders = nn.ModuleList([nn.Sequential(*fc_layer, nn.Linear(len(bvalues), 1))
                                       for fc_layer in self.fc_layers])

    def forward(self, x, samples=1, profile=False):
        if not profile:
            return self._forward_no_profile(x)
        else:
            return self._forward_profile(x, samples)

    def _forward_profile(self, x, samples):
        raise NotImplementedError()
        static = -1
        cache = None
        with profiler.record_function("static_part"):
            ...

        out = []
        with profiler.record_function("dynamic_part"):
            ...
        return out

    def _forward_no_profile(self, x):
        def sigm(val, lim):
            return lim.min + torch.sigmoid(val[:, 0].unsqueeze(1)) * lim.delta()

        # Apply constraints.
        params = [enc(x) for enc in self.encoders]
        Dt = sigm(params[2], bounds_D)
        Fp = sigm(params[0], bounds_f)
        Dp = sigm(params[1], bounds_Dp)
        f0 = sigm(params[3], bounds_f0)

        X = torch.cat((Fp * torch.exp(-self.bvalues * Dp) + f0 * torch.exp(-self.bvalues * Dt)), dim=1)
        return X, Dt, Fp / (f0 + Fp), Dp, f0 + Fp

    def _make_fc_layer(self):
        width = len(self.bvalues)
        return nn.ModuleList([
            nn.Linear(width, width),
            BernoulliDropout(0.1),
            nn.BatchNorm1d(width),
            nn.ELU(),
            # nn.Dropout(0.1),

            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ELU(),
        ])