import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
  def __init__(self, args):
      super(ClassificationLoss, self).__init__()
      self.args = args 
      self.ce = _SmoothCrossEntropyLoss(smoothing=self.args.smoothing)
      
  def forward(self, outs, targets, model, n_batches, n_points):
      ce = self.ce(outs, targets)
      return ce, ce


class MSELoss(nn.Module):
    def __init__(self, args):
        super(MSELoss, self).__init__()
        self.args = args

    def forward(self, outs, target, model, n_batches, n_points):
        assert len(outs) == len(target)

        squared_error = (outs - target) ** 2
        mse = squared_error.sum() / len(outs)
        return mse, mse


class _SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets, n_classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets, smoothing = None):
        if smoothing is None:
          smoothing = self.smoothing
        targets = _SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         smoothing)
        lsm = torch.log(inputs)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss