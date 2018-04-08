import logging
import math
import os

import torch
from torch import nn, optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd


def group_weight(module):
    """Set weight decay to 0 for biases and batch norm. https://github.com/pytorch/pytorch/issues/1402"""
    group_decay = []
    group_no_decay = []

    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.bias is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=0.0)]
    return groups


class CosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, mult=1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.mul = mult
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max = self.T_max * self.mul
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(self.last_epoch / self.T_max * math.pi)) / 2
                for base_lr in self.base_lrs]


class CSVLogger:
    path = "logs/"

    def __init__(self, name, fields, delimiter=','):
        self.filename = name + ".csv"
        self.file = os.path.join(self.path, self.filename)
        self.fields = fields
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        with open(self.file, 'w') as f:
            f.write(delimiter.join(self.fields)+ '\n')

        file_handler = logging.FileHandler(self.file)
        field_tmpl = delimiter.join(['%({0})s'.format(x) for x in self.fields])
        file_handler.setFormatter(logging.Formatter(field_tmpl))
        self.logger.addHandler(file_handler)

    def log(self, values):
        self.logger.info('', extra=values)


def to_hms(s, e):
    seconds = e - s
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s