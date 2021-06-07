import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


def get_optimizer(config, model):
    if config.model.name == 'unet':
        optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.model.name == 'fcn':
        optimizer = optim.RMSprop(model.parameters(), lr=config.optim.lr, momentum=config.optim.momentum,
                                  weight_decay=config.optim.weight_decay)
    return optimizer

def _cross_entropy_soft(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def _cross_entropy_one_hot(pred, targets):
    targets = torch.argmax(targets, 1)
    #_, targets = targets.max(dim=1)
    cross_entr = nn.CrossEntropyLoss()
    return cross_entr(pred, Variable(targets))


def get_loss_fn(config):
    return _cross_entropy_one_hot

