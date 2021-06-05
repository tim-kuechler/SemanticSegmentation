import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from models.fcn.fcn8s import get_parameters
from distutils.version import LooseVersion
import torch.nn.functional as F


def get_optimizer(config, model):
    if config.model.name == 'unet':
        optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.model.name == 'fcn':
        optimizer = optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': config.optim.lr * 2, 'weight_decay': 0},
            ],
            lr=config.optim.lr, momentum=config.optim.momentum, weight_decay=config.optim.weight_decay)
    return optimizer

def _cross_entropy_soft(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def _cross_entropy_one_hot(pred, targets):
    print(type(targets), targets.size())
    targets = torch.argmax(targets, 1)
    print(type(targets), targets.size())
    _, targets = targets.max(dim=1)
    print(type(targets), targets.size())
    print(type(Variable(targets)), Variable(targets).size())
    cross_entr = nn.CrossEntropyLoss()
    return cross_entr(pred, Variable(targets))

def _cross_entropy2d(pred, targets):
    # input: (n, c, h, w), target: (n, c, h, w)
    targets = torch.argmax(targets, 1)
    n, c, h, w = pred.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(pred)
    else:
        # >=0.3
        log_p = F.log_softmax(pred, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = targets >= 0
    target = targets[mask]
    loss = F.nll_loss(log_p, target, reduction='sum')
    return loss

def get_loss_fn(config):
    if config.model.name == 'unet':
        return _cross_entropy_one_hot
    elif config.model.name == 'fcn':
        return _cross_entropy2d
