import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


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
    targets = torch.argmax(targets, dim=1)
    cross_entr = nn.CrossEntropyLoss()
    return cross_entr(pred, Variable(targets))

def get_loss_fn(config):
    return _cross_entropy_one_hot

def get_step_fn(config, model, optimizer, loss_fn, scaler=None):
    """
    Gets the step function for training

    :param config: The config
    :param optimizer: The optimizer to use
    :param loss_fn: The loss function of the model
    :param ema: The EMA of the model
    :param scaler: The torch.cuda.amp.GradScaler if config.optim.mixed_prec is True
    :return: The training step function
    """
    def step_fn(img, target):
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        return loss

    def step_fn_mixed_prec(img, target):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(img)
            print(pred)
            loss = loss_fn(pred, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss

    if not config.optim.mixed_prec:
        return step_fn
    else:
        assert scaler is not None
        return step_fn_mixed_prec


