import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def get_optimizer(config, model):
    if config.model.name == 'unet':
        optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.model.name == 'fcn':
        optimizer = optim.RMSprop(model.parameters(), lr=config.optim.lr, momentum=config.optim.momentum,
                                  weight_decay=config.optim.weight_decay)
    return optimizer

def _cross_entropy_one_hot_cityscapes(pred, targets):
    targets = torch.argmax(targets, dim=1)
    return F.cross_entropy(pred, Variable(targets), ignore_index=0)

def _cross_entropy_one_hot_flickr(pred, targets):
    targets = torch.argmax(targets, dim=1)
    return F.cross_entropy(pred, Variable(targets))

def get_loss_fn(config):
    if config.model.name == 'unet':
        if config.data.dataset == 'cityscapes256':
            return _cross_entropy_one_hot_cityscapes
        elif config.data.dataset == 'flickr':
            return _cross_entropy_one_hot_flickr
    elif config.model.name == 'fcn':
        return F.binary_cross_entropy_with_logits


