import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


def get_optimizer(config, model):
    if config.model.name == 'unet':
        optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    if config.model.name == 'fcdense':
        optimizer = optim.RMSprop(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.model.name == 'fcn':
        optimizer = optim.RMSprop(model.parameters(), lr=config.optim.lr, momentum=config.optim.momentum,
                                  weight_decay=config.optim.weight_decay)
    return optimizer


def get_cross_entropy_loss(config):
    def cross_entropy_one_hot_cityscapes(pred, targets):
        targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(pred, Variable(targets), ignore_index=0)

    def cross_entropy_one_hot_flickr(pred, targets):
        targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(pred, Variable(targets))

    if config.data.dataset == 'cityscapes256':
        return cross_entropy_one_hot_cityscapes
    elif config.data.dataset == 'flickr':
        return cross_entropy_one_hot_flickr

def get_nll_loss(config):
    def nll_loss_cityscapes(pred, targets):
        weights = torch.FloatTensor([0, 0.8373, 0.918, 0.866, 1.0345,
                                     1.0166, 0.9969, 0.9754, 1.0489,
                                     0.8786, 1.0023, 0.9539, 0.9843,
                                     1.1116, 0.9037, 1.0865, 1.0955,
                                     1.0865, 1.1529, 1.0507])
        return F.nll_loss(pred, targets, weight=weights, ignore_index=0)

    def nll_loss_flickr(pred, targets):
        return F.nll_loss(pred, targets)

    if config.data.dataset == 'cityscapes256':
        return nll_loss_cityscapes
    elif config.data.dataset == 'flickr':
        return nll_loss_flickr


def get_loss_fn(config):
    if config.model.name == 'unet':
        return get_cross_entropy_loss(config)
    if config.model.name == 'fcdense':
        return get_nll_loss(config)
    elif config.model.name == 'fcn':
        return F.binary_cross_entropy_with_logits


