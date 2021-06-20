import torch.optim as optim
import torch.nn.functional as F


def get_optimizer(config, model):
    if config.model.name == 'unet':
        optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.model.name == 'fcn':
        optimizer = optim.RMSprop(model.parameters(), lr=config.optim.lr, momentum=config.optim.momentum,
                                  weight_decay=config.optim.weight_decay)
    return optimizer

def _cross_entropy(pred, targets):
    return F.cross_entropy(pred, targets, ignore_index=255)

def get_loss_fn(config):
    if config.model.name == 'unet':
        return _cross_entropy
    elif config.model.name == 'fcn':
        return F.binary_cross_entropy_with_logits


