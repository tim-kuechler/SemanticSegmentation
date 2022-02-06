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
        return F.cross_entropy(pred, Variable(targets), ignore_index=config.model.ignore_index)

    return cross_entropy_one_hot_cityscapes

def get_nll_loss(config):
    def nll_loss_cityscapes(pred, targets):
        targets = torch.argmax(targets, dim=1)
        weights = torch.Tensor([0, 0.8373, 0.918, 0.866, 1.0345,
                                     1.0166, 0.9969, 0.9754, 1.0489,
                                     0.8786, 1.0023, 0.9539, 0.9843,
                                     1.1116, 0.9037, 1.0865, 1.0955,
                                     1.0865, 1.1529, 1.0507])
        weights = weights.to(config.device, dtype=torch.float32)
        return F.nll_loss(pred, Variable(targets), weight=weights, ignore_index=config.model.ignore_index)

    return nll_loss_cityscapes


def get_loss_fn(config):
    if config.model.name == 'unet':
        return get_cross_entropy_loss(config)
    if config.model.name == 'fcdense':
        return get_nll_loss(config)
    elif config.model.name == 'fcn':
        return F.binary_cross_entropy_with_logits


def get_step_fn(config, optimizer, model, loss_fn, scaler=None, train=True):
    def step_fn(img, target):
        # Training step
        if train:
            optimizer.zero_grad()
        if not config.optim.mixed_prec:
            pred = model(img)
            loss = loss_fn(pred, target)
            if train:
                loss.backward()
                optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss = loss_fn(pred, target)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        return loss.detach().item(), pred.detach().cpu()
    return step_fn



