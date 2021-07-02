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
    if config.data.dataset == 'ade20k':
        return cross_entropy_one_hot_flickr

def get_nll_loss(config):
    def nll_loss_cityscapes(pred, targets):
        targets = torch.argmax(targets, dim=1)
        weights = torch.Tensor([0, 0.8373, 0.918, 0.866, 1.0345,
                                     1.0166, 0.9969, 0.9754, 1.0489,
                                     0.8786, 1.0023, 0.9539, 0.9843,
                                     1.1116, 0.9037, 1.0865, 1.0955,
                                     1.0865, 1.1529, 1.0507])
        weights = weights.to(config.device, dtype=torch.float32)
        return F.nll_loss(pred, Variable(targets), weight=weights, ignore_index=0)

    def nll_loss_flickr(pred, targets):
        targets = torch.argmax(targets, dim=1)
        return F.nll_loss(pred, Variable(targets))

    if config.data.dataset == 'cityscapes256':
        return nll_loss_cityscapes
    elif config.data.dataset == 'flickr':
        return nll_loss_flickr
    elif config.data.dataset == 'ade20k':
        return nll_loss_flickr


def get_loss_fn(config):
    if config.model.name == 'unet':
        return get_cross_entropy_loss(config)
    if config.model.name == 'fcdense':
        return get_nll_loss(config)
    elif config.model.name == 'fcn':
        return F.binary_cross_entropy_with_logits


def get_step_fn(config, optimizer, model, loss_fn, sde=None, scaler=None, train=True):
    def step_fn(img, target):
        # Conditioning on noise scales
        if config.model.conditional:
            # t = (0.4 - 1) * torch.rand(int(img.shape[0]), device=config.device) + 1
            eps = 1e-5
            if train:
                t = torch.rand(int(img.shape[0]), device=config.device) * (1 - eps) + eps
            else:
                t = torch.linspace(1, eps, img.shape[0], device=config.device)
            z = torch.randn_like(img)
            mean, std = sde.marginal_prob(img, t)
            perturbed_img = mean + std[:, None, None, None] * z
            #max = torch.ones(perturbed_img.shape[0], device=config.device)
            #min = torch.ones(perturbed_img.shape[0], device=config.device)
            #for N in range(perturbed_img.shape[0]):
            #    max[N] = torch.max(perturbed_img[N, :, :, :])
            #    min[N] = torch.min(perturbed_img[N, :, :, :])
            #perturbed_img = perturbed_img - min[:, None, None, None] * torch.ones_like(img, device=config.device)
            #perturbed_img = torch.div(perturbed_img, (max - min)[:, None, None, None])

        # Training step
        if train:
            optimizer.zero_grad()
        if not config.optim.mixed_prec:
            pred = model(img) if not config.model.conditional else model(perturbed_img, std)
            loss = loss_fn(pred, target)
            if train:
                loss.backward()
                optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                pred = model(img) if not config.model.conditional else model(perturbed_img, std)
                loss = loss_fn(pred, target)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        return loss.detach().item(), pred.detach().cpu()
    return step_fn



