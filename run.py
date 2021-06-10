import torch
import torch.nn as nn
import os
from pathlib import Path
from models.unet.unet import UNet
import datasets.datasets as data_loader
import logging
import time
from models import utils
import losses
from torch.optim import lr_scheduler
from models.fcn import fcn, vgg_net
import numpy as np


def train(config, workdir):
    #Create eval directory
    eval_dir = os.path.join(workdir, 'eval')
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    #Initialize model and optimizer
    if config.model.name == 'unet':
        model = UNet(config.data.n_channels, config.data.n_labels)
    elif config.model.name == 'fcn':
        assert config.data.n_channels == 3
        vgg_model = vgg_net.VGGNet()
        model = fcn.FCNs(pretrained_net=vgg_model, n_class=config.data.n_labels)
    model = model.to(config.device)
    model = nn.DataParallel(model)

    #Get optimizer
    optimizer = losses.get_optimizer(config, model)
    if config.model.name == 'fcn':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.optim.step_size, gamma=config.optim.gamma)
    epoch = 1
    logging.info('Model and optimizer initialized')

    #Create checkpoint directories
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    #Get data iterators
    data_loader_train, data_loader_eval = data_loader.get_dataset(config, train=True)
    logging.info('Dataset initialized')

    #Get loss function
    loss_fn = losses.get_loss_fn(config)

    logging.info(f'Starting training loop at epoch {epoch}')
    step = 0
    for i in range(epoch, config.training.epochs + 1):
        start_time = time.time()
        model.train()

        for img, target in data_loader_train:
            img, target = img.to(config.device), target.to(config.device, dtype=torch.float32)

            #Training step
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            step += 1

            #Report training loss
            if step % config.training.log_freq == 0:
                logging.info('step: %d (epoch: %d), training_loss: %.5e' % (step, epoch, loss.item()))

            #Evaluation of model
            if step % config.training.eval_freq == 0:
                model.eval()
                tot_eval_loss = 0

                for eval_img, eval_target in data_loader_eval:
                    eval_img, eval_target = eval_img.to(config.device), eval_target.to(config.device)

                    with torch.no_grad():
                        eval_pred = model(eval_img)
                    tot_eval_loss += loss_fn(eval_pred, eval_target).item()
                logging.info(f'step: {step} (epoch: {epoch}), eval_loss: {tot_eval_loss / len(data_loader_eval)}')
                model.train()


        #Save the checkpoint.
        logging.info(f'Saving checkpoint of epoch {epoch}')
        if epoch % config.training.checkpoint_save_freq == 0:
            utils.save_checkpoint(optimizer, model, epoch,
                                  os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))
        utils.save_checkpoint(optimizer, model, epoch,
                              os.path.join(checkpoint_dir, 'curr_cpt.pth'))

        #FCN scheduler step
        if config.model.name == 'fcn':
            scheduler.step()

        time_for_epoch = time.time() - start_time
        logging.info(f'Finished epoch {epoch} ({step // epoch} steps in this epoch) in {time_for_epoch} seconds')
        epoch += 1

def eval(config, workdir):
    #Load model
    loaded_state = torch.load(os.path.join(workdir, 'curr_cpt.pth'), map_location=config.device)
    if config.model.name == 'unet':
        model = UNet(config.data.n_channels, config.data.n_labels)
    elif config.model.name == 'fcn':
        vgg_model = vgg_net.VGGNet()
        model = fcn.FCNs(pretrained_net=vgg_model, n_class=config.data.n_labels)
    model = model.to(config.device)
    model.load_state_dict(loaded_state['models'], strict=False)
    logging.info('Model loaded')

    # Get data iterators
    data_loader_train, data_loader_eval = data_loader.get_dataset(config, train=False)
    logging.info('Dataset initialized')

    total_ious = []
    pixel_accs = []
    for img, target in data_loader_eval:
        img = img.to(config.device)
        N, _, h, w = target.shape

        pred = model(img)
        pred = pred.cpu().numpy()
        pred = pred.transpose(0, 2, 3, 1).reshape(-1, config.data.n_labels).argmax(axis=1).reshape(N, h, w)

        target = target.cpu().numpy()
        target = target.transpose(0, 2, 3, 1).reshape(-1, config.data.n_labels).argmax(axis=1).reshape(N, h, w)

        for p, t in zip(pred, target):
            total_ious.append(_iou(p, t))
            pixel_accs.append(_pixel_acc(p, t))

        total_ious = np.array(total_ious).transpose()  # n_class * val_len
        ious = np.nanmean(total_ious, axis=1)
        pixel_accs = np.array(pixel_accs).mean()
        print(f'Evaluation:, pix_acc: {pixel_accs}, meanIoU: {np.nanmean(ious)}, IoUs: {ious}')


# borrow functions and modify it from
# Calculates class intersections over unions
def _iou(pred, target, config):
    ious = []
    for cls in range(config.data.n_labels):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def _pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total
