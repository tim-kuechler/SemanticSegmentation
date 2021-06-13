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
from torchvision.utils import make_grid, save_image
from datasets.cityscapes256.cityscapes256 import trainId2Color


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
        vgg_model.to(config.device)
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

    #Create pred image directory
    pred_dir = os.path.join(workdir, 'pred_img')
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    #Create score dirs
    score_dir = os.path.join(workdir, "scores")
    Path(score_dir).mkdir(parents=True, exist_ok=True)
    IU_scores = np.zeros((config.training.epochs, config.data.n_labels))
    pixel_scores = np.zeros(config.training.epochs)

    #Get data iterators
    data_loader_train, data_loader_eval = data_loader.get_dataset(config)
    logging.info('Dataset initialized')

    #Get loss function
    loss_fn = losses.get_loss_fn(config)

    #Get step function
    scaler = None if not config.optim.mixed_prec else torch.cuda.amp.GradScaler()
    step_fn = losses.get_step_fn(config, model, optimizer, loss_fn, scaler)

    logging.info(f'Starting training loop at epoch {epoch}')
    step = 0
    for i in range(epoch, config.training.epochs + 1):
        start_time = time.time()
        model.train()

        for img, target in data_loader_train:
            img, target = img.to(config.device), target.to(config.device, dtype=torch.float32)

            #Training step
            loss = step_fn(img, target)
            step += 1

            #Report training loss
            if step % config.training.log_freq == 0:
                logging.info('step: %d (epoch: %d), training_loss: %.5e' % (step, epoch, loss.item()))

            #Evaluation of model loss
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
            utils.save_checkpoint(optimizer, model, epoch, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))
        utils.save_checkpoint(optimizer, model, epoch, os.path.join(checkpoint_dir, 'curr_cpt.pth'))

        #FCN scheduler step
        if config.model.name == 'fcn':
            scheduler.step()

        #Save some predictions
        i = 0
        for img, target in data_loader_eval:
            if i == 1:
                break
            model.eval()
            pred = model(img)
            pred = torch.argmax(pred, dim=1)

            #Create dir for epoch
            this_pred_dir = os.path.join(pred_dir, f'epoch_{epoch}')
            Path(this_pred_dir).mkdir(parents=True, exist_ok=True)

            #Save image
            nrow = int(np.sqrt(img.shape[0]))
            image_grid = make_grid(img, nrow, padding=2)
            save_image(image_grid, os.path.join(this_pred_dir, 'image.png'))

            #Save prediction as color image
            pred_color = torch.zeros((pred.shape[0], 3, pred.shape[1], pred.shape[2]), device=config.device)
            for N in range(0, pred.shape[0]):
                for h in range(0, pred.shape[1]):
                    for w in range(0, pred.shape[1]):
                        color = trainId2Color[str(pred[N, h, w].item())]
                        pred_color[N, 0, h, w] = color[0]
                        pred_color[N, 1, h, w] = color[1]
                        pred_color[N, 2, h, w] = color[2]
            nrow = int(np.sqrt(pred_color.shape[0]))
            image_grid = make_grid(pred_color, nrow, padding=2)
            save_image(image_grid, os.path.join(this_pred_dir, 'pred.png'))

            i += 1
        logging.info(f'Images for epoch {epoch} saved')

        #Evalutate model accuracy
        if epoch % config.training.full_eval_freq == 0:
            eval(config, workdir, while_training=True, model=model, data_loader_eval=data_loader_eval)

        time_for_epoch = time.time() - start_time
        logging.info(f'Finished epoch {epoch} ({step // epoch} steps in this epoch) in {time_for_epoch} seconds')
        epoch += 1


def eval(config, workdir, while_training=False, model=None, data_loader_eval=None):
    if not while_training:
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
        data_loader_train, data_loader_eval = data_loader.get_dataset(config)
        logging.info('Dataset initialized')
    else:
        assert model is not None
        assert data_loader_eval is not None
    model.eval()

    total_ious = []
    pixel_accs = []
    for img, target in data_loader_eval:
        img = img.to(config.device)

        pred = model(img)
        pred = torch.argmax(pred, dim=1).cpu().numpy()

        target = torch.argmax(target, dim=1).cpu().numpy()

        for p, t in zip(pred, target):
            total_ious.append(_iou(p, t, config))
            pixel_accs.append(_pixel_acc(p, t))

    total_ious = np.array(total_ious).transpose()  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print(f'Evaluation:, pix_acc: {pixel_accs}, meanIoU: {np.nanmean(ious)}, IoUs: {ious}')


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
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
