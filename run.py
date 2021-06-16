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
import sde_lib


def train(config, workdir):
    # Create eval directory
    eval_dir = os.path.join(workdir, 'eval')
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Initialize model and optimizer
    if config.model.name == 'unet':
        model = UNet(config)
    elif config.model.name == 'fcn':
        assert config.training.conditional == False, "FCN can only be trained unconditionally"
        assert config.data.n_channels == 3, "FCN can only be trained on 3 channel images"
        vgg_model = vgg_net.VGGNet()
        model = fcn.FCNs(pretrained_net=vgg_model, n_class=config.data.n_labels)
        vgg_model.to(config.device)
    model = model.to(config.device)
    model = nn.DataParallel(model)

    # Get optimizer
    optimizer = losses.get_optimizer(config, model)
    if config.model.name == 'fcn':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.optim.step_size, gamma=config.optim.gamma)
    epoch = 1
    logging.info('Model and optimizer initialized')

    # Create checkpoint directories
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create pred image directory
    pred_dir = os.path.join(workdir, 'pred_img')
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    #Get data iterators
    data_loader_train, data_loader_eval = data_loader.get_dataset(config)
    logging.info('Dataset initialized')

    # Get SDE
    sde = sde_lib.get_SDE(config)
    logging.info('SDE initialized')

    #Get loss function
    loss_fn = losses.get_loss_fn(config)

    #Get step function
    scaler = None if not config.optim.mixed_prec else torch.cuda.amp.GradScaler()

    logging.info(f'Starting training loop at epoch {epoch}')
    step = 0
    for i in range(epoch, config.training.epochs + 1):
        start_time = time.time()
        model.train()

        for img, target in data_loader_train:
            img, target = img.to(config.device), target.to(config.device, dtype=torch.float32)

            # Conditioning on noise scales
            if config.training.conditional:
                eps = 1e-5
                t = torch.rand(img.shape[0], device=config.device) * (1 - eps) + eps
                z = torch.randn_like(img)
                mean, std = sde.marginal_prob(img, t)
                print(std)
                perturbed_img = mean + std[:, None, None, None] * z
                noise = sde.marginal_prob(torch.zeros_like(perturbed_img), t)[1]
                print(noise)

            #Training step
            optimizer.zero_grad()
            if not config.optim.mixed_prec:
                pred = model(img) if not config.training.conditional else model(perturbed_img, noise)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
            else:
                with torch.cuda.amp.autocast():
                    pred = model(img) if not config.training.conditional else model(perturbed_img, noise)
                    loss = loss_fn(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            step += 1

            #Report training loss
            if step % config.training.log_freq == 0:
                logging.info('step: %d (epoch: %d), training_loss: %.5e' % (step, epoch, loss.item()))

            #Evaluation of model loss
            if step % config.training.eval_freq == 0:
                model.eval()
                tot_eval_loss = 0

                for eval_img, eval_target in data_loader_eval:
                    eval_img, eval_target = eval_img.to(config.device), eval_target.to(config.device, dtype=torch.float32)

                    with torch.no_grad():
                        eval_pred = model(eval_img)
                    tot_eval_loss += loss_fn(eval_pred, eval_target).item()
                logging.info(f'step: {step} (epoch: {epoch}), eval_loss: {tot_eval_loss / len(data_loader_eval)}')
                model.train()

        # Save the checkpoint.
        logging.info(f'Saving checkpoint of epoch {epoch}')
        if epoch % config.training.checkpoint_save_freq == 0:
            utils.save_checkpoint(optimizer, model, epoch, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))
        utils.save_checkpoint(optimizer, model, epoch, os.path.join(checkpoint_dir, 'curr_cpt.pth'))

        # FCN scheduler step
        if config.model.name == 'fcn':
            scheduler.step()

        # Save some predictions
        for i, (img, target) in enumerate(data_loader_eval):
            img, target = img.to(config.device), target.to(config.device, dtype=torch.float32)
            if i == 1:
                break
            model.eval()

            # Conditioning on noise scales
            if config.training.conditional:
                eps = 1e-5
                t = torch.rand(img.shape[0], device=config.device) * (1 - eps) + eps
                z = torch.randn_like(img)
                mean, std = sde.marginal_prob(img, t)
                perturbed_img = mean + std[:, None, None, None] * z
                noise = sde.marginal_prob(torch.zeros_like(perturbed_img), t)[1]

            pred = model(img) if not config.training.conditional else model(perturbed_img, noise)
            pred = torch.argmax(pred, dim=1)
            target = torch.argmax(target, dim=1)

            # Create dir for epoch
            this_pred_dir = os.path.join(pred_dir, f'epoch_{epoch}')
            Path(this_pred_dir).mkdir(parents=True, exist_ok=True)

            # Save image
            nrow = int(np.sqrt(img.shape[0]))
            image_grid = make_grid(img, nrow, padding=2)
            save_image(image_grid, os.path.join(this_pred_dir, 'image.png'))

            # Save prediction and original mask as color image
            _save_map(pred, this_pred_dir, 'pred.png')
            _save_map(target, this_pred_dir, 'mask.png')
        logging.info(f'Images for epoch {epoch} saved')

        #Evalutate model accuracy
        if epoch % config.training.full_eval_freq == 0:
            eval(config, workdir, while_training=True, model=model, data_loader_eval=data_loader_eval, sde=sde)

        time_for_epoch = time.time() - start_time
        logging.info(f'Finished epoch {epoch} ({step // epoch} steps in this epoch) in {time_for_epoch} seconds')
        epoch += 1


def eval(config, workdir, while_training=False, model=None, data_loader_eval=None, sde=None):
    if not while_training:
        # Load model
        loaded_state = torch.load(os.path.join(workdir, 'curr_cpt.pth'), map_location=config.device)
        if config.model.name == 'unet':
            model = UNet(config)
        elif config.model.name == 'fcn':
            vgg_model = vgg_net.VGGNet()
            model = fcn.FCNs(pretrained_net=vgg_model, n_class=config.data.n_labels)
        model = model.to(config.device)
        model.load_state_dict(loaded_state['models'], strict=False)
        logging.info('Model loaded')

        # Get data iterators
        data_loader_train, data_loader_eval = data_loader.get_dataset(config)
        logging.info('Dataset initialized')

        # Get SDE
        sde = sde_lib.get_SDE(config)
        logging.info('SDE initialized')
    else:
        assert model is not None
        assert data_loader_eval is not None
        assert sde is not None
    model.eval()

    total_ious = []
    pixel_accs = []
    for img, target in data_loader_eval:
        img = img.to(config.device)

        # Conditioning on noise scales
        if config.training.conditional:
            eps = 1e-5
            t = torch.rand(img.shape[0], device=config.device) * (1 - eps) + eps
            z = torch.randn_like(img)
            mean, std = sde.marginal_prob(img, t)
            perturbed_img = mean + std[:, None, None, None] * z
            noise = sde.marginal_prob(torch.zeros_like(perturbed_img), t)[1]

        pred = model(img) if not config.training.conditional else model(perturbed_img, noise)
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
def _iou(pred, target, config):
    """ Calculates class intersections over unions

    :param pred: The predicted map
    :param target: The targed segmentation map
    :param config: The config
    :return: Iou
    """
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
    """ Calculate pixel accuracy of prediction in comparison to original map

    :param pred:
    :param target:
    :return: pixel accuracy
    """
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total

def _save_map(map, filename, save_dir):
    """
    Saves a segmentation map as color image

    :param mask: A mask in format (N, H, W) (not one-hot!)
    :param filename: The name of the image file
    :param save_dir: The directory to save to
    """
    mask_color = torch.zeros((map.shape[0], 3, map.shape[1], map.shape[2]))
    for N in range(0, map.shape[0]):
        for h in range(0, map.shape[1]):
            for w in range(0, map.shape[2]):
                color = trainId2Color[str(map[N, h, w].item())]
                mask_color[N, 0, h, w] = color[0]
                mask_color[N, 1, h, w] = color[1]
                mask_color[N, 2, h, w] = color[2]
    nrow = int(np.sqrt(mask_color.shape[0]))
    image_grid = make_grid(mask_color, nrow, padding=2, normalize=True)
    save_image(image_grid, os.path.join(save_dir, filename))