import torch
import torch.nn as nn
import os
from pathlib import Path
from models.unet.unet import UNet
from models.tiramisu.tiramisu import FCDenseNet103
import datasets.datasets as data_loader
import logging
import time
from models import utils
import losses
from torch.optim import lr_scheduler
from models.fcn import fcn, vgg_net
import numpy as np
from torchvision.utils import make_grid, save_image
import datasets.cityscapes256.cityscapes256 as cityscapes256
from datasets.flickr.flickr import save_output_images
import sde_lib


def train(config, workdir):
    # Create eval directory
    eval_dir = os.path.join(workdir, 'eval')
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Initialize model and optimizer
    if config.model.name == 'unet':
        model = UNet(config)
    elif config.model.name == 'fcdense':
        model = FCDenseNet103(config)
    elif config.model.name == 'fcn':
        assert config.model.conditional == False, "FCN can only be trained unconditionally"
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

    # Check for latest checkpoint
    if os.path.isfile(os.path.join(checkpoint_dir, 'curr_cpt.pth')):
        utils.restore_checkpoint(optimizer, model, os.path.join(checkpoint_dir, 'curr_cpt.pth'))
        logging.info('Checkpoint restored')

    # Create pred image directory
    pred_dir = os.path.join(workdir, 'pred_img')
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    #Get data iterators
    data_loader_train, data_loader_eval = data_loader.get_dataset(config)
    logging.info('Dataset initialized')

    # Get SDE
    if config.model.conditional:
        sde = sde_lib.get_SDE(config)
        logging.info('SDE initialized')

    #Get loss function
    loss_fn = losses.get_loss_fn(config)

    #Get scaler if mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Get step fn
    step_fn_train = losses.get_step_fn(config, optimizer, model, loss_fn, sde if config.model.conditional else None, scaler, train=True)
    step_fn_eval = losses.get_step_fn(config, optimizer, model, loss_fn, sde if config.model.conditional else None, scaler, train=False)

    logging.info(f'Starting training loop at epoch {epoch}')
    step = 0
    loss_per_log_period = 0
    for i in range(epoch, config.training.epochs + 1):
        start_time = time.time()
        model.train()

        for img, target in data_loader_train:
            img, target = img.to(config.device), target.to(config.device, dtype=torch.long)

            # Training step
            loss, _ = step_fn_train(img, target)
            step += 1

            #Report training loss
            loss_per_log_period += loss
            if step % config.training.log_freq == 0:
                mean_loss = loss_per_log_period / config.training.log_freq
                with open(os.path.join(workdir, 'training_loss.txt'), 'a+') as training_loss_file:
                    training_loss_file.write(str(step) + '\t' + str(mean_loss) + '\n')
                logging.info(f'step: {step} (epoch: {epoch}), training_loss: {mean_loss}')
                loss_per_log_period = 0

            #Evaluation of model loss
            if step % config.training.eval_freq == 0 and not config.model.conditional:
                model.eval()
                tot_eval_loss = 0

                for eval_img, eval_target in data_loader_eval:
                    eval_img, eval_target = eval_img.to(config.device), eval_target.to(config.device, dtype=torch.long)

                    with torch.no_grad():
                        pred_loss, _ = step_fn_eval(eval_img, eval_target)
                    tot_eval_loss += pred_loss
                with open(os.path.join(workdir, 'eval_loss.txt'), 'a+') as eval_loss_file:
                    eval_loss_file.write(str(step) + '\t' + str(tot_eval_loss) + '\n')
                logging.info(f'step: {step} (epoch: {epoch}), eval_loss: {tot_eval_loss / len(data_loader_eval)}')
                model.train()

        # FCDenseNet scheduler step
        if config.model.name == 'fcdense':
            new_lr = config.optim.lr * (config.optim.lr_decay ** (epoch // config.optim.step_size))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # FCN scheduler step
        if config.model.name == 'fcn':
            scheduler.step()

        # Save the checkpoint.
        logging.info(f'Saving checkpoint of epoch {epoch}')
        if epoch % config.training.checkpoint_save_freq == 0:
            utils.save_checkpoint(optimizer, model, epoch, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))
        utils.save_checkpoint(optimizer, model, epoch, os.path.join(checkpoint_dir, 'curr_cpt.pth'))

        # Save some predictions
        if epoch % config.training.save_pred_freq == 0:
            img, target = next(iter(data_loader_eval))
            img, target = img.to(config.device), target.to(config.device, dtype=torch.long)
            model.eval()

            with torch.no_grad():
                _, pred = step_fn_eval(img, target)

            # Create dir for epoch
            this_pred_dir = os.path.join(pred_dir, f'epoch_{epoch}')
            Path(this_pred_dir).mkdir(parents=True, exist_ok=True)

            # Save image
            nrow = int(np.sqrt(img.shape[0]))
            image_grid = make_grid(img, nrow, padding=2)
            save_image(image_grid, os.path.join(this_pred_dir, 'image.png'))

            if config.data.dataset == 'cityscapes256':
                # Save prediction and original map as color image
                cityscapes256.save_colorful_images(pred, this_pred_dir, 'pred.png')
                cityscapes256.save_colorful_images(target, this_pred_dir, 'mask.png')
            elif config.data.dataset == 'ade20k':
                # Save prediction and original map as color image
                #ade20k.save_colorful_images(pred, this_pred_dir, 'pred.png')
                #ade20k.save_colorful_images(target, this_pred_dir, 'mask.png')
                pass
            elif config.data.dataset == 'flickr':
                # Save prediction and original map as grayscale image
                save_output_images(pred, this_pred_dir, 'pred.png')
                save_output_images(target, this_pred_dir, 'mask.png')

            logging.info(f'Images for epoch {epoch} saved')

        #Evalutate model accuracy
        if epoch % config.training.full_eval_freq == 0:
            eval(config, workdir, while_training=True, model=model, data_loader_eval=data_loader_eval,
                 sde=None if not config.model.conditional else sde)

        time_for_epoch = time.time() - start_time
        logging.info(f'Finished epoch {epoch} ({step // epoch} steps in this epoch) in {time_for_epoch} seconds')
        epoch += 1


def experiment(config, workdir):
    for t in np.linspace(1., 0., 21):
        print('Starting timestep: ', t)
        iou, acc, noise = eval(config, workdir, timestep=t, save_to_file=False)
        with open(os.path.join(workdir, 'experiment.txt'), 'a+') as exp_file:
            exp_file.write(str(t) + '\t' + str(noise) + '\t' + str(acc) + '\t' + str(iou) + '\n')


def eval(config, workdir, while_training=False, model=None, data_loader_eval=None, sde=None, timestep=None, save_to_file=True):
    if not while_training:
        # Load model
        loaded_state = torch.load(os.path.join(workdir, 'checkpoints', 'curr_cpt.pth'), map_location=config.device)
        if config.model.name == 'unet':
            model = UNet(config)
        elif config.model.name == 'fcdense':
            model = FCDenseNet103(config)
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
        if config.model.conditional: assert sde is not None
    model.eval()

    total_ious = []
    pixel_accs = []
    for img, target in data_loader_eval:
        img = img.to(config.device)

        # Conditioning on noise scales
        if config.model.conditional:
            if timestep is None:
                eps = 1e-5
                t = torch.linspace(eps, config.training.start_noise, img.shape[0], device=config.device)
            else:
                t = torch.ones((img.shape[0]), device=config.device) * timestep
            z = torch.randn_like(img)
            mean, std = sde.marginal_prob(img, t)
            perturbed_img = mean + std[:, None, None, None] * z
            max = torch.ones(perturbed_img.shape[0], device=config.device)
            min = torch.ones(perturbed_img.shape[0], device=config.device)
            for N in range(perturbed_img.shape[0]):
                max[N] = torch.max(perturbed_img[N, :, :, :])
                min[N] = torch.min(perturbed_img[N, :, :, :])
            perturbed_img = perturbed_img - min[:, None, None, None] * torch.ones_like(img, device=config.device)
            perturbed_img = torch.div(perturbed_img, (max - min)[:, None, None, None])

        with torch.no_grad():
            pred = model(img) if not config.model.conditional else model(perturbed_img, std)
        pred = torch.argmax(pred, dim=1).cpu().numpy()

        target = torch.argmax(target, dim=1).cpu().numpy()

        for p, t in zip(pred, target):
            total_ious.append(_iou(p, t, config))
            pixel_accs.append(_pixel_acc(p, t))

    total_ious = np.array(total_ious).transpose()  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    if save_to_file:
        with open(os.path.join(workdir, 'eval_acc_iou.txt'), 'a+') as eval_file:
            eval_file.write(str(pixel_accs) + '\t' + str(np.nanmean(ious)) + '\n')
        with open(os.path.join(workdir, 'eval_label_iou.txt'), 'a+') as eval_file:
            eval_file.write(str(ious) + '\n')
    print(f'Evaluation:, pix_acc: {pixel_accs}, meanIoU: {np.nanmean(ious)}, IoUs: {ious}')
    return np.nanmean(ious), pixel_accs, std


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

