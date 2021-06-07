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
import torchvision.transforms as transforms


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
    data_loader_train, data_loader_eval = data_loader.get_dataset(config, True)
    logging.info('Dataset initialized')

    #Get loss function
    loss_fn = losses.get_loss_fn(config)

    logging.info(f'Starting training loop at epoch {epoch}')
    step = 0
    for i in range(epoch, config.training.epochs + 1):
        model.train()
        if config.model.name == 'fcn':
            scheduler.step()

        start_time = time.time()

        for img, seg in data_loader_train:
            img, seg = img.to(config.device), seg.to(config.device, dtype=torch.float32)

            #Training step
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, seg)
            loss.backward()
            optimizer.step()
            step += 1

            #Report training loss
            if step % config.training.log_freq == 0:
                logging.info('step: %d (epoch: %d), training_loss: %.5e' % (step, epoch, loss.item()))

            #Evaluation of model
            if step % config.training.eval_freq == 0:
                model.eval()
                loss_eval = 0

                for img_eval, seg_eval in data_loader_eval:
                    img_eval, seg_eval = img_eval.to(config.device), seg_eval.to(config.device)

                    with torch.no_grad():
                        pred_eval = model(img_eval)
                    loss_eval += loss_fn(pred_eval, seg_eval)
                logging.info(f'step: {step} (epoch: {epoch}), eval_loss: {loss_eval / len(data_loader_eval)}')
                model.train()


        #Save the checkpoint.
        logging.info(f'Saving checkpoint of epoch {epoch}')
        if epoch % config.training.checkpoint_save_freq == 0:
            utils.save_checkpoint(optimizer, model, epoch,
                                  os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))
        utils.save_checkpoint(optimizer, model, epoch,
                              os.path.join(checkpoint_dir, 'curr_cpt.pth'))


        time_for_epoch = time.time() - start_time
        logging.info(f'Finished epoch {epoch} ({step // epoch} steps in this epoch) in {time_for_epoch} seconds')
        epoch += 1

