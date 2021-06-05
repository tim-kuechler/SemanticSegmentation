import torch
import torch.optim as optim
import torch.nn as nn
import os
from pathlib import Path
from models.unet.unet import UNet
import datasets.datasets as data_loader
import logging
import time
from models import utils
import torch.nn.functional as F
import torchvision.transforms as transforms



def train(config, workdir):
    #Create eval directory
    eval_dir = os.path.join(workdir, 'eval')
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    #Initialize model and optimizer
    if config.model.name == 'unet':
        model = UNet(config.data.n_channels, config.model.n_labels)
    elif config.model.name == 'fcn':
        pass
    model = model.to(config.device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                           eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    epoch = 1
    logging.info('Model and optimizer initialized')

    #Create checkpoint directories
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    #Get data iterators
    data_loader_train, data_loader_eval = data_loader.get_dataset(config, True)
    logging.info('Dataset initialized')

    #Define loss function
    loss_fn = nn.CrossEntropyLoss()

    logging.info(f'Starting training loop at epoch {epoch}')
    step = 0
    for i in range(epoch, config.training.epochs + 1):
        model.train()
        start_time = time.time()

        for img, seg in data_loader_train:
            img = img.to(config.device)
            seg = seg.to(config.device)

            #Training step
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, seg)
            loss.backward()
            if config.optim.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optim.grad_clip)
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
                    img_eval = img_eval.to(config.device)
                    seg_eval = seg_eval.to(config.device)

                    with torch.no_grad():
                        pred = model(img_eval)
                    loss_eval += F.cross_entropy(pred, seg_eval)
                logging.info(f'step: {step} (epoch: {epoch}), eval_loss: {loss_eval}')
                model.train()


        # Save a checkpoint periodically
        # Save the checkpoint.
        logging.info(f'Saving checkpoint of epoch {epoch}')
        if epoch % config.training.checkpoint_save_freq == 0:
            utils.save_checkpoint(optimizer, model, epoch,
                                  os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))
        utils.save_checkpoint(optimizer, model, epoch,
                              os.path.join(checkpoint_dir, 'curr_cpt.pth'))

        #Save some pictures
        model.eval()
        eval_iter = iter(data_loader_eval)
        for i in range(config.training.n_eval_imgs):
            img_eval, img_seg = next(eval_iter)
            img_eval = img_eval.to(config.device)
            with torch.no_grad():
                img_seg_pred = model(img_eval)
            img_eval = img_eval.to(torch.device('cpu'))
            img_seg_pred = img_seg_pred.to(torch.device('cpu'))

            #Save images
            to_pil = transforms.ToPILImage()
            pil_img = to_pil(img_eval)
            pil_img.save(os.path.join(eval_dir, f'{i}_img.png'), 'PNG')
            pil_seg = to_pil(img_seg)
            pil_seg.save(os.path.join(eval_dir, f'{i}_seg.png'), 'PNG')
            pil_pred_seg = to_pil(img_seg_pred)
            pil_pred_seg.save(os.path.join(eval_dir, f'{i}_pred_seg.png'), 'PNG')


        time_for_epoch = time.time() - start_time
        logging.info(f'Finished epoch {epoch} ({step // epoch} steps in this epoch) in {time_for_epoch} seconds')
        epoch += 1

