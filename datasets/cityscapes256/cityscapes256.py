"""Adapted from torchvision.datasets.cityscapes and https://github.com/pochih/FCN-pytorch/blob/master/python/Cityscapes_utils.py"""
from torch.utils.data import Dataset
import os
from PIL import Image
from os.path import exists, split
import torchvision.transforms.functional as F
from random import randint
import numpy as np
import torch.nn.functional


class CITYSCAPES256(Dataset):
    """Cityscapes256 Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
    """
    def __init__(self, root, split="train", mode="fine", crop=True):
        self.root = root
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.split = split
        self.images = []
        self.targets = []
        self.n_labels = 20
        self.crop = crop

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], f'{self.mode}_labelIds.png')

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        #Load and resize
        image = Image.open(self.images[index]).convert('RGB')
        image = F.resize(image, [256, 512], interpolation=F.InterpolationMode.BICUBIC)

        target = Image.open(self.targets[index])
        target = F.resize(target, [256, 512], interpolation=F.InterpolationMode.NEAREST)

        #Crop
        if self.crop:
            left = randint(0, 256)
            image = F.crop(image, 0, left, 256, 256)
            target = F.crop(target, 0, left, 256, 256)

        #To tensor
        image = F.to_tensor(image)
        target = F.to_tensor(target) * 255
        target = target.long()
        target = torch.squeeze(target, dim=0)
        print(target.size())
        target = target.permute(2, 0, 1)

        return image, target

    def __len__(self):
        return len(self.images)


# borrow functions and modify it from https://github.com/fyu/drn/blob/master/segment.py
CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


def save_colorful_images(pred, output_dir, filename):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    pred = torch.argmax(pred, dim=1)
    pred = pred.cpu().numpy()
    im = Image.fromarray(CITYSCAPE_PALETTE[pred[0].squeeze()])
    fn = os.path.join(output_dir, filename + '.png')
    out_dir = split(fn)[0]
    if not exists(out_dir):
        os.makedirs(out_dir)
    im.save(fn)
