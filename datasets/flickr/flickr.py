import torch.nn.functional
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from random import randint


class FLICKR(Dataset):
    """S-Flickr Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
    """
    def __init__(self, root=None, train=True):
        self.root = root
        self.n_labels = 255

        if train:
            self.data_csv = './datasets/flickr/flickr_landscapes_train_split.txt'
        else:
            self.data_csv = './datasets/flickr/flickr_landscapes_val_split.txt'

        if root is None:
            self.images_dir = '/export/data/compvis/datasets/rfw/flickr/data/'
            self.targets_dir = '/export/data/compvis/datasets/rfw/segmentation/flickr_segmentation_v2/'
        else:
            self.images_dir = os.path.join(root, 'data')
            self.targets_dir = os.path.join(root, 'segmentation')

        self.images = []
        self.targets = []
        with open(self.data_csv, 'r') as f:
            image_paths = f.read().splitlines()
            for p in image_paths:
                self.images.append(os.path.join(self.images_dir, p))
                self.targets.append(os.path.join(self.targets_dir, p.replace('.jpg', '.png')))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        #Load and resize
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        too_small = np.minimum(image.size[0], image.size[1]) < 256
        if too_small:
            scale = (256 / np.minimum(image.size[0], image.size[1])) + 0.1
            image = F.resize(image, [int(image.size[1] * scale), int(image.size[0] * scale)], interpolation=F.InterpolationMode.BICUBIC)
            target = F.resize(target, [int(image.size[1] * scale), int(image.size[0] * scale)], interpolation=F.InterpolationMode.BICUBIC)

        #Crop
        top = randint(0, image.size[1] - 256)
        left = randint(0, image.size[0] - 256)
        image = F.crop(image, top, left, 256, 256)
        target = F.crop(target, top, left, 256, 256)

        #To tensor
        image = F.to_tensor(image)
        target = F.to_tensor(target) * 255
        target = torch.squeeze(target, dim=0)
        target = torch.nn.functional.one_hot(target.long(), num_classes=self.n_labels).permute(2, 0, 1)

        return image, target

    def __len__(self):
        return len(self.images)

