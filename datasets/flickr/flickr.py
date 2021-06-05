import os
import numpy as np
import cv2
import albumentations
import torch
from PIL import Image
from torch.utils.data import Dataset


class FLICKR(Dataset):
    def __init__(self, train=True, extra_data=False, size=None, random_crop=False,
                 interpolation='bicubic', seg_v2=True, segmentation=True, data_csv=None,
                 data_dir=None, segmentation_dir=None):
        if data_csv is None:
            if train:
                if not extra_data:
                    self.data_csv = './datasets/flickr/flickr_landscapes_train_split.txt'
                else:
                    self.data_csv = './datasets/flickr/flickr_additional_landscapes_train_split.txt'
            elif not extra_data:
                self.data_csv = './datasets/flickr/flickr_landscapes_val_split.txt'
            else:
                self.data_csv = './datasets/flickr/flickr_additional_landscapes_val_split.txt'
        else:
            self.data_csv = data_csv

        self.n_labels = 182  # coco stuff labels
        if data_dir is None:
            self.data_root = '/export/data/compvis/datasets/rfw/flickr/data/'
        else:
            self.data_root = data_dir
        if segmentation_dir is None:
            self.segmentation_root = 'data/flickr_segmentation_old' if not seg_v2 else '/export/data/compvis/datasets/rfw/segmentation/flickr_segmentation_v2/'
        else:
            self.segmentation_root = segmentation_dir

        with open(self.data_csv, 'r') as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)

        # class labels
        self.synsets = [p.split('/')[0] for p in self.image_paths]
        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]
        self.num_classes = len(unique_synsets)
        # /class labels

        self.labels = {
            'class_label': self.class_labels,
            'relative_file_path_': [l for l in self.image_paths],
            'file_path_': [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            'segmentation_path_': [os.path.join(self.segmentation_root, l.replace('.jpg', '.png'))
                                    for l in self.image_paths]
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                'nearest': cv2.INTER_NEAREST,
                'bilinear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC,
                'area': cv2.INTER_AREA,
                'lanczos': cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper
        self.add_segmentation = segmentation

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example['file_path_'])
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)['image']
        if self.add_segmentation:
            segmentation = Image.open(example['segmentation_path_'])
            segmentation = np.array(segmentation).astype(np.uint8)
            if self.size is not None:
                segmentation = self.segmentation_rescaler(image=segmentation)['image']
                processed = self.preprocessor(image=image, mask=segmentation)
            else:
                processed = {'image': image, 'mask': segmentation}
            segmentation = processed['mask']
            onehot = np.eye(self.n_labels)[segmentation]
            example['segmentation'] = onehot
        else:
            if self.size is not None:
                processed = self.preprocessor(image=image)
            else:
                processed = {'image': image,}

        example['image'] = (processed['image']/127.5 - 1.0).astype(np.float32)
        img = torch.from_numpy(example['image'])
        img = img.permute(2, 0, 1)
        seg = torch.from_numpy(example['segmentation'])
        seg = seg.permute(2, 0, 1)

        return img, seg


if __name__ == '__main__':
    dset = FLICKR(train=True, size=256, data_csv='flickr_landscapes_train_split.txt')
    img, seg = dset[0]

    print(type(img))
    print(img.size())
    print(type(seg))
    print(seg.size())
