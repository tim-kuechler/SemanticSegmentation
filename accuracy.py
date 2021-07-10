import argparse
import os
from PIL import Image
import numpy as np


def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        'dataset1',
        help='Work folder',
    )
    parser.add_argument(
        'dataset2',
        help='Work folder',
    )

    return parser

def pixel_acc(pred, target):
    """ Calculate pixel accuracy of prediction in comparison to original map

    :param pred:
    :param target:
    :return: pixel accuracy
    """
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    dir1 = args.dataset1
    dir2 = args.dataset2

    acc = []
    for file1, file2 in zip(os.listdir(dir1), os.listdir(dir2)):
        print(file1)
        print(file2)
        map1 = Image.open(os.path.join(dir1, file1))
        map2 = Image.open(os.path.join(dir2, file2))

        map1 = np.array(map1)
        map2 = np.array(map2)

        acc.append(pixel_acc(map1, map2))

    print('Accuracy: ', np.nanmean(acc))