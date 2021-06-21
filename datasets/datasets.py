from torch.utils.data import DataLoader
from datasets.flickr.flickr import FLICKR
from datasets.cityscapes256.cityscapes256 import CITYSCAPES256


def get_dataset(config):
    batch_size = config.training.batch_size

    if config.data.dataset == 'flickr':
        dataset_train = FLICKR(train=True)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval = FLICKR(train=False)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)
    if config.data.dataset == 'cityscapes256':
        dataset_train = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='train', mode='fine',
                                      crop=config.data.crop)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='val', mode='fine',
                                     crop=config.data.crop)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader_train, data_loader_eval