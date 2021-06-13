from torch.utils.data import DataLoader
from flickr.flickr import FLICKR
from cityscapes256.cityscapes256 import CITYSCAPES256


def get_dataset(config):
    batch_size = config.training.batch_size

    if config.data.dataset == 'flickr':
        dataset_train = FLICKR(config.data.n_labels, train=True)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval =  FLICKR(config.data.n_labels, train=False)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)
    if config.data.dataset == 'cityscapes256':
        dataset_train = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='train', mode='fine')
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval = CITYSCAPES256(root='/export/data/tkuechle/datasets/cityscapes_full', split='test', mode='fine')
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader_train, data_loader_eval