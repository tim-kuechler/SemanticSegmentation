from torch.utils.data import DataLoader
from flickr.flickr import FLICKR


def get_dataset(config, train=True):
    batch_size = config.training.batch_size if train else config.eval.batch_size

    if config.data.dataset == 'flickr':
        dataset_train = FLICKR(config.data.n_labels, train=True, size=config.data.image_size)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataset_eval =  FLICKR(config.data.n_labels, train=False, size=config.data.image_size)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=4)

    return data_loader_train, data_loader_eval