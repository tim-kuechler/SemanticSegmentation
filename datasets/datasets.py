from torch.utils.data import DataLoader
from datasets.cityscapes256.cityscapes256 import CITYSCAPES256


def get_dataset(config):
    batch_size = config.training.batch_size

    dataset_train = CITYSCAPES256(config=config, root='./cityscapes', split='train', mode='fine',
                                  crop=config.data.crop)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataset_eval = CITYSCAPES256(config=config, root='./cityscapes', split='val', mode='fine',
                                 crop=config.data.crop)
    data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader_train, data_loader_eval
