import ml_collections
import torch


def get_config(model):
    if model == 'unet':
        return get_config_unet()
    elif model == 'fcn':
        return get_config_fcn()

def get_config_unet():
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 10
    training.batch_size = 64
    training.log_freq = 100
    training.eval_freq = 5000
    training.n_eval_imgs = 10
    training.checkpoint_save_freq = 5

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'unet'
    model.n_labels = 182

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'flickr'
    data.n_channels = 3
    data.image_size = 256

    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.grad_clip = 0.1

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config

def get_config_fcn():
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 10
    training.batch_size = 64
    training.log_freq = 100
    training.eval_freq = 5000
    training.n_eval_imgs = 10
    training.checkpoint_save_freq = 5

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'fcn'

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'flickr'
    data.n_labels = 182
    data.n_channels = 3
    data.image_size = 256

    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0005
    optim.lr = 1e-10
    optim.momentum = 0.99
    optim.grad_clip = 0.

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
