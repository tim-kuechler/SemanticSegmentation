import ml_collections
import torch


def get_config(dataset):
    if dataset == 'flickr':
        return get_config_unet()
    elif dataset == 'cityscapes':
        return get_config_cityscapes()

def get_config_unet():
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 500
    training.batch_size = 16
    training.log_freq = 100
    training.eval_freq = 2500
    training.checkpoint_save_freq = 15
    training.sde = 'vesde'

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'unet'
    model.n_labels = 182

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'flickr'
    data.n_labels = 182
    data.n_channels = 3

    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.mixed_prec = False

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config

def get_config_cityscapes():
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 500
    training.batch_size = 8
    training.log_freq = 12
    training.eval_freq = 500
    training.save_pred_freq = 5
    training.full_eval_freq = 5
    training.checkpoint_save_freq = 15
    training.sde = 'vesde'

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 338
    model.num_scales = 2000
    model.bilinear = True
    model.conditional = True
    model.name = 'unet'

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'cityscapes256'
    data.n_labels = 20
    data.n_channels = 3

    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.mixed_prec = False

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config


# def get_config_cityscapes():
#     config = ml_collections.ConfigDict()
#
#     # Training
#     config.training = training = ml_collections.ConfigDict()
#     training.epochs = 500
#     training.batch_size = 8
#     training.log_freq = 25
#     training.eval_freq = 500
#     training.save_pred_freq = 5
#     training.full_eval_freq = 5
#     training.checkpoint_save_freq = 5
#     training.sde = 'vesde'
#
#     # Model
#     config.model = model = ml_collections.ConfigDict()
#     model.conditional = False
#     model.name = 'fcn'
#
#     # Data
#     config.data = data = ml_collections.ConfigDict()
#     data.dataset = 'cityscapes256'
#     data.n_labels = 20
#     data.n_channels = 3
#
#     # Optimization
#     config.optim = optim = ml_collections.ConfigDict()
#     optim.weight_decay = 1e-5
#     optim.lr = 1e-4
#     optim.momentum = 0.
#     optim.gamma = 0.5
#     optim.step_size = 30
#     optim.mixed_prec = False
#
#     config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#
#     return config
