import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 10
    training.batch_size = 64
    training.log_freq = 100
    training.eval_freq = 250
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
