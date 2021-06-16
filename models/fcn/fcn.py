"""FCN model. Modified from https://github.com/pochih/FCN-pytorch"""
import torch.nn as nn
import numpy as np


def variance_scaling(scale, in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        denominator = (fan_in + fan_out) / 2
        variance = scale / denominator

        return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale)

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FCNs(nn.Module):
    def __init__(self, pretrained_net, n_class, nf=128, fourier_scale=16):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net

        # Gaussian Fourier features embeddings.
        self.gaussian = GaussianFourierProjection(embedding_size=nf, scale=fourier_scale)
        self.lin1 = nn.Linear(2 * nf, nf * 4)
        self.lin1.weight.data = default_init()(self.lin1.weight.shape)
        nn.init.zeros_(self.lin1.bias)
        self.lin2 = nn.Linear(nf * 4, nf * 4)
        self.lin2.weight.data = default_init()(self.lin2.weight.shape)
        nn.init.zeros_(self.lin2.bias)
        self.act = nn.SiLU()

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dense1 = nn.Linear(nf * 4, 512)
        #self.dense1.weight.data = default_init()(self.dense1.weight.data.shape)
        #nn.init.zeros_(self.dense1.bias)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dense2 = nn.Linear(nf * 4, 256)
        #self.dense2.weight.data = default_init()(self.dense2.weight.data.shape)
        #nn.init.zeros_(self.dense2.bias)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dense3 = nn.Linear(nf * 4, 128)
        #self.dense3.weight.data = default_init()(self.dense3.weight.data.shape)
        #nn.init.zeros_(self.dense3.bias)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dense4 = nn.Linear(nf * 4, 64)
        #self.dense4.weight.data = default_init()(self.dense4.weight.data.shape)
        #nn.init.zeros_(self.dense4.bias)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dense5 = nn.Linear(nf * 4, 32)
        #self.dense5.weight.data = default_init()(self.dense5.weight.data.shape)
        #nn.init.zeros_(self.dense5.bias)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

<<<<<<< HEAD:models/fcn.py
    def forward(self, x, noise):
        # Gaussian Fourier features embeddings.
        temb = self.gaussian(torch.log(noise))
        temb = self.lin1(temb)
        temb = self.lin2(self.act(temb))

=======
    def forward(self, x):
>>>>>>> parent of 4fa5f5f (Change to time embedding):models/fcn/fcn.py
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score += x4                                       # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score += self.dense1(self.act(temb))[:, :, None, None]
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score += x3                                       # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score += self.dense2(self.act(temb))[:, :, None, None]
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score += x2                                       # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score += self.dense3(self.act(temb))[:, :, None, None]
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score += x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score += self.dense4(self.act(temb))[:, :, None, None]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score += self.dense5(self.act(temb))[:, :, None, None]
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

def embedd_timesteps(x, noise):
    # Combine x and noise
    noise = torch.unsqueeze(noise, -1)
    noise = torch.unsqueeze(noise, -1)
    noise = noise.expand(noise.shape[0], 1, x.shape[2])
    noise = torch.unsqueeze(noise, -1)
    noise = noise.expand(noise.shape[0], 1, x.shape[2], x.shape[3])
    x = torch.cat([x, noise], dim=1)