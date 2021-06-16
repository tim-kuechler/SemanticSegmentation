"""VGG net. Modified from https://github.com/pochih/FCN-pytorch"""
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn as nn


class VGGNet(VGG):
    def __init__(self, pretrained=True):
        super().__init__(make_layers())
        self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))

        if pretrained:
            self.load_state_dict(models.vgg16(pretrained=True).state_dict())

        del self.classifier

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

def make_layers(batch_norm=False):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)