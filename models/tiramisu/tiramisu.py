"""FCDenseNet model. Modified from https://github.com/bfortuner/pytorch_tiramisu"""
from .layers import *


class FCDenseNet103(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.data.n_channels + (1 if config.model.conditional else 0)
        n_classes = config.data.n_labels
        bottleneck_layers = 15
        growth_rate = 16
        out_chans_first_conv = 48
        self.down_blocks = down_blocks = (4,5,7,10,12)
        self.up_blocks = up_blocks = (12,10,7,5,4)
        skip_connection_channel_counts = []

        self.gaussian = GaussianFourierProjection(128, 16)
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 512)
        self.act = nn.SiLU()

        ## First Convolution ##
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv


        #####################
        # Downsampling path #
        #####################
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))


        #####################
        #     Bottleneck    #
        #####################
        self.add_module('bottleneck',Bottleneck(cur_channels_count, growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels


        #######################
        #   Upsampling path   #
        #######################
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(cur_channels_count, growth_rate, up_blocks[i], upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels


        ## Final DenseBlock ##
        self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(cur_channels_count, growth_rate, up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]


        ## Softmax + Final Conv ##
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, noise=None):
        temb = self.gaussian(torch.log(noise))
        temb = self.linear1(temb)
        temb = self.linear2(self.act(temb))

        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out, temb)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out, temb)

        out = self.finalConv(out)
        out = self.softmax(out)
        return out