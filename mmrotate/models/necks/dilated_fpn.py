# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)
from torch.nn import BatchNorm2d
import torch.nn.functional as F
from ..builder import ROTATED_NECKS


class Bottleneck(nn.Module):
    """Bottleneck block for DilatedEncoder used in `YOLOF.

    <https://arxiv.org/abs/2103.09460>`.

    The Bottleneck contains three ConvLayers and one residual connection.

    Args:
        in_channels (int): The number of input channels.
        mid_channels (int): The number of middle output channels.
        dilation (int): Dilation rate.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvModule(
            in_channels, mid_channels, 1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            mid_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg)
        self.conv3 = ConvModule(
            mid_channels, in_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


@ROTATED_NECKS.register_module()
class DFPN(nn.Module):
    """Dilated Encoder for YOLOF <https://arxiv.org/abs/2103.09460>`.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
              which are 1x1 conv + 3x3 conv
        - the dilated residual block

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        block_mid_channels (int): The number of middle block output channels
        num_residual_blocks (int): The number of residual blocks.
        block_dilations (list): The list of residual blocks dilation.
    """

    def __init__(self, in_channels, out_channels, block_mid_channels,
                 num_residual_blocks, block_dilations):
        super(DFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations
        self._init_layers()

    def _init_layers(self):
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.dilated_encoder_blocks = nn.ModuleList()
        self.lateral_norm = BatchNorm2d(self.out_channels)
        self.fpn_norm = BatchNorm2d(self.out_channels)
        for i in range(len(self.in_channels)):
            lateral_conv = nn.Conv2d(
                self.in_channels[i], self.out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            dilation = self.block_dilations[i]
            self.dilated_encoder_blocks.append(
                Bottleneck(
                    self.out_channels,
                    self.block_mid_channels,
                    dilation=dilation))
        for i in range(3, 5):
            lateral_conv = nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            dilation = self.block_dilations[i]
            self.dilated_encoder_blocks.append(
                Bottleneck(
                    self.out_channels,
                    self.block_mid_channels,
                    dilation=dilation))

    def init_weights(self):
        caffe2_xavier_init(self.lateral_convs)
        caffe2_xavier_init(self.fpn_convs)
        for m in [self.lateral_norm, self.fpn_norm]:
            constant_init(m, 1)
        for block in self.dilated_encoder_blocks:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    print("normal_init")
                    normal_init(m, mean=0, std=0.01)
                if is_norm(m):
                    constant_init(m, 1)

    def forward(self, feature):
        """Forward function."""
        assert len(feature) == len(self.in_channels)
        laterals = [
            self.lateral_convs[i](f)
            for i, f in enumerate(feature)
        ]
        for i in range(len(laterals), len(self.lateral_convs)):
            laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))
        
        laterals_norms = [
            self.lateral_norm(lateral)
            for i, lateral in enumerate(laterals)
        ]

        fpn_norms = [
            self.fpn_norm(self.fpn_convs[i](laterals_norm))
            for i, laterals_norm in enumerate(laterals_norms)
        ]

        outs = [
            self.dilated_encoder_blocks[i](fpn_norm)
            for i, fpn_norm in enumerate(fpn_norms)
        ]
        # out = self.lateral_norm(self.lateral_conv(feature[-1]))
        # out = self.fpn_norm(self.fpn_conv(out))
        # return self.dilated_encoder_blocks(out),
        return tuple(outs)
