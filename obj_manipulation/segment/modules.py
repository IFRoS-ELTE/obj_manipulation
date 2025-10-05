from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import FloatTensor


# Code adapted from https://github.com/chrisdxie/uois
class ConvBlock(nn.Module):
    """Module that combines Conv2d + Group norm + ReLU. 
    
    Note: Kernel size is assumed to be odd.
    """

    def __init__(self, in_channels, out_channels, num_groups, kernel_size=3, stride=1):
        assert kernel_size % 2 == 1, f"Expected odd kernel size, got k = {kernel_size}."
        super().__init__()
        padding = 0 if kernel_size == 1 else kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
    
    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.relu1(self.gn1(self.conv1(x)))
    

class ConvBlockx2(nn.Module):
    """Module that combines two sequential ConvBlocks.
    
    Note: Kernel size is assumed to be odd.
    """

    def __init__(self, in_channels, out_channels, num_groups, kernel_size=3, stride=1):
        assert kernel_size % 2 == 1, f"Expected odd kernel size, got k = {kernel_size}."
        super().__init__()
        self.layer1 = ConvBlock(
            in_channels, out_channels, num_groups, kernel_size=kernel_size, stride=stride
        )
        self.layer2 = ConvBlock(
            out_channels, out_channels, num_groups, kernel_size=kernel_size, stride=stride
        )
    
    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.layer2(self.layer1(x))


class ESPModule(nn.Module):
    """ESP module from ESP-Netv2 (https://doi.org/10.48550/arXiv.1811.11431).
    
    Changes:
        - First convolution is a normal 3x3 conv.
        - Uses GroupNorm instead of BatchNorm.
        - Uses ReLU instead of PReLU.
    
    Note: Kernel size is assumed to be odd and out_channels is forced to equal in_channels.
    """

    def __init__(self, in_channels, num_groups, kernel_size=3):
        assert kernel_size % 2 == 1, f"Expected odd kernel size, got k = {kernel_size}."
        super().__init__()
        nd = int(in_channels / 5)
        n1 = in_channels - 4 * nd
        
        c1_padding = 0 if kernel_size == 1 else kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels, nd, kernel_size=kernel_size, stride=1, padding=c1_padding, bias=False
        )

        self.dilated1 = nn.Conv2d(nd, n1, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.dilated2 = nn.Conv2d(nd, nd, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.dilated4 = nn.Conv2d(nd, nd, kernel_size=3, stride=1, padding=4, bias=False, dilation=4)
        self.dilated8 = nn.Conv2d(nd, nd, kernel_size=3, stride=1, padding=8, bias=False, dilation=8)
        self.dilated16 = nn.Conv2d(nd, nd, kernel_size=3, stride=1, padding=16, bias=False, dilation=16)

        self.gn = nn.GroupNorm(num_groups, in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: FloatTensor) -> FloatTensor:
        # Reduce number of channels
        out = self.conv1(x)

        # Apply dilated convs to the same output
        d1 = self.dilated1(out)
        d2 = self.dilated2(out)
        d4 = self.dilated4(out)
        d8 = self.dilated8(out)
        d16 = self.dilated16(out)

        # Hierarchically fuse features from dilated convs
        # Note: d1 cannot be fused because n1 != nd
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # Merge all features from d1 and added dilated convs
        combined = torch.cat([d1, add1, add2, add3, add4], dim=1)

        # Apply residual connection with input
        combined = x + combined
        return self.relu(self.gn(combined))


class UpsampleConcatConvBlock(nn.Module):
    """Module that peforms the following operations:
    
    Operations:
        - Apply ConvBlock to reduce in_channels by a factor of 2.
        - Upsample with bilinear sampling by a factor of 2.
        - Concat with another input with in_channels // 2 (coming from encoder).
        - Apply another ConvBlock to concatenated features.
    """
    def __init__(self, in_channels, out_channels, num_groups):
        super().__init__()
        self.channel_reduction_layer = ConvBlock(in_channels, in_channels // 2, num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_gn_relu = ConvBlock(in_channels, out_channels, num_groups)
    
    def forward(self, dec_x: FloatTensor, enc_x: FloatTensor) -> FloatTensor:
        out = self.channel_reduction_layer(dec_x)
        out = self.upsample(out)
        out = torch.cat([out, enc_x], dim=1)
        return self.conv_gn_relu(out)