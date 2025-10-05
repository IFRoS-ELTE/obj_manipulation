from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch.nn as nn
import torch.nn.functional as F

from obj_manipulation.segment.modules import (
    ConvBlock,
    ConvBlockx2,
    ESPModule,
    UpsampleConcatConvBlock,
)

if TYPE_CHECKING:
    from torch import FloatTensor


# Code adapted from https://github.com/chrisdxie/uois
class UNetEncoder(nn.Module):
    """UNet Encoder Network."""

    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layer1 = ConvBlockx2(in_channels, feature_dim, feature_dim)
        self.layer2 = ConvBlockx2(feature_dim, feature_dim * 2, feature_dim)
        self.layer3 = ConvBlock(feature_dim * 2, feature_dim * 4, feature_dim)
        self.layer4 = ConvBlock(feature_dim * 4, feature_dim * 8, feature_dim)
        self.last_layer = ConvBlock(feature_dim * 8, feature_dim * 16, feature_dim)

    def forward(self, images: FloatTensor) -> Tuple[FloatTensor, List[FloatTensor]]:
        x1 = self.layer1(images)
        mp_x1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x2 = self.layer2(mp_x1)
        mp_x2 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = self.layer3(mp_x2)
        mp_x3 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x4 = self.layer4(mp_x3)
        mp_x4 = F.max_pool2d(x4, kernel_size=2, stride=2)
        x5 = self.last_layer(mp_x4)
        return x5, [x1, x2, x3, x4]


class UNetDecoder(nn.Module):
    """UNet Decoder Network."""

    def __init__(self, feature_dim):
        super().__init__()
        # Fusion layer
        self.fuse_layer = ConvBlock(feature_dim * 16, feature_dim * 16, feature_dim, kernel_size=1)

        # Decoding
        self.layer1 = UpsampleConcatConvBlock(feature_dim * 16, feature_dim * 8, feature_dim)
        self.layer2 = UpsampleConcatConvBlock(feature_dim * 8, feature_dim * 4, feature_dim)
        self.layer3 = UpsampleConcatConvBlock(feature_dim * 4, feature_dim * 2, feature_dim)
        self.layer4 = UpsampleConcatConvBlock(feature_dim * 2, feature_dim, feature_dim)

        # Final layer
        self.layer5 = ConvBlock(feature_dim, feature_dim, feature_dim)

        # Allows feature to become < 0 since ReLU is used and other convs don't have biases
        self.last_conv = nn.Conv2d(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=True
        )
    
    def forward(self, enc_outs: FloatTensor, enc_inter_outs: List[FloatTensor]) -> FloatTensor:
        # Apply fusion layer to the final encoder output
        out = self.fuse_layer(enc_outs)

        # Go through decoder layers for upsampling
        out = self.layer1(out, enc_inter_outs[3])
        out = self.layer2(out, enc_inter_outs[2])
        out = self.layer3(out, enc_inter_outs[1])
        out = self.layer4(out, enc_inter_outs[0])

        # Apply final conv layers
        out = self.layer5(out)
        out = self.last_conv(out)
        return out


class UNetESPEncoder(nn.Module):
    """UNet ESP-based Encoder Network."""

    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layer1 = ConvBlockx2(in_channels, feature_dim, feature_dim)
        self.layer2 = ConvBlockx2(feature_dim, feature_dim * 2, feature_dim)
        self.layer3a = ConvBlock(feature_dim * 2, feature_dim * 4, feature_dim)
        self.layer3b = ESPModule(feature_dim * 4, feature_dim * 4, feature_dim)
        self.layer4a = ConvBlock(feature_dim * 4, feature_dim * 8, feature_dim)
        self.layer4b = ESPModule(feature_dim * 8, feature_dim * 8, feature_dim)
        self.last_layer = ConvBlock(feature_dim * 8, feature_dim * 16, feature_dim)

    def forward(self, images: FloatTensor) -> Tuple[FloatTensor, List[FloatTensor]]:
        x1 = self.layer1(images)
        mp_x1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x2 = self.layer2(mp_x1)
        mp_x2 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = self.layer3b(self.layer3a(mp_x2))
        mp_x3 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x4 = self.layer4b(self.layer4a(mp_x3))
        mp_x4 = F.max_pool2d(x4, kernel_size=2, stride=2)
        x5 = self.last_layer(mp_x4)
        return x5, [x1, x2, x3, x4]


class UNetESPDecoder(nn.Module):
    """UNet ESP-based Decoder Network."""

    def __init__(self, feature_dim):
        super().__init__()
        # Fusion layer
        self.fuse_layer = ESPModule(feature_dim * 16, feature_dim * 16, feature_dim, kernel_size=1)

        # Decoding
        self.layer1 = UpsampleConcatConvBlock(feature_dim * 16, feature_dim * 8, feature_dim)
        self.layer2 = UpsampleConcatConvBlock(feature_dim * 8, feature_dim * 4, feature_dim)
        self.layer3 = UpsampleConcatConvBlock(feature_dim * 4, feature_dim * 2, feature_dim)
        self.layer4 = UpsampleConcatConvBlock(feature_dim * 2, feature_dim, feature_dim)

        # Final layer
        self.layer5 = ConvBlock(feature_dim, feature_dim, feature_dim)

        # Allows feature to become < 0 since ReLU is used and other convs don't have biases
        self.last_conv = nn.Conv2d(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=True
        )
    
    def forward(self, enc_outs: FloatTensor, enc_inter_outs: List[FloatTensor]) -> FloatTensor:
        # Apply fusion layer to the final encoder output
        out = self.fuse_layer(enc_outs)

        # Go through decoder layers for upsampling
        out = self.layer1(out, enc_inter_outs[3])
        out = self.layer2(out, enc_inter_outs[2])
        out = self.layer3(out, enc_inter_outs[1])
        out = self.layer4(out, enc_inter_outs[0])

        # Apply final conv layers
        out = self.layer5(out)
        out = self.last_conv(out)
        return out