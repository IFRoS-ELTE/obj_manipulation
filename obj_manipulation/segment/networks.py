from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

from obj_manipulation.segment.unet import (
    UNetDecoder,
    UNetEncoder,
    UNetESPDecoder,
    UNetESPEncoder,
)

if TYPE_CHECKING:
    from torch import FloatTensor


# Code adapted from https://github.com/chrisdxie/uois
class DepthSeedingNetwork(nn.Module):
    """DSN full network from (https://doi.org/10.48550/arXiv.2007.08073)."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.encoder = UNetESPEncoder(in_channels, feature_dim)
        self.decoder = UNetESPDecoder(feature_dim)
        # Foreground head for converting features to logits for 3 classes: 
        # background (0), table (1), objects (2).
        self.fg_module = nn.Conv2d(
            feature_dim, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False
        )
        # Center offsets head for converting features to 3D offsets towards the center of each cluster
        self.cd_module = nn.Conv2d(
            feature_dim, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False
        )
    
    def forward(self, xyz_img: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """Forward pass through full DSN network.
        
        Args:
            xyz_img: [N x 3 x H x W] tensor of batched xyz depth images.
        
        Returns:
            tuple
            - fg_logits: [N x 3 x H x W] tensor of background/table/objects logits.
            - center_offsets: [N x 3 x H x W] tensor of center offset predictions.
        """
        features = self.decoder(*self.encoder(xyz_img))
        fg_logits = self.fg_module(features)
        center_offsets = self.cd_module(features)
        return fg_logits, center_offsets
    
    def save(self, path: Path) -> None:
        assert path.exists(), f"Path {path} does not exist."
        assert path.is_dir(), f"Path {path} is not a directory."
        checkpoint = {"model": self.state_dict()}
        torch.save(checkpoint, path / "DSN.pth")
    
    def load(self, path: Path) -> None:
        path /= "DSN.pth"
        assert path.exists(), f"Path {path} does not exist."
        state_dict = torch.load(path, weights_only=True)["model"]
        state_dict = {
            ('.'.join(key.split('.')[1:]) if key.split('.')[0] == "module" else key):value
            for key, value in state_dict.items()
        }
        self.load_state_dict(state_dict)


class RegionRefinementNetwork(nn.Module):
    """RRN full network from (https://doi.org/10.48550/arXiv.2007.08073)."""

    def __init__(self, in_channels: int = 4, feature_dim: int = 64):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, feature_dim)
        self.decoder = UNetDecoder(feature_dim)
        # Foreground head for converting features to foreground logits
        self.fg_module = nn.Conv2d(
            feature_dim, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False
        )
    
    def forward(self, rgb_img: FloatTensor, init_mask: FloatTensor) -> FloatTensor:
        """Forward pass through full RRN network.
        
        Args:
            rgb_img: [N x 3 x H x W] tensor of batched rgb images.
            init_mask: [N x 1 x H x W] tensor of initial foreground predicted mask.
        
        Returns:
            [N x 1 x H x W] tensor of foreground logits.
        """
        enc_input = torch.cat([rgb_img, init_mask], dim=1)  # Shape: (N, 4, H, W)
        features = self.decoder(*self.encoder(enc_input))
        return self.fg_module(features)
    
    def save(self, path: Path) -> None:
        assert path.exists(), f"Path {path} does not exist."
        assert path.is_dir(), f"Path {path} is not a directory."
        checkpoint = {"model": self.state_dict()}
        torch.save(checkpoint, path / "RRN.pth")
    
    def load(self, path: Path) -> None:
        path /= "RRN.pth"
        assert path.exists(), f"Path {path} does not exist."
        state_dict = torch.load(path, weights_only=True)["model"]
        state_dict = {
            ('.'.join(key.split('.')[1:]) if key.split('.')[0] == "module" else key):value
            for key, value in state_dict.items()
        }
        self.load_state_dict(state_dict)