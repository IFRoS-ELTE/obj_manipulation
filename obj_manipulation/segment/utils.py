from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal

import cv2
import numpy as np
import toml
import torch
import torchvision.transforms.functional as VF

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import FloatTensor, IntTensor


# Code adapted from https://github.com/chrisdxie/uois
def depth_map_to_xyz(
    depth: FloatTensor,
    intrinsics: FloatTensor
) -> FloatTensor:
    """
    Convert a depth map to 3D XYZ coordinates in the camera frame.
    
    Args:
        depth: [H x W] tensor containing depth values in meters.
        intrinsics: [3, 3] tensor containing the camera intrinsic matrix:
                    [[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]]
    
    Returns:
        [3 x H x W] tensor of 3D coordinates (X, Y, Z) in the camera frame.
    """
    device = depth.device
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create meshgrid of pixel coordinates
    u = torch.arange(width, device=device)
    v = torch.arange(height, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # [H, W]
    
    # Compute X, Y, Z coordinates
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    # Stack into (3, H, W)
    return torch.stack((X, Y, Z), dim=0)


def standardize_image(rgb_img: ndarray, device: torch.device) -> FloatTensor:
    """Converts input array [0, 255] to tensor [0, 1] then normalizes with fixed mean and std.
    
    Args:
        rgb_img: [H x W x 3] array of rgb image data of type uint8 from [0, 255].
        device: Torch device to move transformed image to.
    
    Returns:
        [3 x H x W] tensor of standardized rgb image data of type float.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb_img = VF.to_tensor(rgb_img).to(device=device)
    rgb_img = VF.normalize(rgb_img, mean, std)
    return rgb_img


def unstandardize_image(rgb_img: FloatTensor) -> ndarray:
    """Convert normalized tensor image back to uint8 array ranging [0, 255].
    Inverse of standardize_image()

    Args:
        rgb_img: [3 x H x W] tensor of standardized rgb image data of type float.
    
    Returns:
        [H x W x 3] array of rgb image data of type uint8 from [0, 255].
    """
    device = rgb_img.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    rgb_img = (rgb_img * std[None, None, :] + mean[None, None, :])
    rgb_img = (rgb_img * 255.).byte()
    return rgb_img.permute(1, 2, 0).cpu().numpy()


def apply_open_close_morph(init_mask: IntTensor, kernel_size: int = 9) -> IntTensor:
    """Apply successive open/close morphological operations to the initial mask.
    
    Args:
        init_mask: [H x W] int tensor of cluster labels for each pixel.
        kernel_size: Kernel size used in morphological operations.
    
    Returns:
        [H x W] int tensor of refined cluster labels for each pixel.
    """
    device = init_mask.device
    init_mask = init_mask.cpu().numpy()

    # Get unique object ids and remove background (0)
    obj_ids = np.unique(init_mask[init_mask > 0])
    
    # Apply open/close operation for each object id mask
    for obj_id in obj_ids:
        obj_mask = init_mask == obj_id  # Shape (H, W)
        obj_mask_new = cv2.morphologyEx(
            obj_mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
        obj_mask_new = cv2.morphologyEx(
            obj_mask_new.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )

        # Update cluster image
        init_mask[obj_mask] = 0  # Undo old mask
        init_mask[obj_mask_new] = obj_id  # Apply new mask
    
    return torch.from_numpy(init_mask).to(device, dtype=torch.int)


def apply_connected_comp(init_mask: IntTensor, connectivity: Literal[4, 8] = 4) -> IntTensor:
    """Apply connected component (CC) algorithm to initial mask, keeping only the largest CC.
    
    Args:
        init_mask: [H x W] int tensor of cluster labels for each pixel.
        connectivity: Type of connectivity used in determining CCs, either 4 or 8.
    
    Returns:
        [H x W] int tensor of refined cluster labels for each pixel.
    """
    device = init_mask.device
    init_mask = init_mask.cpu().numpy()

    # Get unique object ids and remove background (0)
    obj_ids = np.unique(init_mask[init_mask > 0])
    
    # Keep only the largest CC from each object id mask
    for obj_id in obj_ids:
        obj_mask = init_mask == obj_id  # Shape (H, W)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            obj_mask.astype(np.uint8), connectivity=connectivity
        )
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 0 is background
        obj_mask_new = labels == largest_label

        # Update cluster image
        init_mask[obj_mask] = 0  # Undo old mask
        init_mask[obj_mask_new] = obj_id  # Apply new mask
    
    return torch.from_numpy(init_mask).to(device, dtype=torch.int)


def load_config(path: Path) -> Dict[str, Any]:
    """Load a toml configuration file and return it as a dict."""
    config = toml.load(path)
    return config