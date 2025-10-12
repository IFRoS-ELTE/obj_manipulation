from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
import torchvision.transforms.functional as VF

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import FloatTensor, IntTensor


# Code adapted from https://github.com/chrisdxie/uois
def standardize_image_rgb(rgb_img: ndarray, device: torch.device) -> FloatTensor:
    """Converts input array [0, 255] to tensor [0, 1] then normalizes with fixed mean and std.
    Image is resized using interpolation to a fixed size (480, 640) expected by segmentation module.
    
    Args:
        rgb_img: [H x W x 3] array of rgb image data of type uint8 from [0, 255].
        device: Torch device to move transformed image to.
    
    Returns:
        [3 x 480 x 640] tensor of standardized rgb image data of type float.
    """
    size = (480, 640)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb_img = VF.to_tensor(rgb_img).to(device=device, dtype=torch.float)
    rgb_img = VF.resize(rgb_img, size=size)
    rgb_img = VF.normalize(rgb_img, mean, std)
    return rgb_img


def standardize_image_xyz(xyz_img: ndarray, device: torch.device) -> FloatTensor:
    """Converts input array to tensor with fixed size (480, 640) used by segmentation module.
    
    Args:
        xyz_img: [H x W x 3] array of xyz depth image.
        device: Torch device to move transformed image to.
    
    Returns:
        [3 x 480 x 640] tensor of standardized (resized) xyz depth image.
    """
    size = (480, 640)
    xyz_img = VF.to_tensor(xyz_img).to(device=device, dtype=torch.float)
    xyz_img = VF.resize(xyz_img, size=size, interpolation=VF.InterpolationMode.NEAREST)
    return xyz_img


def unstandardize_image_rgb(rgb_img: FloatTensor) -> ndarray:
    """Convert normalized tensor image back to uint8 array ranging [0, 255].
    Inverse of standardize_image_rgb() except that size is not altered.

    Args:
        rgb_img: [3 x 480 x 640] tensor of standardized rgb image data of type float.
    
    Returns:
        [480 x 640 x 3] array of rgb image data of type uint8 from [0, 255].
    """
    device = rgb_img.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    rgb_img = (rgb_img * std[:, None, None] + mean[:, None, None])
    rgb_img = (rgb_img * 255.).byte()
    return rgb_img.permute(1, 2, 0).cpu().numpy()


def unstandardize_image_xyz(xyz_img: FloatTensor) -> ndarray:
    """Convert tensor of depth image back to NumPy array.
    Inverse of standardize_image_xyz() except that size is not altered. 

    Args:
        xyz_img: [3 x 480 x 640] tensor of standardized (cropped) xyz depth image.
    
    Returns:
        [480 x 640 x 3] array of xyz depth image.
    """
    return xyz_img.permute(1, 2, 0).cpu().numpy()


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
        ).astype(np.bool_)

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


def visualize_rgb_segmap(rgb: ndarray, segmap: ndarray, bboxes: Optional[ndarray] = None) -> None:
    """Overlay rgb image with segmentation and visualize using Matplotlib. Optionally overlay
    bounding boxes if given.

    Args:
        rgb: [H x W x 3] uint8 array containing the image rgb intensities.
        segmap: [H x W] int containing the object labels of each pixel in image.
        bboxes: [N x 4] int array containing the bounding box dimensions (x_min, y_min, x_max, y_max).
    """
    if rgb is not None:
        plt.imshow(rgb)
    if segmap is not None:
        segmap_min, segmap_max = np.min(segmap), np.max(segmap)
        segmap = (segmap - segmap_min) / (segmap_max - segmap_min) * 255
        cmap = plt.get_cmap('rainbow')
        cmap.set_under(alpha=0.0)
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)
    if bboxes is not None:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            height, width = y_max - y_min, x_max - x_min
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
            )
            plt.gca().add_patch(rect)
    plt.tight_layout()
    plt.show()
