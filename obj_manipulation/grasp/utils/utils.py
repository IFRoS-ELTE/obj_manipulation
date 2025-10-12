from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import toml
import torch

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import FloatTensor


def depth_map_to_xyz(depth: ndarray, intrinsics: ndarray) -> ndarray:
    """Convert a depth map to 3D XYZ coordinates in the camera frame.
    
    Args:
        depth: [H x W] array containing depth values in meters.
        intrinsics: [3, 3] array containing the camera intrinsic matrix:
                    [[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]]

    Returns:
        [H x W x 3] array of 3D coordinates (X, Y, Z) in the camera frame.
    """
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create meshgrid of pixel coordinates
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v, indexing='xy')  # [H, W]

    # Compute X, Y, Z coordinates
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    
    # Stack into (H, W, 3)
    return np.stack((X, Y, Z), axis=2)


def filter_point_cloud_by_depth(
    xyz_pc: FloatTensor,
    min_d: float = 0.2,
    max_d: float = 1.8,
) -> FloatTensor:
    """Filter given point cloud based on minium and maximum depth limits.
    
    Args:
        xyz_pc: [N x 3] tensor of full 3-dim point cloud.
        min_d: Minimum allowable depth in the point cloud meters.
        max_d: Maximum allowable depth in the point cloud in meters.
    
    Returns:
        [M x 3] tensor containing the valid points in the input point cloud only.
    """
    mask = torch.logical_and(xyz_pc[:, 2] > min_d, xyz_pc[:, 2] < max_d)
    return xyz_pc[mask]


def reject_median_outliers(xyz_pc: FloatTensor, thresh: float = 0.4) -> FloatTensor:
    """Reject point outliers with median absolute distance greater than set threshold.

    Args:
        xyz_pc: [N x 3] tensor of full 3-dim point cloud.
        thresh: Absolute median distance threshold.

    Returns:
        [M x 3] tensor containing the valid points in the input point cloud only.
    """
    abs_med_dist = torch.abs(xyz_pc - torch.median(xyz_pc, dim=0, keepdim=True)[0])
    abs_med_dist = torch.sum(abs_med_dist, dim=1)
    return xyz_pc[abs_med_dist < thresh]


def oversample_point_cloud(xyz_pc: FloatTensor, n_points: int) -> FloatTensor:
    """If point cloud has less points than n_points, then oversample it to reach n_points.
    
    Args:
        xyz_pc: [N x 3] tensor of full 3-dim point cloud.
        n_points: Desired number of points in each point cloud.
      
    Returns:
        [n_points x 3] tensor containing oversampled point cloud with n_points.
    """
    device = xyz_pc.device
    required = n_points - xyz_pc.shape[0]
    if required > 0:
        index = torch.randint(0, xyz_pc.shape[0], size=(required,), device=device)
        xyz_pc = torch.cat([xyz_pc, xyz_pc[index]], dim=0)

    return xyz_pc


def load_config(path: Path) -> Dict[str, Any]:
    """Load a toml configuration file and return it as a dict."""
    config = toml.load(path)
    return config
