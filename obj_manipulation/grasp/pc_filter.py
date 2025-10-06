from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as VF
from torchvision.ops import masks_to_boxes

from obj_manipulation.segment import InstanceSegmentationFull
from obj_manipulation.segment.utils import (
    load_config,
    standardize_image_rgb,
    standardize_image_xyz,
)

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import BoolTensor, FloatTensor, IntTensor


class PointCloudFilter:
    """Point cloud filter to focus points around a single object and its background."""

    def __init__(self):
        # Load instance segmentation module config
        config_path = Path(__file__).parents[1] / "segment/config/config.toml"
        assert config_path.exists(), "Instance segmentation configuration is missing."
        config = load_config(config_path)

        # Initialize instance segmentation module and load its trained weights
        self.ins_seg = InstanceSegmentationFull(config)
        self.ins_seg.load()
        self.ins_seg.eval_mode()
    
    def filter_point_cloud(
        self,
        xyz_img: ndarray,
        rgb_img: Optional[ndarray],
        n_points: int = 20_000,
    ) -> np.ndarray:
        """Produce filtered PC from input xyz depth image around a single object.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image data.
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
            n_points: Desired number of points in final point cloud.
        
        Returns:
            [Np x 3] array of filtered xyz point cloud data.
        """
        # Get segmentation mask and object cluster center locations
        seg_mask, obj_centers = self._get_seg_mask_and_centers(xyz_img, rgb_img)

        # Find closest object whose bounding-box fits inside the given PC size
        obj_mask, obj_bbox = self._get_closest_valid_object(seg_mask, obj_centers, n_points)
        
        # Exit early if all objects failed check
        if obj_mask is None:
            warnings.warn("Point cloud filter failed to find suitable object.")
            return np.zeros((n_points, 3))
        
        # Compute center location of object mask in terms of row and columns
        obj_y, obj_x = torch.nonzero(obj_mask, as_tuple=True)
        center_y, center_x = obj_y.float().mean(), obj_x.float().mean()
        center_y, center_x = center_y.round().int().item(), center_x.round().int().item()

        # Create center-crop with same aspect ratio around object center
        height, width = obj_bbox[3] - obj_bbox[1], obj_bbox[2] - obj_bbox[0]
        height = torch.sqrt(n_points * height / width).int().item()
        width = int(n_points / height)
        xyz_img_crop = VF.crop(
            xyz_img,
            top=center_y - height // 2,
            left=center_x - width // 2,
            height=height,
            width=width,
        )

        # Fill point cloud with flattened xyz image data
        filtered_pc = np.zeros((n_points, 3))
        n_points_xyz = xyz_img_crop.shape[1] * xyz_img_crop.shape[2]
        filtered_pc[:n_points_xyz] = xyz_img_crop.view(3, n_points_xyz).t().cpu().numpy()
        
        return filtered_pc
    
    def _get_seg_mask_and_centers(
        self,
        xyz_img: ndarray,
        rgb_img: Optional[ndarray],
    ) -> Tuple[IntTensor, FloatTensor]:
        """Apply instance segmentation to input and return segmentation mask and cluster centers."""
        # Convert inputs to expected tensor formats
        device = self.ins_seg.device
        xyz_img_t = standardize_image_xyz(xyz_img, device=device)
        rgb_img_t = standardize_image_rgb(rgb_img, device=device) if rgb_img is not None else None

        # Apply instance segmentation module
        return self.ins_seg.segement(xyz_img_t, rgb_img_t)
    
    def _get_closest_valid_object(
        self,
        seg_mask: IntTensor,
        obj_centers: FloatTensor,
        n_points: int,
    ) -> Tuple[Optional[BoolTensor], Optional[FloatTensor]]:
        """Find closest object whose bounding-box fits inside the given PC size.
        Returns the segmentation mask and bounding-box of that object if found 
        or None if all objects fail check.
        """
        sorted_indices = torch.argsort(torch.norm(obj_centers, dim=1))
        for i in sorted_indices:
            # Get object bounding box
            obj_label = i + self.ins_seg.OBJECTS_LABEL
            obj_mask = seg_mask == obj_label
            x_min, y_min, x_max, y_max = masks_to_boxes(obj_mask.unsqueeze(0))[0]

            # Check bounding box size
            bbox_size = ((x_max - x_min) * (y_max - y_min)).int()
            if bbox_size <= n_points:
                return obj_mask, torch.stack([x_min, y_min, x_max, y_max], dim=0)
        
        return None, None