from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torchvision.transforms.functional as VF
from torchvision.ops import masks_to_boxes

from obj_manipulation.grasp.utils.utils_pointnet import sample_farthest_points
from obj_manipulation.sam.segmentation import InstanceSegmentationSAM
from obj_manipulation.segment.segmentation import InstanceSegmentationFull
from obj_manipulation.segment.utils import (
    load_config,
    standardize_image_rgb,
    standardize_image_xyz,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import BoolTensor, FloatTensor, IntTensor


class PointCloudFilter(ABC):
    """Abstract class for point cloud filter.
    
    The point cloud filter is used to focus points around a single object and its background.
    """

    @abstractmethod
    def filter_point_cloud(
        self,
        xyz_img: NDArray,
        rgb_img: NDArray,
        n_points: int = 20_000,
    ) -> Tuple[FloatTensor, FloatTensor, BoolTensor]:
        """Produce filtered PC from input xyz depth image around a single object.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image data.
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
            n_points: Desired number of points in final point cloud.
        
        Returns:
            tuple
            - filtered_pc: [Np x 3] tensor of filtered xyz point cloud data.
            - object_pc: [M x 3] tensor of xyz point cloud belonging to object only.
            - obj_mask: [H x W] tensor of boolean mask for selected object. 
        """
        pass

    @staticmethod
    def _get_point_cloud_bbox(
        obj_mask: BoolTensor,
        obj_bbox: FloatTensor,
        n_points: int,
    ) -> Tuple[int, int, int, int]:
        """Compute top-left corner position and size of object bounding box.
        
        Args:
            obj_mask: [H x W] tensor containing boolean object mask.
            obj_bbox: [4] tensor of (x_min, y_min, x_max, y_max) bounding box coordinates.
            n_points: Desired number of points in final point cloud.
        
        Returns:
            tuple
            - top: Row index of top-left corner for object bounding box.
            - left: Column index of top-left corner for object bounding box.
            - height: Height for object bounding box.
            - width: Width for object bounding box.
        """
        img_height, img_width = obj_mask.shape
        
        # Compute center location of object mask in terms of row and columns
        obj_y, obj_x = torch.nonzero(obj_mask, as_tuple=True)
        center_y, center_x = obj_y.float().mean(), obj_x.float().mean()
        center_y, center_x = center_y.round().int(), center_x.round().int()

        # Create center-crop with same aspect ratio around object center
        height, width = obj_bbox[3] - obj_bbox[1], obj_bbox[2] - obj_bbox[0]
        height = torch.sqrt(n_points * height / width).int()
        height = torch.minimum(height, 2 * (img_height - center_y))
        width = (n_points / height).int()
        width = torch.minimum(width, 2 * (img_width - center_x))

        # Compute top-left corner position
        top = center_y - height // 2
        left = center_x - width // 2
        return top.item(), left.item(), height.item(), width.item()


class PointCloudFilterUOIS(PointCloudFilter):
    """Point cloud filter using the Unseen Object Instance Segmentation (UOIS) module.
    
    The UOIS module was introduced in the paper titled
    `Unseen Object Instance Segmentation for Robotic Environments` available at
    <https://arxiv.org/abs/2007.08073>.
    """

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
        xyz_img: NDArray,
        rgb_img: Optional[NDArray],
        n_points: int = 20_000,
    ) -> Tuple[FloatTensor, FloatTensor, BoolTensor]:
        """Produce filtered PC from input xyz depth image around a single object.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image data.
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
            n_points: Desired number of points in final point cloud.
        
        Returns:
            tuple
            - filtered_pc: [Np x 3] tensor of filtered xyz point cloud data.
            - object_pc: [M x 3] tensor of xyz point cloud belonging to object only.
            - obj_mask: [H x W] tensor of boolean mask for selected object. 
        """
        # Convert inputs to expected tensor formats
        device = self.ins_seg.device
        xyz_img_t = standardize_image_xyz(xyz_img, device=device)
        rgb_img_t = standardize_image_rgb(rgb_img, device=device) if rgb_img is not None else None

        # Get segmentation mask and object cluster center locations
        seg_mask, obj_centers = self.ins_seg.segment(xyz_img_t, rgb_img_t)

        # Find best object whose bounding-box is the closest to the given PC size
        obj_mask, obj_bbox = self._get_best_valid_object(seg_mask, obj_centers, n_points)
        n_points_obj = (obj_bbox[3] - obj_bbox[1]) * (obj_bbox[2] - obj_bbox[0])
        
        if n_points_obj <= n_points:
            # Compute closest estimate to bounding box that encompasses the chosen object
            # and has n_points within it then extract it out of xyz depth image
            top, left, height, width = self._get_point_cloud_bbox(obj_mask, obj_bbox, n_points)
            xyz_img_crop = VF.crop(
                xyz_img_t,
                top=top,
                left=left,
                height=height,
                width=width,
            )

            # Fill point cloud with flattened xyz image data
            filtered_pc = torch.zeros((n_points, 3), device=device)
            n_points_xyz = xyz_img_crop.shape[1] * xyz_img_crop.shape[2]
            filtered_pc[:n_points_xyz] = xyz_img_crop.flatten(1, 2).t()
            
            # Get object point cloud from object mask
            object_pc = xyz_img_t[:, obj_mask].t()
        else:
            # Downsample object point cloud to get required number of points
            xyz_img_t = xyz_img_t[:, obj_mask].t()
            point_idxs = sample_farthest_points(
                xyz_img_t.unsqueeze(0),
                n_points=n_points,
            ).squeeze(0)
            filtered_pc = xyz_img_t[point_idxs]
            object_pc = filtered_pc.clone()

        return filtered_pc, object_pc, obj_mask
    
    def _get_best_valid_object(
        self,
        seg_mask: IntTensor,
        obj_centers: FloatTensor,
        n_points: int,
    ) -> Tuple[BoolTensor, FloatTensor]:
        """Find best object whose bounding-box is the closest to the given PC size. Best is defined
        as an object whose centroid is neither very close nor very far from the camera.
        
        Returns the segmentation mask and bounding-box of that object if found 
        or None if all objects fail check.

        Args:
            seg_mask: [H x W] int tensor containing object labels for each pixel.
            obj_centers: [N x 3] tensor containing the center locations of each object's cluster.
            n_points: Desired number of points in final point cloud.
        
        Returns:
            tuple
            - obj_mask: [H x W] tensor containing boolean object mask or None.
            - obj_bbox: [4] tensor of (x_min, y_min, x_max, y_max) bounding box coordinates.
        """
        delta_points_min, obj_mask_min = torch.inf, None
        obj_dists = torch.norm(obj_centers, dim=1)
        sorted_indices = torch.argsort(torch.abs(obj_dists - obj_dists.mean()))
        for i in sorted_indices:
            # Get object bounding box
            obj_label = i + self.ins_seg.OBJECTS_LABEL
            obj_mask = seg_mask == obj_label
            x_min, y_min, x_max, y_max = masks_to_boxes(obj_mask.unsqueeze(0))[0]

            # Check bounding box size
            bbox_size = ((x_max - x_min) * (y_max - y_min)).int()
            delta_points = bbox_size - n_points
            if delta_points <= 0:
                return obj_mask, torch.stack([x_min, y_min, x_max, y_max], dim=0)
            elif delta_points < delta_points_min:
                delta_points_min = delta_points
                obj_mask_min = obj_mask.clone()

        # Return mask of closest object to required point cloud size
        obj_bbox = masks_to_boxes(obj_mask_min.unsqueeze(0))[0]
        return obj_mask_min, obj_bbox


class PointCloudFilterSAM(PointCloudFilter):
    """Point cloud filter using the Segment Anything Model (SAM).
    
    For more details about SAM, check out the official repo from Meta at
    <https://github.com/facebookresearch/segment-anything>
    """

    def __init__(self):
        # Load the SAM module config
        config_path = Path(__file__).parents[1] / "sam/config/config.toml"
        assert config_path.exists(), "SAM configuration is missing."
        config = load_config(config_path)

        # Initialize the SAM module and load its trained weights
        self.ins_seg = InstanceSegmentationSAM(config)
    
    def filter_point_cloud(
        self,
        xyz_img: NDArray,
        rgb_img: NDArray,
        n_points: int = 20_000,
    ) -> Tuple[FloatTensor, FloatTensor, BoolTensor]:
        """Produce filtered PC from input xyz depth image around a single object.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image data.
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
            n_points: Desired number of points in final point cloud.
        
        Returns:
            tuple
            - filtered_pc: [Np x 3] tensor of filtered xyz point cloud data.
            - object_pc: [M x 3] tensor of xyz point cloud belonging to object only.
            - obj_mask: [H x W] tensor of boolean mask for selected object. 
        """
        device = self.ins_seg.device
        xyz_img_t = VF.to_tensor(xyz_img).to(device=device, dtype=torch.float32)

        # Get segmentation mask and object cluster center locations
        masks, centers = self.ins_seg.segment(xyz_img, rgb_img)
        masks = torch.from_numpy(masks).to(device)
        centers = torch.from_numpy(centers).to(device)

        # Find best object whose bounding-box is the closest to the given PC size
        obj_mask, obj_bbox = self._get_best_valid_object(masks, centers, n_points)
        n_points_obj = (obj_bbox[3] - obj_bbox[1]) * (obj_bbox[2] - obj_bbox[0])
        
        if n_points_obj <= n_points:
            # Compute closest estimate to bounding box that encompasses the chosen object
            # and has n_points within it then extract it out of xyz depth image
            top, left, height, width = self._get_point_cloud_bbox(obj_mask, obj_bbox, n_points)
            xyz_img_crop = VF.crop(
                xyz_img_t,
                top=top,
                left=left,
                height=height,
                width=width,
            )

            # Fill point cloud with flattened xyz image data
            filtered_pc = torch.zeros((n_points, 3), device=device)
            n_points_xyz = xyz_img_crop.shape[1] * xyz_img_crop.shape[2]
            filtered_pc[:n_points_xyz] = xyz_img_crop.flatten(1, 2).t()
            
            # Get object point cloud from object mask
            object_pc = xyz_img_t[:, obj_mask].t()
        else:
            # Downsample object point cloud to get required number of points
            xyz_img_t = xyz_img_t[:, obj_mask].t()
            point_idxs = sample_farthest_points(
                xyz_img_t.unsqueeze(0),
                n_points=n_points,
            ).squeeze(0)
            filtered_pc = xyz_img_t[point_idxs]
            object_pc = filtered_pc.clone()

        return filtered_pc, object_pc, obj_mask
    
    def _get_best_valid_object(
        self,
        masks: BoolTensor,
        centers: FloatTensor,
        n_points: int,
    ) -> Tuple[BoolTensor, FloatTensor]:
        """Find best object whose bounding-box is the closest to the given PC size. Best is defined
        as an object whose centroid is neither very close nor very far from the camera.
        
        Returns the segmentation mask and bounding-box of that object if found 
        or None if all objects fail check.

        Args:
            masks: [N x H x W] tensor of masks for all N detected objects.
            centers: [N x 3] tensor containing the center locations of each object's cluster.
            n_points: Desired number of points in final point cloud.
        
        Returns:
            tuple
            - obj_mask: [H x W] tensor containing boolean object mask or None.
            - obj_bbox: [4] tensor of (x_min, y_min, x_max, y_max) bounding box coordinates.
        """
        delta_points_min, idx_obj_min = torch.inf, -1
        dists = torch.norm(centers, dim=1)
        sorted_indices = torch.argsort(torch.abs(dists - dists.mean()))
        for i in sorted_indices:
            # Get object bounding box
            x_min, y_min, x_max, y_max = masks_to_boxes(masks[i].unsqueeze(0))[0]

            # Check bounding box size
            bbox_size = ((x_max - x_min) * (y_max - y_min)).int()
            delta_points = bbox_size - n_points
            if delta_points <= 0:
                return masks[i], torch.stack([x_min, y_min, x_max, y_max], dim=0)
            elif delta_points < delta_points_min:
                delta_points_min = delta_points
                idx_obj_min = i
        
        # Return mask of closest object to required point cloud size
        obj_bbox = masks_to_boxes(masks[idx_obj_min].unsqueeze(0))[0]
        return masks[idx_obj_min], obj_bbox
