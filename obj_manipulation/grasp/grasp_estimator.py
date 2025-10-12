from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from obj_manipulation.grasp.graspnet import ContactGraspNet
from obj_manipulation.grasp.pc_filter import PointCloudFilter
from obj_manipulation.grasp.utils import (
    filter_point_cloud_by_depth,
    oversample_point_cloud,
    reject_median_outliers,
)
from obj_manipulation.grasp.utils.utils_pointnet import (
    get_squared_distances,
    sample_farthest_points,
)

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import FloatTensor, IntTensor


# Code adapted from https://github.com/elchun/contact_graspnet_pytorch
class GraspEstimatorBase(ABC):
    """Base class for grasp estimator modules."""
    
    def __init__(self, config: Dict[str, Any]):
        assert torch.cuda.is_available(), "No CUDA-capable devices are available."
        self.device = torch.device("cuda")
        self.config = config.copy()
    
    @abstractmethod
    def predict_grasps(
        self,
        xyz_img: ndarray,
        rgb_img: Optional[ndarray],
    ) -> Optional[Dict[str, ndarray]]:
        """Predict grasp pose for a single object in image based on xyz depth image and
        optionally rgb image data.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image data.
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
        
        Returns:
            dict or None
            - "pred_grasps": [N x 4 x 4] array of homogeneous matrices representing
            grasp poses in the camera coordinate system.
            - "pred_scores": [N x 1] array of grasp success probabilities.
            - "pred_points": [N x 3] array of contact points used for grasp predictions.
            - "pred_widths": [N x 1] array of grasp widths for grasp predictions.
        """
        pass

    @abstractmethod
    def train_mode(self) -> None:
        pass

    @abstractmethod
    def eval_mode(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass


class GraspEstimatorCGN(GraspEstimatorBase):
    """Grasp estimator module based on Contact-GraspNet from (https://doi.org/10.48550/arXiv.2103.14127)."""
    FILTER_DIST_THRESH = 1e-4

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pc_filter = PointCloudFilter()
        self.cgn = ContactGraspNet(gripper_depth=config.get("gripper_depth", 0.1034))
        self.cgn.to(self.device)

        # Store commonly used config entries to reduce dict look-ups
        self.gripper_width = config.get("gripper_width", 0.08)
        self.gripper_extra_opening = config.get("gripper_extra_opening", 0.005)
        self.n_input_points = config.get("n_input_points", 20_000)
        self.convert_cam_coords = config.get("convert_cam_coords", True)
        self.forward_passes = config.get("forward_passes", 1)
        self.min_depth = config.get("min_depth", 0.2)
        self.max_depth = config.get("max_depth", 1.8)
        self.med_thresh = config.get("med_thresh", 0.4)
        self.grasp_conf_thresh = config.get("grasp_conf_thresh", 0.15)
        self.grasp_downsample_min = config.get("grasp_downsample_min", 20)
        self.grasp_downsample_factor = config.get("grasp_downsample_factor", 0.25)
        self.filter_grasps = config.get("filter_grasps", True)
    
    def predict_grasps(
        self,
        xyz_img: ndarray,
        rgb_img: Optional[ndarray],
    ) -> Optional[Dict[str, ndarray]]:
        """Predict grasp pose for a single object in image based on xyz depth image and
        optionally rgb image data.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image data.
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
        
        Returns:
            dict or None
            - "pred_grasps": [N x 4 x 4] array of homogeneous matrices representing
            grasp poses in the camera coordinate system.
            - "pred_scores": [N x 1] array of grasp success probabilities.
            - "pred_points": [N x 3] array of contact points used for grasp predictions.
            - "pred_widths": [N x 1] array of grasp widths for grasp predictions.
        """
        # Get local point cloud around a single object using instance segmentation-based filter
        xyz_pc, xyz_object_pc = self.pc_filter.filter_point_cloud(
            xyz_img, rgb_img, n_points=self.n_input_points
        )

        # Get grasp predictions for object using model
        pred = self._predict_grasps(xyz_pc)
        
        # Filter, downsample and sort grasp predictions according to their confidence scores
        grasp_indices = self._select_grasps(pred["pred_points"], pred["pred_scores"])
        if grasp_indices.shape[0] == 0:
            return None

        # Filter grasp contacts to lie within exact object point cloud
        if self.filter_grasps:
            grasp_indices_f = self._filter_grasps(pred["pred_points"][grasp_indices], xyz_object_pc)
            if grasp_indices_f.shape[0] == 0:
                warnings.warn("No grasp predicitions fall within object point cloud.")
                return None
            grasp_indices = grasp_indices[grasp_indices_f]
            
        # Index predicitons with filtered indices and convert to NumPy
        pred["pred_grasps"] = pred["pred_grasps"][grasp_indices].cpu().numpy()
        pred["pred_scores"] = pred["pred_scores"][grasp_indices].cpu().numpy()
        pred["pred_points"] = pred["pred_points"][grasp_indices].cpu().numpy()
        pred["pred_widths"] = pred["pred_widths"][grasp_indices].cpu().numpy()
        return pred

    def train_mode(self) -> None:
        self.cgn.train()

    def eval_mode(self) -> None:
        self.cgn.eval()

    def save(self) -> None:
        path = Path(__file__).parent / "models"
        path.mkdir(parents=True, exist_ok=True)
        self.cgn.save(path)

    def load(self) -> None:
        path = Path(__file__).parent / "models"
        self.cgn.load(path)
    
    def _preprocess_pc_for_inference(
        self,
        xyz_pc: FloatTensor,
        convert_cam_coords: bool,
    ) -> FloatTensor:
        """Prepare input point cloud for grasp prediction by reshaping, filtering, centering and
        coordinate transforms, etc.
        
        Args:
            xyz_pc: [N x 3] tensor of full 3-dim point cloud.
            convert_cam_coords: Convert from OpenCV to internal camera coordinates (x left, y up, z front).
        
        Returns:
            tuple
            - xyz_pc: [1 x N x 3] tensor containing processed point cloud with added batch dimension.
            - xyz_pc_mean: [1 x 1 x 3] tensor containing the mean values of point cloud before centering. 
        """
        # Filter point cloud by depth and abs median distance
        xyz_pc = filter_point_cloud_by_depth(xyz_pc, min_d=self.min_depth, max_d=self.max_depth)
        xyz_pc = reject_median_outliers(xyz_pc, thresh=self.med_thresh)

        # Oversample to recover original number of points
        xyz_pc = oversample_point_cloud(xyz_pc, self.n_input_points)
        
        # Convert coordinates to internal coordinate system from OpenCV (x right, y down, z front) 
        if convert_cam_coords:
            xyz_pc[:, :2] *= -1
        
        # Center point cloud according to component-wise mean values
        xyz_pc_mean = torch.mean(xyz_pc, dim=0, keepdim=True)
        xyz_pc -= xyz_pc_mean

        # Add batch dimension to both point cloud and its means
        xyz_pc = xyz_pc.unsqueeze(dim=0)
        xyz_pc_mean = xyz_pc_mean.unsqueeze(dim=0)
        return xyz_pc, xyz_pc_mean
    
    @torch.no_grad
    def _predict_grasps(self, xyz_pc: FloatTensor) -> Dict[str, FloatTensor]:
        """Predict raw grasps based on input point cloud.

        Args:
            xyz_pc: [N x 3] tensor of full 3-dim point cloud.
        
        Returns:
            dict
            - "pred_grasps": [N x 4 x 4] tensor of homogeneous matrices representing
            grasp poses in the camera coordinate system.
            - "pred_scores": [N x 1] tensor of grasp success probabilities.
            - "pred_points": [N x 3] tensor of contact points used for grasp predictions.
            - "pred_widths": [N x 1] tensor of grasp widths for grasp predictions.
        """
        # Pre-process input point cloud to prepare for inference
        xyz_pc, xyz_pc_mean = self._preprocess_pc_for_inference(xyz_pc, self.convert_cam_coords)

        # Repeat point cloud to perform simulataneous forward passes on same point cloud
        xyz_pc = xyz_pc.repeat((self.forward_passes, 1, 1))  # Shape = (forward_passes, N, 3)

        # Run model inference and flatten batch dimension
        pred = self.cgn(xyz_pc)
        pred["pred_grasps"] = pred["pred_grasps"].flatten(0, 1)
        pred["pred_scores"] = pred["pred_scores"].flatten(0, 1)
        pred["pred_points"] = pred["pred_points"].flatten(0, 1)
        pred["pred_widths"] = pred["pred_widths"].flatten(0, 1)

        # Uncenter grasps and contact points
        pred["pred_grasps"][:, :3, 3] += xyz_pc_mean.squeeze(dim=0)
        pred["pred_points"] += xyz_pc_mean.squeeze(dim=0)
        
        # Add extra opening margin to gripper width
        pred["pred_widths"] = torch.minimum(
            pred["pred_widths"] + self.gripper_extra_opening, torch.tensor(self.gripper_width)
        )

        # Convert back to OpenCV coordinates (x right, y down, z front)
        if self.convert_cam_coords:
            pred["pred_grasps"][:, :2, :] *= -1
            pred["pred_points"][:, :2] *= -1
        
        return pred
    
    def _select_grasps(self, pred_points: FloatTensor, pred_scores: FloatTensor) -> IntTensor:
        """Select subset of grasps according to confidence threshold and farthest contact point downsampling. 

        1) Filters all grasps whose confidence score is less than or equal to grasp_conf_thresh.
        2) Downsample remaining grasps using farthest point sampling according to set factor and
        min number of grasps.
        3) Sort remaining grasps in descending order in terms of their confidence scores.
        
        Args:
            pred_points: [N x 3] tensor of contact points used for grasp predictions.
            pred_scores: [N x 1] tensor of grasp success probabilities.

        Returns:
            [M] int tensor containing the indices of selected contact points, sorted from highest
            to lowest confidence. 
        """
        # Filter grasps according to their confidence scores
        conf_indices = torch.nonzero(
            pred_scores[:, 0] > self.grasp_conf_thresh, as_tuple=True
        )[0]
        n_grasps_total = conf_indices.shape[0]
        if n_grasps_total == 0:
            warnings.warn("No grasp predicitions satisfy set confidence threshold contraint.")
            return conf_indices
        
        # Downsample accepted grasps using farthest points sampling
        n_grasps = int(self.grasp_downsample_factor * n_grasps_total)
        n_grasps = max(n_grasps, self.grasp_downsample_min)
        if n_grasps < n_grasps_total:
            center_indices = sample_farthest_points(
                pred_points[conf_indices].unsqueeze(dim=0), n_points=n_grasps,
            ).squeeze(dim=0)
            center_indices = conf_indices[center_indices]  # Indices in original tensor
        else:
            center_indices = conf_indices

        # Sort remaining grasps by their confidence scores
        grasp_indices = torch.argsort(pred_scores[center_indices, 0], descending=True)
        grasp_indices = center_indices[grasp_indices]
        return grasp_indices
    
    def _filter_grasps(self, pred_points: FloatTensor, obj_points: FloatTensor) -> IntTensor:
        """Filter all grasps whose contact point falls outside the object's point cloud.
        
        Args:
            pred_points: [N x 3] tensor of contact points used for grasp predictions.
            obj_points: [M x 3] tensor of xyz point cloud belonging to the selected object.
        
        Returns:
            [K] int tensor containing the indices of valid contact points.
        """
        # Get pair-wise distances between contact and object points
        dist_sqr = get_squared_distances(
            pred_points.unsqueeze(dim=0), obj_points.unsqueeze(dim=0),
        ).squeeze(dim=0)

        # Filter based on minimum distance for each contact point
        dist_sqr = torch.amin(dist_sqr, dim=1)
        grasp_indices = torch.nonzero(dist_sqr < self.FILTER_DIST_THRESH, as_tuple=True)[0]
        return grasp_indices