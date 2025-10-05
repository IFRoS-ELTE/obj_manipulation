from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

from obj_manipulation.segment.cluster import GaussianMeanShift
from obj_manipulation.segment.networks import (
    DepthSeedingNetwork,
    RegionRefinementNetwork,
)
from obj_manipulation.segment.utils import (
    apply_connected_comp,
    apply_open_close_morph,
)

if TYPE_CHECKING:
    from torch import FloatTensor, IntTensor


# Code adapted from https://github.com/chrisdxie/uois
class InstanceSegmentationBase(ABC):
    """Base abstract class for instance segementation modules."""
    OBJECTS_LABEL = 2  # Background (0), Table (1)

    def __init__(self, config: Dict[str, Any]):
        assert torch.cuda.is_available(), "No CUDA-capable devices are available."
        self.device = torch.device("cuda")
        self.config = config.copy()
    
    @abstractmethod
    def segement(self, **args) -> Tuple[IntTensor, Optional[FloatTensor]]:
        """Apply instance segmentation on input data."""
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


class InstanceSegmentationDSN(InstanceSegmentationBase):
    """Instance segementation module using DSN and Gaussian mean-shift only."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dsn = DepthSeedingNetwork(in_channels=3, feature_dim=config["feature_dim"])
        self.gms = GaussianMeanShift(**config["gms"])
        self.dsn.to(self.device)

        # Store commonly used config entries to reduce dict look-ups
        self.min_pixels_thresh = self.config["min_pixels_thresh"]
    
    @torch.no_grad
    def segement(self, xyz_img: FloatTensor) -> Tuple[IntTensor, Optional[FloatTensor]]:
        """Apply instance segmentation using DSN and GMS algorithm on input data.
        
        Args:
            xyz_img: [3 x H x W] tensor of xyz depth image.
        
        Returns:
            tuple
            - cluster_img: [H x W] int tensor of cluster labels for each pixel.
            - None.
        """
        # Apply DSN model
        fg_logits, center_offsets = self.dsn(xyz_img.unsqueeze(0))
        fg_logits, center_offsets = fg_logits[0], center_offsets[0]

        # Get foreground mask
        fg_mask = torch.argmax(fg_logits, dim=1)  # Shape (H, W)

        # Run GMS clustering algorithm
        cluster_img = self._cluster(
            xyz_img, center_offsets, fg_mask == self.OBJECTS_LABEL  # Filter background and table
        )
        return cluster_img, None

    def train_mode(self) -> None:
        self.dsn.train()
    
    def eval_mode(self) -> None:
        self.dsn.eval()
    
    def save(self) -> None:
        path = Path(__file__).parent / "models"
        path.mkdir(parents=True, exist_ok=True)
        self.dsn.save(path)
    
    def load(self) -> None:
        path = Path(__file__).parent / "models"
        self.dsn.load(path)
    
    def _cluster(
        self,
        xyz_img: FloatTensor,
        offsets: FloatTensor,
        fg_mask: IntTensor,
    ) -> IntTensor:
        """Run Gaussian mean shift algorithm on predicted 3D centers.
        
        Args:
            xyz_img: [3 x H x W] tensor of xyz depth image.
            offsets: [3 x H x W] tensor of predicted center offsets.
            fg_mask: [H x W] mask tensor for filtering backgroud pixels (0).
        
        Returns:
            [H x W] int tensor of cluster labels for each pixel.
        """
        clustered_img = torch.zeros_like(fg_mask)
        if torch.sum(fg_mask) == 0:  # No foreground pixels to cluster
            return clustered_img

        # Run Gaussian mean-shift on predicted centers of foreground pixels
        predicted_centers = xyz_img + offsets
        predicted_centers = predicted_centers.permute(1, 2, 0)  # Shape: (H, W, 3)
        cluster_labels = self.gms.mean_shift(predicted_centers[fg_mask])

        # Reshape clustered labels back to [H x W]
        clustered_img[fg_mask] = cluster_labels + self.OBJECTS_LABEL

        # Get cluster labels
        uniq_labels, uniq_counts = torch.unique(cluster_labels, return_counts=True)
        uniq_labels += self.OBJECTS_LABEL

        # Filter small clusters
        valid_labels_mask = uniq_counts >= self.min_pixels_thresh
        uniq_labels = uniq_labels[valid_labels_mask]
        if uniq_labels.numel() == 0:  # All clusters were filtered out
            return torch.zeros_like(clustered_img)
        
        # Relabel clustered_img and compute cluster centers
        new_cl_img = torch.zeros_like(clustered_img)
        for i, label in enumerate(uniq_labels):
            new_cl_img[clustered_img == label] = i + self.OBJECTS_LABEL
        
        return new_cl_img
    

class InstanceSegmentationRRN(InstanceSegmentationBase):
    """Instance segementation module using the RRN."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rrn = RegionRefinementNetwork(in_channels=4, feature_dim=config["feature_dim"])
        self.rrn.to(self.device)

        # Store commonly used config entries to reduce dict look-ups
        self.crop_size = tuple(config.get("crop_size", (224, 224)))
        self.crop_pad_perc = config.get("crop_pad_perc", 0.25)
        self.foreground_thresh = config.get("foreground_thresh", 0.5)
    
    @torch.no_grad
    def segement(
        self,
        rgb_img: FloatTensor,
        init_mask: FloatTensor,
    ) -> Tuple[IntTensor, Optional[FloatTensor]]:
        """Apply instance segmentation using the RRN.
        
        Args:
            rgb_img: [3 x H x W] tensor of rgb image data.
            init_mask: [H x W] tensor of initial cluster mask.
        
        Returns:
            tuple
            - mask: [H x W] int tensor of refined cluster mask.
            - None
        """
        # Get local crops around each object
        obj_rgbs, obj_masks, obj_bboxes = self._rrn_preprocess(rgb_img, init_mask)
        
        # Run all object crops through RRN model
        n_objs = obj_rgbs.shape[0]
        step_size = min(128, n_objs)
        for b in range(0, n_objs, step_size):
            obj_logits = self.rrn(
                obj_rgbs[b: b + step_size], obj_masks[b: b + step_size].unsqueeze(1)
            )[:, 0]
            obj_probs = torch.sigmoid(obj_logits)
            obj_masks[b: b + step_size] = obj_probs > self.foreground_thresh
            
        # Combine refined object masks back into a single cluster mask
        mask = self._rrn_postprocess(rgb_img, obj_masks, obj_bboxes)
        
        return mask, None
        
    def train_mode(self) -> None:
        self.rrn.train()
    
    def eval_mode(self) -> None:
        self.rrn.eval()
    
    def save(self) -> None:
        path = Path(__file__).parent / "models"
        path.mkdir(parents=True, exist_ok=True)
        self.rrn.save(path)
    
    def load(self) -> None:
        path = Path(__file__).parent / "models"
        self.rrn.load(path)
    
    def _rrn_preprocess(
        self,
        rgb_img: FloatTensor,
        init_mask: IntTensor,
    ) -> Tuple[FloatTensor, FloatTensor, IntTensor]:
        """Pad, crop and resize input rgb and masks around each individual object to prepare for RRN.
    
        Args:
            rgb_img: [3 x H x W] tensor of rgb image data.
            init_mask: [H x W] int tensor of cluster labels for each pixel.
        
        Returns:
            tuple
            - rgb_imgs: [N x 3 x size[0] x size[1]] tensor of rgb crops around objects.
            - init_masks: [N x size[0] x size[1]] tensor of mask crops around objects.
            - bboxes: [N, 4] tensor of bounding box dimensions (x_min, y_min, x_max, y_max).
        """
        device = rgb_img.device
        height, width = rgb_img.shape[1:]

        # Get unique object clusters
        fg_mask = init_mask >= self.OBJECTS_LABEL
        mask_ids = torch.unique(init_mask[fg_mask])
        n_objects = mask_ids.shape[0]

        # Initialize outputs 
        rgb_imgs = torch.empty((n_objects, 3, self.crop_size[0], self.crop_size[1]), device=device)
        init_masks = torch.empty((n_objects, self.crop_size[0], self.crop_size[1]), device=device)
        bboxes = torch.empty((n_objects, 4), dtype=torch.int, device=device)
        
        # Loop over objects to create local crops
        for index, mask_id in enumerate(mask_ids):
            # Determine object bbox
            obj_mask = init_mask == mask_id
            x_min, y_min, x_max, y_max = masks_to_boxes(obj_mask.unsqueeze(0))[0]

            # Determine padding and bbox boundaries
            x_padding = torch.round((x_max - x_min) * self.crop_pad_perc).int()
            y_padding = torch.round((y_max - y_min) * self.crop_pad_perc).int()
            x_min = torch.maximum(x_min - x_padding, 0)
            y_min = torch.maximum(y_min - y_padding, 0)
            x_max = torch.minimum(x_max + x_padding, width - 1)
            y_max = torch.minimum(y_max + y_padding, height - 1)
            bboxes[index] = torch.stack([x_min, y_min, x_max, y_max], dim=0).int()

            # Crop rgb image and object mask
            rgb_crop = rgb_img[:, y_min: y_max + 1, x_min: x_max + 1]
            mask_crop = obj_mask[y_min: y_max + 1, x_min: x_max + 1]

            # Resize to new size and store
            rgb_imgs[index] = F.interpolate(
                rgb_crop.unsqueeze(0), self.crop_size, mode="bilinear"
            )[0]
            init_masks[index] = F.interpolate(
                mask_crop.unsqueeze(0).unsqueeze(0).float(), self.crop_size, mode="nearest"
            )[0, 0]
        
        return rgb_imgs, init_masks, bboxes
    
    def _rrn_postprocess(
        self,
        rgb_img: FloatTensor,
        refined_crops: FloatTensor,
        bboxes: IntTensor,
    ) -> IntTensor:
        """Resize refined masks back to original size and combine them into one image mask.
    
        Args:
            rgb_img: [3 x H x W] tensor of rgb image data.
            refined_crops: [N x H_resize, W_resize] tensor of refined mask crops.
            bboxes: [N, 4] tensor of bounding box dimensions (x_min, y_min, x_max, y_max).
        
        Returns:
            [H x W] int tensor of refined image cluster mask.
        """
        refined_mask = torch.zeros_like(rgb_img[0], dtype=torch.int)
        for index, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            orig_H = y_max - y_min + 1
            orig_W = x_max - x_min + 1
            mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
            resized_mask = F.interpolate(mask, (orig_H, orig_W), mode="nearest")[0, 0]
            refined_mask[y_min: y_max + 1, x_min: x_max + 1][resized_mask.bool()] = index + self.OBJECTS_LABEL

        return refined_mask


class InstanceSegmentationFull(InstanceSegmentationBase):
    """Full instance segmentation module from (https://doi.org/10.48550/arXiv.2007.08073)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dsn_mod = InstanceSegmentationDSN(config["dsn"])
        self.rrn_mod = InstanceSegmentationRRN(config["rrn"]) if config["use_rrn"] else None

        # Store commonly used config entries to reduce dict look-ups
        self.use_rrn = config.get("use_rrn", False)
        self.use_open_close_morphology = config.get("use_open_close_morphology", False) 
        self.oc_morph_kernel_size = config.get("oc_morph_kernel_size", 9) 
        self.use_largest_connected_component = config.get("use_largest_connected_component", False) 
        self.lcc_connectivity = config.get("lcc_connectivity", 4)
        self.filter_small_clusters = config.get("filter_small_clusters", False)
        self.min_pixels_thresh = config.get("min_pixels_thresh", 500)

    @torch.no_grad
    def segement(
        self,
        xyz_img: FloatTensor,
        rgb_img: Optional[FloatTensor] = None,
    ) -> Tuple[IntTensor, Optional[FloatTensor]]:
        """Apply instance segmentation using DSN and GMS algorithm on input data.
        
        Args:
            xyz_img: [3 x H x W] tensor of xyz depth image.
            rgb_img: Optional [3 x H x W] tensor of rgb image data if using RRN.
        
        Returns:
            tuple
            - cluster_img: [H x W] int tensor of cluster labels for each pixel.
            - cluster_centers: [N x 3] tensor containing the center locations of each cluster.
        """
        # Apply DSN module-based segmentation
        cluster_img, _ = self.dsn_mod.segement(xyz_img)

        # Process object masks with open/close morphology
        if self.use_open_close_morphology:
            cluster_img = apply_open_close_morph(cluster_img, kernel_size=self.oc_morph_kernel_size)
        
        # Process object masks with largest connected component filter
        if self.use_largest_connected_component:
            cluster_img = apply_connected_comp(cluster_img, connectivity=self.lcc_connectivity)
        
        # Process object masks with RRN
        if self.use_rrn and rgb_img is not None:
            cluster_img, _ = self.rrn_mod.segement(rgb_img, cluster_img)
        
        # Filter small clusters
        if self.filter_small_clusters:
            cluster_img = self._filter_small_clusters(cluster_img, min_thresh=self.min_pixels_thresh)
        
        # Calculate object cluster centers based on final cluster image
        cluster_centers = self._get_cluster_centers(xyz_img, cluster_img)
        
        return cluster_img, cluster_centers 

    def train_mode(self) -> None:
        self.dsn_mod.train_mode()
        if self.use_rrn:
            self.rrn_mod.train_mode()

    def eval_mode(self) -> None:
        self.dsn_mod.eval_mode()
        if self.use_rrn:
            self.rrn_mod.eval_mode()

    def save(self) -> None:
        self.dsn_mod.save()
        if self.use_rrn:
            self.rrn_mod.save()

    def load(self) -> None:
        self.dsn_mod.load()
        if self.use_rrn:
            self.rrn_mod.load()
    
    def _filter_small_clusters(self, cluster_img: IntTensor, min_thresh: int) -> IntTensor:
        """"Filter small clusters with number of pixels less than the minimum threshold.
        
        Args:
            cluster_img: [H x W] int tensor of cluster labels for each pixel.
            min_thresh: Minimum allowed number of pixels in a cluster.
        
        Returns:
            [H x W] int tensor of filtered cluster labels for each pixel.
        """
        # Get unique cluster labels and remove background
        fg_mask = cluster_img >= self.OBJECTS_LABEL
        uniq_labels, uniq_counts = torch.unique(cluster_img[fg_mask], return_counts=True)

        # Filter invalid labels and re-label pixels
        valid_labels_mask = uniq_counts >= min_thresh
        uniq_labels = uniq_labels[valid_labels_mask]
        if uniq_labels.numel() == 0:  # All clusters were filtered out
            return torch.zeros_like(cluster_img)
        
        # Relabel clustered_img and compute cluster centers
        new_cluster_img = torch.zeros_like(cluster_img)
        for i, label in enumerate(uniq_labels):
            new_cluster_img[cluster_img == label] = i + self.OBJECTS_LABEL
        
        return new_cluster_img
    
    def _get_cluster_centers(self, xyz_img: FloatTensor, cluster_img: IntTensor) -> FloatTensor:
        """Calculate and return the centroids of all object clusters.
        
        Args:
            xyz_img: [3 x H x W] tensor of xyz depth image.
            cluster_img: [H x W] int tensor of cluster labels for each pixel.
        
        Returns:
            [N_Objects, 3] tensor of object center locations.
        """
        # Find object cluster labels
        fg_mask = cluster_img >= self.OBJECTS_LABEL
        uniq_labels = torch.unique(cluster_img[fg_mask])

        # Calculate centroids for each object cluster
        n_obj = uniq_labels.shape[0]
        cluster_centers = torch.zeros((n_obj, 3), device=xyz_img.device)
        for index, label in enumerate(uniq_labels):
            cluster_centers[index] = xyz_img[:, cluster_img == label].mean(dim=1)
        
        return cluster_centers 