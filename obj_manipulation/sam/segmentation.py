from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from obj_manipulation.sam.utils import collect_points

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InstanceSegmentationSAM:
    """Instance segmentation module using the Segment-Anything Model (SAM)."""
    CHECKPOINTS = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    def __init__(self, config: Dict[str, Any]):
        assert torch.cuda.is_available(), "No CUDA-capable devices are available."
        assert config["model_type"] in ["vit_h", "vit_l", "vit_b"]
        self.device = torch.device("cuda")
        model_type = config["model_type"]
        model = sam_model_registry[model_type](
            checkpoint=str(Path(__file__).parent / f"models/{self.CHECKPOINTS[model_type]}")
        )
        model = model.to(self.device)
        if config["segment_mode"] == "select":
            self.sam = SamPredictor(model)
            self.segment_fn = self._segment_select
        elif config["segment_mode"] == "auto":
            self.sam = SamAutomaticMaskGenerator(model, **config["auto"])
            self.segment_fn = self._segment_auto
        else:
            raise ValueError(
                f"Unrecognized mode: {config['segment_mode']}. Expected one of 'select' or 'auto'"
            )
    
    def segment(
        self,
        xyz_img: NDArray[np.float32],
        rgb_img: NDArray[np.uint8],
    ) -> Tuple[NDArray, NDArray]:
        """Apply instance segmentation using SAM on input data.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image.
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
        
        Returns:
            tuple
            - masks: [N x H x W] array of masks for each detected object.
            - centers: [N x 3] array containing the center locations of each object.
        """
        # Apply segmentation function to get masks
        masks = self.segment_fn(rgb_img)

        # Calculate the center locations of each object cluster
        centers = self._get_mask_centers(xyz_img, masks)
        return masks, centers

    def _segment_select(self, rgb_img: NDArray[np.uint8]) -> NDArray[np.bool_]:
        """Instance segmentation using the selective predictor.
        
        Prompts the user for positive and negative point labels then uses them for segmentation.
        
        Args:
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
        
        Returns:
            [1 x H x W] array of mask for selected object.
        """
        # Encode image using SAM
        self.sam.set_image(rgb_img, image_format="RGB")

        # Get point coordinates and labels
        points, labels = collect_points(rgb_img)
        
        # Get mask predictions
        masks, scores, _ = self.sam.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        masks = masks[None, np.argmax(scores)]
        return masks

    def _segment_auto(self, rgb_img: NDArray[np.uint8]) -> NDArray[np.bool_]:
        """Instance segmentation using the auto predictor.
        
        Detects objects across the whole image and returns all their masks.
        
        Args:
            rgb_img: [H x W x 3] array of rgb image data of type uint8.
        
        Returns:
            [N x H x W] array of masks for all N detected objects.
        """
        # Get all mask predictions
        outputs = self.sam.generate(rgb_img)

        # Keep masks only
        masks = np.stack([out["segmentation"] for out in outputs])
        return masks
    
    def _get_mask_centers(
        self,
        xyz_img: NDArray[np.float32],
        masks: NDArray[np.bool_],
    ) -> NDArray:
        """Calculate and return the centroids of all object clusters.
        
        Args:
            xyz_img: [H x W x 3] array of xyz depth image.
            masks: [N x H x W] array of masks for all N detected objects.
        
        Returns:
            [N x 3] array containing the center locations of each object.
        """
        # Calculate centroids for each object cluster
        n_obj = len(masks)
        mask_centers = np.zeros((n_obj, 3))
        for i, mask in enumerate(masks):
            mask_centers[i] = xyz_img[mask].mean(axis=0)
        return mask_centers
