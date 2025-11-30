from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from obj_manipulation.grasp import PointCloudFilterSAM
from obj_manipulation.grasp.utils import depth_map_to_xyz, load_config
from obj_manipulation.sam import InstanceSegmentationSAM
from obj_manipulation.segment.utils import visualize_rgb_segmap

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def show_mask(mask: NDArray, ax: Axes, rgb: Optional[Tuple[float, float, float]]) -> None:
    """Visualize mask image using the given color onto the axes object."""
    rgb = np.random.random(3) if rgb is None else np.array(rgb)
    color = np.concatenate([rgb, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main(file: str):
    # Initialize point cloud filter using the SAM segmentation module
    pc_filter = PointCloudFilterSAM()

    # Replace SAM module with custom config
    config_path = Path(__file__).parents[2] / "obj_manipulation/sam/config/config.toml"
    assert config_path.exists()
    config = load_config(config_path)
    config["segment_mode"] = "select"
    pc_filter.ins_seg = InstanceSegmentationSAM(config)

    # Load test example
    path = Path(__file__).parent / f"examples/{file}"
    assert path.exists(), f"Test file {file} does not exist at {path}."
    print(f"Loading test example: {path}")
    data = np.load(path, allow_pickle=True).item()

    # Get RGB image and XYZ (directly or from depth and camera intrinsics)
    assert "rgb" in data, f"RGB image data is not available in {file}."
    rgb_img = np.array(data["rgb"], dtype=np.uint8)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    if "xyz" in data:
        print("\tusing stored XYZ image")
        xyz_img = np.array(data["xyz"])
    else:
        assert "depth" in data, f"Depth image data not available in {file}."
        assert "K" in data, f"Calibration data not available in {file}."
        print("\textracting XYZ image from depth map and camera intrinsics")
        depth, intrinsics = np.array(data["depth"]), np.array(data["K"])
        xyz_img = depth_map_to_xyz(depth, intrinsics)
    
    # Resize to same input resolution as provided by RGB-D camera
    rgb_img = cv2.resize(rgb_img, (640, 480))
    xyz_img = cv2.resize(xyz_img, (640, 480))

    # Test instance segmentation module separatley
    masks, centers = pc_filter.ins_seg.segement(xyz_img, rgb_img)
    # Plot input image and segmentation mask
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_img)
    show_mask(masks[0], plt.gca(), rgb=(30/255, 144/255, 255/255))
    plt.title("SAM Output Segmentation Mask", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Test object selection and bounding box extraction functionalities
    masks = torch.from_numpy(masks).to(pc_filter.ins_seg.device)
    centers = torch.from_numpy(centers).to(pc_filter.ins_seg.device)
    obj_mask, obj_bbox = pc_filter._get_best_valid_object(masks, centers, 20_000)
    assert obj_mask is not None
    top, left, height, width = pc_filter._get_point_cloud_bbox(obj_mask, obj_bbox, 20_000)
    pc_bbox = np.array([left, top, left + width, top + height], dtype=np.int_) 
    visualize_rgb_segmap(
        rgb_img,
        obj_mask.int().cpu().numpy(),
        bboxes=pc_bbox.reshape(1, 4),
    )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="0.npy")
    args = parser.parse_args()

    main(file=args.file)
