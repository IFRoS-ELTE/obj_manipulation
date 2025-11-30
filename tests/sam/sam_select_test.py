from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from obj_manipulation.grasp.utils import (
    depth_map_to_xyz,
    load_config,
)
from obj_manipulation.sam import InstanceSegmentationSAM

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
    # Load configuration and override segmentation mode to "select"
    config_path = Path(__file__).parents[2] / "obj_manipulation/sam/config/config.toml"
    assert config_path.exists()
    config = load_config(config_path)
    config["segment_mode"] = "select"

    # Initialize SAM module and load its weights
    ins_seg = InstanceSegmentationSAM(config)

    # Load test example
    path = Path(__file__).parents[1] / f"grasp/examples/{file}"
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

    # Run SAM module on example
    masks, _ = ins_seg.segment(xyz_img, rgb_img)
    assert masks.shape == (1, *rgb_img.shape[:2])
    
    # Plot input image and segmentation mask
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_img)
    show_mask(masks[0], plt.gca(), rgb=(30/255, 144/255, 255/255))
    plt.title("SAM Output Segmentation Mask", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="0.npy")
    args = parser.parse_args()

    main(file=args.file)
