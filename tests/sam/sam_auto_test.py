from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sam_select_test import show_mask

from obj_manipulation.grasp.utils import (
    depth_map_to_xyz,
    load_config,
)
from obj_manipulation.sam import InstanceSegmentationSAM


def main(file: str):
    # Seed NumPy RNG for consistent colors for segmentation masks
    np.random.seed(seed=10)

    # Load configuration and override segmentation mode to "auto"
    config_path = Path(__file__).parents[2] / "obj_manipulation/sam/config/config.toml"
    assert config_path.exists()
    config = load_config(config_path)
    config["segment_mode"] = "auto"

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
    if len(masks) == 0:
        print("SAM failed to find any objects in the scene.")
    else:
        print(f"Found {len(masks)} objects in the scene.")
    assert masks.shape == (len(masks), *rgb_img.shape[:2])

    # Plot input image and segmentation mask
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_img)
    for mask in masks:
        show_mask(mask, plt.gca(), rgb=None)
    plt.title("SAM Output Segmentation Mask", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="0.npy")
    args = parser.parse_args()

    main(file=args.file)
