import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from obj_manipulation.grasp import GraspEstimatorCGN
from obj_manipulation.grasp.utils import (
    depth_map_to_xyz,
    load_config,
)
from obj_manipulation.grasp.utils.utils_visualization import visualize_grasps


def main(file: str, vis_width: bool):
    # Load configuration
    config_path = Path(__file__).parents[2] / "obj_manipulation/grasp/config/config.toml"
    assert config_path.exists()
    config = load_config(config_path)

    # Initialize grasp estimator and load its pre-trained weights 
    grasp_est = GraspEstimatorCGN(config)
    grasp_est.load()
    grasp_est.eval_mode()

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
    
    # Predict grasps for a single object
    start = time.time()
    pred = grasp_est.predict_grasps(xyz_img.copy(), rgb_img.copy())
    delta = time.time() - start
    print(f"Grasp prediciton took {delta:.2f} seconds")
    if pred is not None:
        depth_mask = np.logical_and(depth > grasp_est.min_depth, depth < grasp_est.max_depth)
        pred_widths = pred["pred_widths"] if vis_width \
            else np.full_like(pred["pred_widths"], grasp_est.gripper_width)
        visualize_grasps(
            xyz_img[depth_mask],
            rgb_img[depth_mask],
            [pred["pred_grasps"]],
            [pred["pred_scores"]],
            [pred_widths],
            config["gripper_depth"],
        )
    else:
        print("Grasp estimator failed to find any valid grasps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="0.npy")
    parser.add_argument("-v", "--vis_width", action="store_true", default=False)
    args = parser.parse_args()

    main(file=args.file, vis_width=args.vis_width)