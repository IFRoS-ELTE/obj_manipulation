import argparse
from pathlib import Path

import cv2
import numpy as np

from obj_manipulation.grasp import PointCloudFilter
from obj_manipulation.grasp.utils import depth_map_to_xyz
from obj_manipulation.segment.utils import (
    standardize_image_rgb,
    standardize_image_xyz,
    unstandardize_image_rgb,
    visualize_rgb_segmap,
)


def main(file: str):
    # Initialize point cloud filter
    pc_filter = PointCloudFilter()

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

    # Test instance segmentation module separatley
    device = pc_filter.ins_seg.device
    xyz_img_t = standardize_image_xyz(xyz_img, device=device)
    rgb_img_t = standardize_image_rgb(rgb_img, device=device)
    segmap, obj_centers = pc_filter.ins_seg.segement(xyz_img_t, rgb_img_t)
    visualize_rgb_segmap(unstandardize_image_rgb(rgb_img_t), segmap.cpu().numpy())

    # Test object selection and bounding box extraction functionalities
    obj_mask, obj_bbox = pc_filter._get_best_valid_object(segmap, obj_centers, 20_000)
    assert obj_mask is not None
    top, left, height, width = pc_filter._get_point_cloud_bbox(obj_mask, obj_bbox, 20_000)
    pc_bbox = np.array([left, top, left + width, top + height], dtype=np.int_) 
    visualize_rgb_segmap(
        unstandardize_image_rgb(rgb_img_t),
        obj_mask.int().cpu().numpy(),
        bboxes=pc_bbox.reshape(1, 4),
    )

    # Test full point cloud filtering functionality
    n_points = height * width
    xyz_pc, _ = pc_filter.filter_point_cloud(xyz_img, rgb_img, n_points=20_000)
    xyz_pc = xyz_pc[:n_points].cpu().numpy()
    xyz_pc = xyz_pc.reshape(height, width, 3)
    print(f"Extacted {n_points} points from object bounding box")
    visualize_rgb_segmap(xyz_pc[..., 2], None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="0.npy")
    args = parser.parse_args()

    main(file=args.file)