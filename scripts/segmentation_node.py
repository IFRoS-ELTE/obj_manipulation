#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from obj_manipulation.segment.utils import (
    standardize_image_rgb,
    standardize_image_xyz,
)
from obj_manipulation.grasp.utils import (
    depth_map_to_xyz,
    load_config,
)
from obj_manipulation.segment import InstanceSegmentationFull
from obj_manipulation.sam import InstanceSegmentationSAM


class InstanceSegmentationNode:
    def __init__(self):
        rospy.init_node('instance_segmentation', anonymous=True)

        self.bridge = CvBridge()
        self.rgb_image = None
        self.xyz_image = None
        self.camera_intrinsics = None

        # -------- Load node parameters --------
        self.alpha = rospy.get_param("/instance_segmentation/alpha", 0.6)
        self.max_trials = rospy.get_param("/instance_segmentation/max_trials", 4)
        seg_model = rospy.get_param("/instance_segmentation/seg_model", "sam")
        assert seg_model in ["sam", "uois"]

        # -------- Load configuration and model --------
        if seg_model == "sam":
            config_path = Path(__file__).parents[1] / "obj_manipulation/sam/config/config.toml"
            assert config_path.exists()
            config = load_config(config_path)
            self.ins_seg = InstanceSegmentationSAM(config)
            self.get_seg_mask = self._get_seg_mask_sam
        else:  # uois
            config_path = Path(__file__).parents[1] / "obj_manipulation/segment/config/config.toml"
            assert config_path.exists()
            config = load_config(config_path)
            self.ins_seg = InstanceSegmentationFull(config)
            self.ins_seg.load()
            self.ins_seg.eval_mode()
            self.get_seg_mask = self._get_seg_mask_uois

        # -------- Subscribers --------
        rgb_topic_name = "/camera/color/image_raw"
        depth_topic_name = "/camera/aligned_depth_to_color/image_raw"
        cam_info_topic_name = "/camera/aligned_depth_to_color/camera_info"

        self.rgb_sub = rospy.Subscriber(rgb_topic_name, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(
            depth_topic_name, Image, self.depth_callback, queue_size=1
        )
        self.cam_info_sub = rospy.Subscriber(
            cam_info_topic_name, CameraInfo, self.cam_info_callback, queue_size=1
        )

        # -------- Publishers --------
        seg_mask_topic_name = "/instance_segmentation/seg_mask"
        self.seg_mask_pub = rospy.Publisher(seg_mask_topic_name, Image, queue_size=1)

    @property
    def ready_to_publish(self) -> bool:
        ready = all([
            self.rgb_image is not None,
            self.xyz_image is not None,
        ])
        return ready
    
    @staticmethod
    def _get_color_mask(seg_mask: NDArray[np.int32]) -> NDArray[np.uint8]:
        n_objs = seg_mask.max() - 1
        cm = plt.get_cmap('gist_rainbow')
        colors = [cm(1. * i/n_objs) for i in range(n_objs)]

        color_mask = np.zeros(seg_mask.shape + (3,), dtype=np.uint8)
        for i in range(n_objs):
            color_mask[seg_mask == (i + 2), :] = np.array(colors[i][:3]) * 255
        return color_mask

    @staticmethod
    def _masks_to_seg_mask(masks: NDArray[np.bool_]) -> NDArray[np.in32]:
        assert masks.ndim == 3 and len(masks) > 0
        seg_mask = np.zeros(masks.shape[1:], dtype=np.int32)
        for i, mask in enumerate(masks):
            seg_mask[mask] = i + 2  # (+ 2) to match UOIS object labels
        return seg_mask

    def cam_info_callback(self, msg: CameraInfo) -> None:
        K = np.array(msg.K).reshape(3, 3)
        self.camera_intrinsics = K

    def rgb_callback(self, msg: Image) -> None:
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def depth_callback(self, msg: Image) -> None:
        if self.camera_intrinsics is not None:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding == "16UC1":
                depth_image = depth_image.astype(np.float32) / 1000.0
            self.xyz_image = depth_map_to_xyz(depth_image, self.camera_intrinsics)

    def publish_seg_mask(self, seg_mask: NDArray[np.int32]):
        # Get color mask and blend it with input RGB image
        color_mask = self._get_color_mask(seg_mask)
        seg_image = np.where(
            color_mask,
            self.alpha * color_mask + (1 - self.alpha) * self.rgb_image,
            self.rgb_image,
        ).astype(np.uint8)

        # Convert from NumPy to ROS Image
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "depth_optical_frame"
        seg_image_msg = self.bridge.cv2_to_imgmsg(seg_image, encoding="rgb8", header=header)
        
        # Publish message
        self.seg_mask_pub.publish(seg_image_msg)

    def _get_seg_mask_uois(self) -> Optional[NDArray[np.int32]]:
        device = self.ins_seg.device
        # Transform inputs
        rgb = standardize_image_rgb(self.rgb_image, device=device)
        xyz = standardize_image_xyz(self.xyz_image, device=device)
        # Predict segmentation mask
        for _ in range(self.max_trials):
            seg_mask, _ = self.ins_seg.segment(xyz, rgb_img=rgb)
            max_label = seg_mask.amax()
            if max_label > 1:
                return seg_mask.cpu().numpy()
        return None
    
    def _get_seg_mask_sam(self) -> Optional[NDArray[np.int32]]:
        # Predict object masks
        for _ in range(self.max_trials):
            masks, _ = self.ins_seg.segment(self.xyz_image, self.rgb_image)
            if len(masks) > 0:
                seg_mask = self._masks_to_seg_mask(masks)
                return seg_mask
        return None


def main():
    node = InstanceSegmentationNode()
    rate = rospy.Rate(hz=5)  # Publish at a max rate of 5 Hz
    while not rospy.is_shutdown():
        if node.ready_to_publish:
            seg_mask = node.get_seg_mask()
            if seg_mask is not None:
                node.publish_seg_mask(seg_mask)
            else:
                rospy.logwarn("No objects detected in image")
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Grasp Estimation Node terminated.")
