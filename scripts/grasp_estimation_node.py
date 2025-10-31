#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import torch
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from obj_manipulation.grasp import GraspEstimatorCGN
from obj_manipulation.grasp.utils import depth_map_to_xyz, load_config


class GraspEstimationNode:
    def __init__(self):
        rospy.init_node('grasp_estimation_node', anonymous=True)
        rospy.loginfo("Initializing Grasp Estimation Node...")

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.camera_intrinsics = None

        # -------- Grasp Saving Configuration --------
        self.saved_grasps = []
        self.max_saved = 10
        self.output_path = "/catkin_ws/src/obj_manipulation/grasp_poses.txt"

        # Clear old file on startup
        try:
            open(self.output_path, "w").close()
            rospy.loginfo(f"Cleared previous grasp log: {self.output_path}")
        except Exception as e:
            rospy.logwarn(f"Could not clear grasp file: {e}")

        # -------- Load configuration and model --------
        config_path = "/catkin_ws/src/obj_manipulation/obj_manipulation/grasp/config/config.toml"
        rospy.loginfo(f"Loading configuration from: {config_path}")
        self.config = load_config(config_path)

        rospy.loginfo("Initializing grasp estimator...")
        self.grasp_est = GraspEstimatorCGN(self.config)
        self.grasp_est.load()
        self.grasp_est.eval_mode()
        rospy.loginfo("Grasp estimator model loaded and ready.")

        # -------- Subscribers --------
        rospy.loginfo("Subscribing to camera topics...")
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.cam_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.cam_info_callback)

        # -------- Publishers --------
        self.pose_pub = rospy.Publisher("/grasp_estimation_node/grasp_pose", PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/grasp_estimation_node/marker", Marker, queue_size=10)
        rospy.loginfo("Publishers created for grasp pose and RViz marker.")

        rospy.loginfo("Initialization complete. Waiting for camera data...")

    # ------------------ Callbacks ------------------

    def cam_info_callback(self, msg):
        K = np.array(msg.K).reshape(3, 3)
        self.camera_intrinsics = K
        rospy.loginfo_once("Received camera intrinsics.")

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        rospy.loginfo_once("Received first RGB image.")

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32) / 1000.0  # convert mm → meters
        self.depth_image = depth
        rospy.loginfo_once("Received first depth image.")
        self.try_predict_grasp()

    # ------------------ Main Prediction ------------------

    def try_predict_grasp(self):
        if self.rgb_image is None:
            rospy.logwarn("RGB image not received yet.")
            return
        if self.depth_image is None:
            rospy.logwarn("Depth image not received yet.")
            return
        if self.camera_intrinsics is None:
            rospy.logwarn("Camera intrinsics not received yet.")
            return

        rospy.loginfo("Starting grasp prediction pipeline...")

        # Step 1: Convert depth → XYZ point cloud
        rospy.loginfo("Converting depth map to XYZ point cloud...")
        xyz_img = depth_map_to_xyz(self.depth_image, self.camera_intrinsics)
        rospy.loginfo("Depth map converted to XYZ successfully.")

        # Step 2: Run model
        rospy.loginfo("Running grasp estimation model...")
        try:
            with torch.no_grad():
                result = self.grasp_est.predict_grasps(xyz_img, self.rgb_image)
            rospy.loginfo("Grasp estimation model finished inference.")
        except Exception as e:
            rospy.logerr(f"Error during model inference: {e}")
            return

        # Step 3: Extract best grasp
        if result is None or "pred_grasps" not in result or len(result["pred_grasps"]) == 0:
            rospy.logwarn("No valid grasps predicted.")
            return

        best_pose = result["pred_grasps"][0]
        rospy.loginfo("Best grasp pose extracted (index 0).")

        # Step 4: Publish Pose + Marker
        self.publish_grasp_pose(best_pose)
        self.publish_marker(best_pose)
        rospy.loginfo("Grasp pose and marker published successfully.")

        # Step 5: Save first 10 grasps to file
        self.save_grasp_pose(best_pose)

    # ------------------ Grasp Saving ------------------

    def save_grasp_pose(self, grasp_matrix):
        """Save up to the first N grasp poses into a text file."""
        try:
            if len(self.saved_grasps) < self.max_saved:
                self.saved_grasps.append(grasp_matrix)
                np.savetxt(
                    self.output_path,
                    np.array(self.saved_grasps).reshape(-1, 4, 4),
                    fmt="%.6f"
                )
                rospy.loginfo(f"Saved grasp pose {len(self.saved_grasps)}/{self.max_saved} to {self.output_path}")

                # Optional: save timestamped backup too
                timestamp = rospy.get_time()
                backup_path = f"/catkin_ws/src/obj_manipulation/grasp_poses_{int(timestamp)}.txt"
                np.savetxt(backup_path, grasp_matrix, fmt="%.6f")
            else:
                # Only log after the limit
                if len(self.saved_grasps) == self.max_saved:
                    rospy.loginfo("Reached maximum saved grasps. Further poses will not be logged.")
        except Exception as e:
            rospy.logwarn(f"Error saving grasp pose: {e}")

    # ------------------ Publishing ------------------

    def publish_grasp_pose(self, grasp_matrix):
        """Publish the grasp as a PoseStamped message."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "camera_link"

        # Translation
        pose_msg.pose.position.x = grasp_matrix[0, 3]
        pose_msg.pose.position.y = grasp_matrix[1, 3]
        pose_msg.pose.position.z = grasp_matrix[2, 3]

        # Rotation matrix → Quaternion
        rot = grasp_matrix[:3, :3]
        qw = np.sqrt(1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2]) / 2.0
        qx = (rot[2, 1] - rot[1, 2]) / (4.0 * qw)
        qy = (rot[0, 2] - rot[2, 0]) / (4.0 * qw)
        qz = (rot[1, 0] - rot[0, 1]) / (4.0 * qw)

        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Published PoseStamped message on /grasp_estimation_node/grasp_pose.")

    def publish_marker(self, grasp_matrix):
        """Publish a 3D arrow marker in RViz at the grasp pose."""
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "camera_link"
        marker.ns = "grasp_marker"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Pose (same as grasp pose)
        marker.pose.position.x = grasp_matrix[0, 3]
        marker.pose.position.y = grasp_matrix[1, 3]
        marker.pose.position.z = grasp_matrix[2, 3]

        rot = grasp_matrix[:3, :3]
        qw = np.sqrt(1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2]) / 2.0
        qx = (rot[2, 1] - rot[1, 2]) / (4.0 * qw)
        qy = (rot[0, 2] - rot[2, 0]) / (4.0 * qw)
        qz = (rot[1, 0] - rot[0, 1]) / (4.0 * qw)

        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        # Marker size and color
        marker.scale.x = 0.15  # arrow length
        marker.scale.y = 0.04  # shaft width
        marker.scale.z = 0.04  # head size
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # fully opaque

        marker.lifetime = rospy.Duration(1.0)
        self.marker_pub.publish(marker)

    # --------------------------------------------------

if __name__ == '__main__':
    try:
        node = GraspEstimationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Grasp Estimation Node terminated.")
