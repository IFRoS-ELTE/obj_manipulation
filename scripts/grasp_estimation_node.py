#!/usr/bin/env python3

import os

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

from obj_manipulation.grasp import GraspEstimatorCGN
from obj_manipulation.grasp.utils import (
    depth_map_to_xyz,
    load_config,
)
from obj_manipulation.grasp.utils.utils_visualization import visualize_grasps


class GraspEstimationNode:
    def __init__(self):
        rospy.init_node('grasp_estimation_node', anonymous=True)
        rospy.loginfo("Initializing Grasp Estimation Node...")

        self.bridge = CvBridge()
        self.rgb_image = None
        self.xyz_image = None
        self.camera_intrinsics = None

        # -------- Logging Configuration --------
        self.saved_grasps = []
        self.saved_clouds = 0
        self.max_saved = 10
        self.log_dir = "/catkin_ws/src/obj_manipulation/data_logs"

        os.makedirs(self.log_dir, exist_ok=True)
        rospy.loginfo(f"Logging directory: {self.log_dir}")

        # -------- Load configuration and model --------
        config_path = "/catkin_ws/src/obj_manipulation/obj_manipulation/grasp/config/config.toml"
        assert os.path.exists(config_path)
        rospy.loginfo(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        self.gripper_depth = config["gripper_depth"]

        rospy.loginfo("Initializing grasp estimator...")
        self.grasp_est = GraspEstimatorCGN(config)
        self.grasp_est.load()
        self.grasp_est.eval_mode()
        rospy.loginfo("Grasp estimator model loaded and ready.")

        # -------- Load node parameters --------
        self.max_trials = rospy.get_param("/grasp_estimation_node/max_trials", 4)
        self.visualize_grasps = rospy.get_param("/grasp_estimation_node/visualize_grasps", False)

        # -------- Subscribers --------
        rospy.loginfo("Subscribing to camera topics...")
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.cam_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.cam_info_callback)

        # Trigger subscriber
        self.trigger_sub = rospy.Subscriber(
            "/grasp_estimation_node/trigger", Bool, self.trigger_callback
        )
        rospy.loginfo("Subscribed to /grasp_estimation_node/trigger for one-shot control.")

        # -------- Publishers --------
        self.pose_pub = rospy.Publisher("/grasp_estimation_node/grasp_pose", PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/grasp_estimation_node/marker", Marker, queue_size=10)
        rospy.loginfo("Publishers created for grasp pose and RViz marker.")

        rospy.loginfo("Initialization complete. Waiting for camera data...")

    @property
    def ready_to_publish(self) -> bool:
        ready = all([
            self.rgb_image is not None,
            self.xyz_image is not None,
        ])
        return ready
    
    # ------------------ Trigger Control ------------------
    def trigger_callback(self, msg: Bool) -> None:
        """Enable grasp estimation when a Bool trigger (data: true) is received."""
        if msg.data and self.ready_to_publish:
            rospy.loginfo("Received trigger signal — grasp estimation enabled for one run.")
            self.try_predict_grasp()

    # ------------------ Callbacks ------------------
    def cam_info_callback(self, msg: CameraInfo) -> None:
        K = np.array(msg.K).reshape(3, 3)
        self.camera_intrinsics = K
        rospy.loginfo_once("Received camera intrinsics.")

    def rgb_callback(self, msg: Image) -> None:
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        rospy.loginfo_once("Received first RGB image.")

    def depth_callback(self, msg: Image) -> None:
        if self.camera_intrinsics is not None:    
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding == "16UC1":
                self.depth = self.depth.astype(np.float32) / 1000.0  # convert mm → meters 
            self.xyz_image = depth_map_to_xyz(self.depth, self.camera_intrinsics)
            rospy.loginfo_once("Received first xyz image.")

    # ------------------ Main Prediction ------------------
    def try_predict_grasp(self):
        rospy.loginfo("Starting grasp prediction pipeline...")

        # # Optional: save first 10 point clouds
        # if self.saved_clouds < self.max_saved:
        #     self.save_point_cloud(xyz_img)
        #     self.saved_clouds += 1

        # Step 1: Run model
        rospy.loginfo("Running grasp estimation model...")
        for _ in range(self.max_trials):
            result = self.grasp_est.predict_grasps(self.xyz_image, self.rgb_image)
            if result is not None:
                break
        rospy.loginfo("Grasp estimation model finished inference.")

        # Step 2: Extract best grasp
        if result is None or "pred_grasps" not in result or len(result["pred_grasps"]) == 0:
            rospy.logwarn("No valid grasps predicted.")
            return
        best_pose = result["pred_grasps"][0]
        rospy.loginfo("Best grasp pose extracted (index 0).")

        # Step 3: Publish Pose + Marker
        self.publish_grasp_pose(best_pose)
        self.publish_marker(best_pose)
        rospy.loginfo("Grasp pose and marker published successfully.")

        # Step 4: Optionally visualize grasps in 3D
        if self.visualize_grasps:
            self.visualize_grasps_open3d(result)

        # Step 5: Save first 10 best grasp poses
        # if len(self.saved_grasps) < self.max_saved:
        #     self.save_grasp_pose(best_pose)

        rospy.loginfo(f"Grasp estimation run complete at {rospy.get_time():.2f}. Waiting for next trigger.")

    # ------------------ Grasp Visualization ------------------
    def visualize_grasps_open3d(self, grasp_pred: dict[str, NDArray]) -> None:
        # Filter xyz and rgb data according to depth mask used by grasp predictor
        depth_mask = np.logical_and(
            self.depth > self.grasp_est.min_depth,
            self.depth < self.grasp_est.max_depth,
        )
        if not np.any(depth_mask):
            return
        xyz_pc = self.xyz_image[depth_mask]
        rgb_pc = self.rgb_image[depth_mask]

        # Visualize grasps using Open3D
        visualize_grasps(
            xyz_pc,
            rgb_pc,
            [grasp_pred["pred_grasps"]],
            [grasp_pred["pred_scores"]],
            [grasp_pred["pred_widths"]],
            self.gripper_depth
        )


    # ------------------ Data Saving ------------------
    def save_point_cloud(self, xyz_img):
        """Save point cloud as .ply file."""
        try:
            points = xyz_img.reshape(-1, 3)
            mask = np.isfinite(points).all(axis=1)
            points = points[mask]

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)

            path = os.path.join(self.log_dir, f"pointcloud_{self.saved_clouds + 1:02d}.ply")
            o3d.io.write_point_cloud(path, cloud)
            rospy.loginfo(f"Saved point cloud #{self.saved_clouds + 1} to: {path}")
        except Exception as e:
            rospy.logwarn(f"Error saving point cloud: {e}")

    def save_grasp_pose(self, grasp_matrix):
        """Save grasp pose as .npy file."""
        try:
            idx = len(self.saved_grasps) + 1
            self.saved_grasps.append(grasp_matrix)
            path = os.path.join(self.log_dir, f"grasp_pose_{idx:02d}.npy")
            np.save(path, grasp_matrix)
            rospy.loginfo(f"Saved grasp pose #{idx} to: {path}")
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
        marker.scale.x = 0.15
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(1.0)
        self.marker_pub.publish(marker)


if __name__ == '__main__':
    try:
        node = GraspEstimationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Grasp Estimation Node terminated.")
