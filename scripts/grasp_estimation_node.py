#!/usr/bin/env python3

from typing import Dict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

import rospy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header

from obj_manipulation.grasp import (
    GraspEstimatorCGN,
    PointCloudFilterSAM,
    PointCloudFilterUOIS,
)
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
        self.trigger = False

        # -------- Load node parameters --------
        self.alpha = rospy.get_param("/grasp_estimation_node/alpha", 0.6)
        self.max_trials = rospy.get_param("/grasp_estimation_node/max_trials", 25)
        self.approach_distance = rospy.get_param("/grasp_estimation_node/approach_distance", 0.17)
        self.visualize_grasps = rospy.get_param("/grasp_estimation_node/visualize_grasps", False)
        seg_model = rospy.get_param("/grasp_estimation_node/seg_model", "sam")
        assert seg_model in ["sam", "uois"]

        # -------- Load configuration and model --------
        self.pc_filter = PointCloudFilterSAM() if seg_model == "sam" else PointCloudFilterUOIS()
        config_path = Path(__file__).parents[1] / "obj_manipulation/grasp/config/config.toml"
        assert config_path.exists()
        config = load_config(config_path)
        self.n_input_points = config["n_input_points"]
        self.gripper_depth = config["gripper_depth"]

        self.grasp_est = GraspEstimatorCGN(config)
        self.grasp_est.load()
        self.grasp_est.eval_mode()

        # -------- TF2 Setup ---------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
        self.trigger_sub = rospy.Subscriber(  # Trigger subscriber
            "/grasp_estimation_node/trigger", Bool, self.trigger_callback
        )
        rospy.loginfo("Subscribed to /grasp_estimation_node/trigger for one-shot control.")

        # -------- Publishers --------
        self.pose_pub = rospy.Publisher("/grasp_estimation_node/grasp_pose", PoseStamped, queue_size=1)
        self.obj_mask_pub = rospy.Publisher("/grasp_estimation_node/obj_mask", Image, queue_size=1)
        rospy.loginfo("Initialization complete. Waiting for camera data...")

    @property
    def ready_to_publish(self) -> bool:
        ready = all([
            self.rgb_image is not None,
            self.xyz_image is not None,
            self.trigger,
        ])
        return ready
    
    # ------------------ Trigger Control ------------------
    def trigger_callback(self, msg: Bool) -> None:
        """Enable grasp estimation when a Bool trigger (data: true) is received."""
        if msg.data:
            rospy.loginfo("Received trigger signal — grasp estimation enabled for one run.")
            self.trigger = True
        
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
        # Step 1: Run model
        rospy.loginfo("Running grasp estimation model...")
        for _ in range(self.max_trials):
            pc_filter_out = self.pc_filter.filter_point_cloud(
                self.xyz_image, self.rgb_image, n_points=self.n_input_points
            )
            xyz_pc, xyz_object_pc, obj_mask = pc_filter_out
            result = self.grasp_est.predict_grasps(xyz_pc, xyz_object_pc)
            if result is not None:
                break
        rospy.loginfo("Grasp estimation model finished inference.")

        # Step 2: Extract best grasp
        if result is None or result["pred_grasps"].shape == ():
            rospy.logwarn("No valid grasps predicted.")
            return
        best_pose = result["pred_grasps"][np.random.randint(0, len(result["pred_grasps"]))]

        # Step 3: Publish Pose + Object Mask
        self.publish_grasp_pose(best_pose)
        self.publish_obj_mask(obj_mask.cpu().numpy())
        rospy.loginfo("Grasp pose and marker published successfully.")

        # Step 4: Optionally visualize grasps in 3D
        if self.visualize_grasps:
            self.visualize_grasps_open3d(result)
        rospy.loginfo(f"Grasp estimation run complete at {rospy.get_time():.2f}. Waiting for next trigger.")

    # ------------------ Grasp Visualization ------------------
    def visualize_grasps_open3d(self, grasp_pred: Dict[str, NDArray]) -> None:
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

    # ------------------ Publishing ------------------
    def publish_grasp_pose(self, grasp_matrix) -> None:
        """Publish the grasp as a PoseStamped message."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()

        # Translation
        grasp_matrix[:3, :3] = grasp_matrix[:3, :3] @ R.from_euler('y', [-np.pi/2]).as_matrix()
        grasp_matrix[:3, :3] = grasp_matrix[:3, :3] @ R.from_euler('x', [-np.pi/2]).as_matrix()
        grasp_matrix[:3, 3] -= grasp_matrix[:3, 0] * 0.17

        pose_msg.pose.position.x = grasp_matrix[0, 3]
        pose_msg.pose.position.y = grasp_matrix[1, 3]
        pose_msg.pose.position.z = grasp_matrix[2, 3]

        # Rotation matrix → Quaternion
        quat = R.from_matrix(grasp_matrix[:3, :3]).as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        # Transform pose to base link
        try:
            transform = self.tf_buffer.lookup_transform(
                "dummy_base_link",                    # target frame
                "camera_color_optical_frame",   # source frame
                rospy.Time(0),                  # latest available transform
                rospy.Duration(secs=1),         # timeout duration
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Failed to fetch tf2 transform to base link due to error: {e}")
            return
        pose_msg = tf2_geometry_msgs.do_transform_pose(pose_msg, transform)
        
        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Published PoseStamped message on /grasp_estimation_node/grasp_pose.")

    def publish_obj_mask(self, obj_mask: NDArray[np.bool_]) -> None:
        # Set color of pixels covered by object mask
        COLOR_MASK = np.array([30, 144, 255], dtype=np.uint8)
        seg_image = np.where(
            obj_mask[..., None],
            self.alpha * COLOR_MASK + (1 - self.alpha) * self.rgb_image,
            self.rgb_image,
        ).astype(np.uint8)

        # Convert from NumPy to ROS Image
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_color_optical_frame"
        seg_image_msg = self.bridge.cv2_to_imgmsg(seg_image, encoding="rgb8", header=header)
        
        # Publish message
        self.obj_mask_pub.publish(seg_image_msg)


def main():
    node = GraspEstimationNode()
    rate = rospy.Rate(hz=1)  # Check for triggers at a fixed rate of 1 Hz
    while not rospy.is_shutdown():
        if node.ready_to_publish:
            node.try_predict_grasp()
            node.trigger = False
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Grasp Estimation Node terminated.")
