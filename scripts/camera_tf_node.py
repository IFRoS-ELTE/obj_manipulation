#!/usr/bin/env python3
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


class CameraTFNode:
    def __init__(self):
        rospy.init_node("camera_tf_node", anonymous=True)
        
        # -------- Load node parameters --------
        self.base_frame = rospy.get_param("/camera_tf_node/base_frame", "base_link")
        self.input_frame = rospy.get_param(
            "/camera_tf_node/input_frame", "camera_color_optical_frame"
        )
        self.output_frame = rospy.get_param(
            "/camera_tf_node/output_frame", "camera_color_optical_frame_virtual"
        )
        
        offset_dist = rospy.get_param("/camera_tf_node/offset_dist", 1.0)
        self.offset = np.array([offset_dist, 0.0, 0.0])
        self.rate = rospy.get_param("/camera_tf_node/rate", 5)
        
        # -------- TF2 Setup ---------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    def get_camera_transform(self) -> Tuple[R, rospy.Time]:
        """Get the latest camera frame transform with respect to the base frame.
        
        Returns:
            tuple
            - position: [3] array containing the camera's position.
            - rotation: Rotation object for the camera's orientation.
            - stamp: Timestamp of the recieved transform.
        """
        # Look-up transform
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,            # target frame
            self.input_frame,           # source frame
            rospy.Time(0),              # latest available transform
            rospy.Duration(secs=2),     # timeout duration
        )
        
        # Extract translation and orientation
        r = transform.transform.rotation
        rotation = R.from_quat([r.x, r.y, r.z, r.w])
        stamp = transform.header.stamp 
        return rotation, stamp

    def pub_virtual_camera_transform(self, rot: R, stamp: rospy.Time) -> None:
        """Publish the latest virtual camera frame transform with respect to the camera frame.
        """
        # Offset position along x-axis of the world frame
        pos_tf = rot.apply(self.offset, inverse=True)

        # Compute virtual frame orientation based on actual camera orientation
        roll = rot.as_euler("xyz")[0]
        offset_rot = R.from_euler("XZ", [-2 * roll, np.pi])
        quat_tf = offset_rot.as_quat()

        # Create transform
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.input_frame
        transform.child_frame_id = self.output_frame

        transform.transform.translation.x = pos_tf[0]
        transform.transform.translation.y = pos_tf[1]
        transform.transform.translation.z = pos_tf[2]

        transform.transform.rotation.x = quat_tf[0]
        transform.transform.rotation.y = quat_tf[1]
        transform.transform.rotation.z = quat_tf[2]
        transform.transform.rotation.w = quat_tf[3]
        
        # Publish transform
        self.tf_broadcaster.sendTransform(transform)


def main():
    node = CameraTFNode()
    rate = rospy.Rate(node.rate)
    while not rospy.is_shutdown():
        rot, stamp = node.get_camera_transform()
        node.pub_virtual_camera_transform(rot, stamp)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera Transform Node terminated.")
