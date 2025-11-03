#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped


class GraspPoseTransformer:
    def __init__(self):
        rospy.init_node("grasp_pose_transformer", anonymous=True)
        rospy.loginfo("Starting Grasp Pose Transformer Node...")

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscriber and publisher
        self.pose_sub = rospy.Subscriber(
            "/grasp_estimation_node/grasp_pose", PoseStamped, self.pose_callback
        )
        self.pose_pub = rospy.Publisher("/grasp_pose_base", PoseStamped, queue_size=10)

        rospy.loginfo(
            "Listening for grasp poses in 'camera_link' and transforming to 'base_link'."
        )

    def pose_callback(self, msg):
        try:
            # Transform grasp pose to base_link frame
            transform = self.tf_buffer.lookup_transform(
                "base_link",  # target frame
                msg.header.frame_id,  # source frame (camera_link)
                rospy.Time(0),  # get the latest available transform
                rospy.Duration(1.0),  # timeout
            )

            pose_base = tf2_geometry_msgs.do_transform_pose(msg, transform)
            pose_base.header.stamp = rospy.Time.now()
            pose_base.header.frame_id = "base_link"

            # Publish transformed pose
            self.pose_pub.publish(pose_base)
            rospy.loginfo_throttle(
                1.0, f"Published transformed grasp pose in 'base_link' frame."
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(2.0, f"TF transform failed: {e}")


if __name__ == "__main__":
    try:
        node = GraspPoseTransformer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
