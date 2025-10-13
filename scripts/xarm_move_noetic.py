#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
import math

rospy.init_node("goal_publisher", anonymous=True)
pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=10)
rate = rospy.Rate(1)  # 1 Hz

while not rospy.is_shutdown():
    x = float(input("X: "))
    y = float(input("Y: "))
    z = float(input("Z: "))
    roll = float(input("Roll: "))
    pitch = float(input("Pitch: "))
    yaw = float(input("Yaw: "))

    qx, qy, qz, qw = quaternion_from_euler(
        math.radians(roll), math.radians(pitch), math.radians(yaw)
    )
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"
    msg.pose = pose

    pub.publish(msg)
    rospy.loginfo("Published new goal to Melodic")
    rate.sleep()
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
import math

rospy.init_node("goal_publisher", anonymous=True)
pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=10)
rate = rospy.Rate(1)  # 1 Hz

while not rospy.is_shutdown():
    x, y, z = 0.5, 0.3, 0.5
    roll, pitch, yaw = 0, 0, 0
    qx, qy, qz, qw = quaternion_from_euler(
        math.radians(roll), math.radians(pitch), math.radians(yaw)
    )
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"
    msg.pose = pose

    pub.publish(msg)
    rospy.loginfo("Published new goal to Melodic")
    rate.sleep()
