#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler
import time

def get_goal_from_user():
    """Get goal position (m) and orientation (rpy in degrees) from user input"""
    print("Enter goal pose (position in meters, orientation in degrees):")
    try:
        x = float(input("X position: "))
        y = float(input("Y position: "))
        z = float(input("Z position: "))
        roll_deg = float(input("Roll (deg): ") or 0)
        pitch_deg = float(input("Pitch (deg): ") or 0)
        yaw_deg = float(input("Yaw (deg): ") or 0)
        return x, y, z, roll_deg, pitch_deg, yaw_deg
    except ValueError:
        print("Invalid input. Using defaults (0.5,0.4,0.5, rpy=0,0,0).")
        return 0.5, 0.4, 0.5, 0.0, 0.0, 0.0

def main():
    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm_python', anonymous=True)

    # Publisher for RViz visualization
    goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

    # Create robot and scene objects
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("xarm6")

    # Planner and motion configuration for robustness
    print("Planning frame:", group.get_planning_frame())
    print("End effector link:", group.get_end_effector_link())
    group.set_pose_reference_frame(group.get_planning_frame())
    group.set_planner_id("RRTConnectkConfigDefault")
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(10)
    group.allow_replanning(True)
    group.set_goal_position_tolerance(0.01)
    group.set_goal_orientation_tolerance(0.05)
    group.set_max_velocity_scaling_factor(0.3)
    group.set_max_acceleration_scaling_factor(0.3)

    # Allow some time for MoveIt to initialize
    rospy.sleep(2)

    # Get goal from user
    x, y, z, roll_deg, pitch_deg, yaw_deg = get_goal_from_user()

    # Define target pose
    target_pose = Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z
    # Compute normalized quaternion from RPY (convert degrees to radians)
    qx, qy, qz, qw = quaternion_from_euler(
        roll_deg * 3.141592653589793 / 180.0,
        pitch_deg * 3.141592653589793 / 180.0,
        yaw_deg * 3.141592653589793 / 180.0
    )
    target_pose.orientation.x = qx
    target_pose.orientation.y = qy
    target_pose.orientation.z = qz
    target_pose.orientation.w = qw

    # Create PoseStamped for RViz visualization
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = group.get_planning_frame()
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose = target_pose
    goal_pub.publish(pose_stamped)
    rospy.sleep(0.5)  # allow RViz to update

    print("EEF link:", group.get_end_effector_link())
    print("Planning to pose: x={}, y={}, z={}, rpy(deg)=({}, {}, {})".format(x, y, z, roll_deg, pitch_deg, yaw_deg))

    # Set target and plan
    group.set_start_state_to_current_state()
    group.set_pose_target(target_pose)
    print(group)
    plan = group.plan()  # returns RobotTrajectory object in MoveIt 1
    print(plan)
    # Check if planning succeeded
    if plan and plan.joint_trajectory.points:
        group.execute(plan, wait=True)
        print("Movement completed!")
    else:
        print("Planning failed with RRTConnect. Retrying with BKPIECE and relaxed tolerances...")
        group.set_planner_id("BKPIECEkConfigDefault")
        group.set_goal_orientation_tolerance(0.15)
        group.set_start_state_to_current_state()
        group.set_pose_target(target_pose)
        plan_retry = group.plan()
        if plan_retry and plan_retry.joint_trajectory.points:
            group.execute(plan_retry, wait=True)
            print("Movement completed on retry!")
        else:
            print("Planning failed! Target may be unreachable or in collision.")

    # Clean shutdown
    group.stop()
    group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
