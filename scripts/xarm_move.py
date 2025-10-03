#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
import time

def get_goal_from_user():
    """Get goal coordinates from user input"""
    print("Enter goal coordinates:")
    try:
        x = float(input("X position: "))
        y = float(input("Y position: "))
        z = float(input("Z position: "))
        return x, y, z
    except ValueError:
        print("Invalid input. Using default values.")
        return 0.5, 0.4, 0.5

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

    # Configure planning for robustness
    planning_frame = group.get_planning_frame()
    group.set_pose_reference_frame(planning_frame)
    group.set_goal_position_tolerance(0.01)
    group.set_goal_orientation_tolerance(0.1)
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(10)
    group.allow_replanning(True)
    group.set_max_velocity_scaling_factor(0.3)
    group.set_max_acceleration_scaling_factor(0.3)

    # Allow some time for MoveIt to initialize
    rospy.sleep(2)

    # Get goal from user
    x, y, z = get_goal_from_user()

    # Define target pose
    target_pose = Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z
    target_pose.orientation.x = 0.0
    target_pose.orientation.y = 0.0
    target_pose.orientation.z = 0.0
    target_pose.orientation.w = 1.0

    # Create PoseStamped for RViz visualization
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = group.get_planning_frame()
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose = target_pose
    goal_pub.publish(pose_stamped)
    rospy.sleep(0.5)  # allow RViz to update

    print("Planning to position: x={}, y={}, z={}".format(x, y, z))

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
        print("Planning failed with OMPL. Trying Cartesian path fallback...")
        waypoints = []
        current_pose = group.get_current_pose().pose
        waypoint = Pose()
        waypoint.position.x = x
        waypoint.position.y = y
        waypoint.position.z = z
        waypoint.orientation = current_pose.orientation if current_pose.orientation.w != 0.0 else target_pose.orientation
        waypoints.append(waypoint)

        (fraction, cartesian_plan, _) = group.compute_cartesian_path(
            waypoints,
            eef_step=0.01,
            jump_threshold=0.0,
            avoid_collisions=True,
        )
        if fraction > 0.7 and cartesian_plan and cartesian_plan.joint_trajectory.points:
            group.execute(cartesian_plan, wait=True)
            print("Cartesian movement completed (fraction={:.2f})".format(fraction))
        else:
            print("Planning failed! Target may be unreachable or in collision. Cartesian fraction={:.2f}".format(fraction))

    # Clean shutdown
    group.stop()
    group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
