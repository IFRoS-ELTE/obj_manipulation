#!/usr/bin/env python
"""
Demo script showing how to use end-effector offsets with xarm_move.py
This script demonstrates the integrated offset functionality.
"""

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler
import time

def demo_offset_functionality():
    """Demonstrate end-effector offset functionality"""
    print("=== XArm End-Effector Offset Demo ===")
    print("This demo shows how to use shift_pose_target for end-effector offsets")
    
    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('offset_demo', anonymous=True)

    # Create robot and scene objects
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("xarm6")

    # Configuration
    group.set_pose_reference_frame(group.get_planning_frame())
    group.set_planner_id("RRTConnectkConfigDefault")
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(10)
    group.allow_replanning(True)
    group.set_goal_position_tolerance(0.01)
    group.set_goal_orientation_tolerance(0.05)
    group.set_max_velocity_scaling_factor(0.3)
    group.set_max_acceleration_scaling_factor(0.3)

    # Allow MoveIt to initialize
    rospy.sleep(2)

    eef_link = group.get_end_effector_link()
    print(f"End-effector link: {eef_link}")
    print(f"Planning frame: {group.get_planning_frame()}")

    # Example 1: Move to a base position
    print("\n--- Example 1: Moving to base position ---")
    base_pose = Pose()
    base_pose.position.x = 0.5
    base_pose.position.y = 0.0
    base_pose.position.z = 0.5
    base_pose.orientation.w = 1.0  # No rotation
    
    group.set_start_state_to_current_state()
    group.set_pose_target(base_pose)
    
    plan = group.plan()
    if plan and plan.joint_trajectory.points:
        group.execute(plan, wait=True)
        print("✓ Moved to base position")
    else:
        print("✗ Failed to move to base position")
        return

    # Example 2: Apply Z offset (move up by 5cm)
    print("\n--- Example 2: Applying Z offset (+5cm) ---")
    group.shift_pose_target(2, 0.05, eef_link)  # Z-axis offset
    print("Applied Z offset: +0.05 m")
    
    plan = group.plan()
    if plan and plan.joint_trajectory.points:
        group.execute(plan, wait=True)
        print("✓ Applied Z offset successfully")
    else:
        print("✗ Failed to apply Z offset")

    # Example 3: Apply X offset (move forward by 2cm)
    print("\n--- Example 3: Applying X offset (+2cm) ---")
    group.shift_pose_target(0, 0.02, eef_link)  # X-axis offset
    print("Applied X offset: +0.02 m")
    
    plan = group.plan()
    if plan and plan.joint_trajectory.points:
        group.execute(plan, wait=True)
        print("✓ Applied X offset successfully")
    else:
        print("✗ Failed to apply X offset")

    # Example 4: Apply Y offset (move left by 3cm)
    print("\n--- Example 4: Applying Y offset (-3cm) ---")
    group.shift_pose_target(1, -0.03, eef_link)  # Y-axis offset
    print("Applied Y offset: -0.03 m")
    
    plan = group.plan()
    if plan and plan.joint_trajectory.points:
        group.execute(plan, wait=True)
        print("✓ Applied Y offset successfully")
    else:
        print("✗ Failed to apply Y offset")

    # Example 5: Multiple offsets at once
    print("\n--- Example 5: Applying multiple offsets ---")
    group.shift_pose_target(0, 0.01, eef_link)  # X: +1cm
    group.shift_pose_target(1, 0.02, eef_link)  # Y: +2cm
    group.shift_pose_target(2, -0.03, eef_link)  # Z: -3cm
    print("Applied multiple offsets: X:+0.01, Y:+0.02, Z:-0.03")
    
    plan = group.plan()
    if plan and plan.joint_trajectory.points:
        group.execute(plan, wait=True)
        print("✓ Applied multiple offsets successfully")
    else:
        print("✗ Failed to apply multiple offsets")

    print("\n=== Demo completed ===")
    print("You can now use the updated xarm_move.py script with offset functionality!")
    
    # Clean shutdown
    group.stop()
    group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        demo_offset_functionality()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        pass
