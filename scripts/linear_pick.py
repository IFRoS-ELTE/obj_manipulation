#!/usr/bin/env python
"""
Linear pick - approaches target with linear/Cartesian motion then grips
"""

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler, quaternion_matrix
import time
import numpy as np

def monitor_gripper_effort():
    """Monitor gripper effort from joint states"""
    effort_data = {'finger_joint': {'effort': 0.0, 'position': 0.0}}
    
    def callback(msg):
        try:
            idx = msg.name.index('finger_joint')
            effort_data['finger_joint']['effort'] = msg.effort[idx] if idx < len(msg.effort) else 0.0
            effort_data['finger_joint']['position'] = msg.position[idx] if idx < len(msg.position) else 0.0
        except:
            pass
    
    rospy.Subscriber('/joint_states', JointState, callback)
    rospy.sleep(0.5)
    return effort_data

def open_gripper(gripper_group):
    """Open the gripper"""
    print("Opening gripper...")
    try:
        gripper_group.set_start_state_to_current_state()
        gripper_group.set_named_target("open")
        plan = gripper_group.plan()
        if plan and plan.joint_trajectory.points:
            gripper_group.execute(plan, wait=True)
            print("Gripper opened!")
            return True
    except Exception as e:
        print("Error opening gripper: {}".format(e))
    return False

def approach_with_linear_motion(arm_group, target_pose, offset_z=0.05):
    """
    Approach target with linear/Cartesian motion in the end effector's +z direction
    
    Args:
        arm_group: MoveGroupCommander for arm
        target_pose: Target Pose to approach
        offset_z: Distance above target to approach from (default 5cm) in eef frame
    
    Returns:
        success: True if approach successful
    """
    print("\n=== Approach with Linear Motion (in EEF frame) ===")
    
    # Get planning frame to show user
    planning_frame = arm_group.get_planning_frame()
    pose_reference_frame = arm_group.get_pose_reference_frame()
    print("Planning frame: {}".format(planning_frame))
    print("Pose reference frame: {}".format(pose_reference_frame))
    
    # First, get current pose
    current_pose = arm_group.get_current_pose()
    print("Current pose in {}: ({:.3f}, {:.3f}, {:.3f})".format(
        pose_reference_frame or planning_frame,
        current_pose.pose.position.x,
        current_pose.pose.position.y,
        current_pose.pose.position.z))
    
    # Show target pose
    print("Target pose in {}: ({:.3f}, {:.3f}, {:.3f})".format(
        pose_reference_frame or planning_frame,
        target_pose.position.x,
        target_pose.position.y,
        target_pose.position.z))
    
    # Get the z-axis of the end effector frame in the world/base frame
    # The z-axis direction in the end effector frame is transformed by the orientation
    
    # Get current orientation as quaternion
    q = [current_pose.pose.orientation.x,
         current_pose.pose.orientation.y,
         current_pose.pose.orientation.z,
         current_pose.pose.orientation.w]
    
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_matrix(q)
    
    # Extract z-axis from rotation matrix (3rd column, first 3 elements)
    # In the end effector frame, +z is [0, 0, 1]
    # Transform to world frame by rotating it
    eef_z_in_world = np.array(rotation_matrix[:3, 2])  # Extract z-axis as numpy array
    
    print("EEF z-axis in world frame: ({:.3f}, {:.3f}, {:.3f})".format(
        eef_z_in_world[0], eef_z_in_world[1], eef_z_in_world[2]))
    
    # Create approach position by moving in +z direction of end effector
    approach_pose = Pose()
    approach_pose.position.x = target_pose.position.x - offset_z * eef_z_in_world[0]
    approach_pose.position.y = target_pose.position.y - offset_z * eef_z_in_world[1]
    approach_pose.position.z = target_pose.position.z - offset_z * eef_z_in_world[2]
    approach_pose.orientation = target_pose.orientation
    
    print("Approach pose: {:.3f}, {:.3f}, {:.3f}".format(
        approach_pose.position.x,
        approach_pose.position.y,
        approach_pose.position.z))
    
    # Step 1: Move to approach position using regular planning
    print("\nStep 1: Moving to approach position...")
    arm_group.set_start_state_to_current_state()
    arm_group.set_pose_target(approach_pose)
    plan = arm_group.plan()
    
    if not plan or not plan.joint_trajectory.points:
        print("Failed to plan to approach position!")
        return False
    
    arm_group.execute(plan, wait=True)
    print("Reached approach position!")
    
    # Step 2: Linear approach to target using Cartesian path
    print("\nStep 2: Linear approach to target...")
    
    # Compute Cartesian path - this is the linear motion!
    waypoints = [approach_pose, target_pose]
    
    try:
        # Generate Cartesian path
        (plan_cart, fraction) = arm_group.compute_cartesian_path(
            waypoints,        # waypoints to follow
            0.01,             # eef_step (1 cm resolution)
            0.0,              # jump_threshold (0.0 = disabled)
            avoid_collisions=True
        )
        
        print("Generated Cartesian path with {:.1f}% success".format(fraction * 100))
        
        if fraction > 0.5:  # At least 50% of path is feasible
            plan_cart.joint_trajectory.header.stamp = rospy.Time.now()
            arm_group.execute(plan_cart, wait=True)
            print("Reached target with linear motion!")
            return True
        else:
            print("Cartesian path has low success rate ({:.1f}%)".format(fraction * 100))
            print("Trying direct pose...")
            
            # Fallback: direct pose approach
            arm_group.set_start_state_to_current_state()
            arm_group.set_pose_target(target_pose)
            plan2 = arm_group.plan()
            if plan2 and plan2.joint_trajectory.points:
                arm_group.execute(plan2, wait=True)
                print("Reached target with joint motion (fallback)")
                return True
            
            return False
            
    except Exception as e:
        print("Error in linear approach: {}".format(e))
        return False

def close_gripper(gripper_group, effort_data, max_effort=50.0):
    """
    Close gripper with effort monitoring
    
    Args:
        gripper_group: MoveGroupCommander for gripper
        effort_data: Dictionary with effort monitoring
        max_effort: Maximum effort threshold (Nm)
    
    Returns:
        success, final_effort
    """
    print("\n=== Closing Gripper ===")
    
    try:
        # Use close state
        gripper_group.set_start_state_to_current_state()
        gripper_group.set_named_target("close")
        plan = gripper_group.plan()
        
        if not plan or not plan.joint_trajectory.points:
            print("Failed to plan gripper closing")
            return False, 0.0
        
        # Start closing
        gripper_group.async_execute(plan)
        
        # Monitor effort during closing
        start_time = time.time()
        timeout = 5.0
        max_effort_seen = 0.0
        
        print("Monitoring effort (max: {} Nm)...".format(max_effort))
        
        while time.time() - start_time < timeout:
            effort = abs(effort_data['finger_joint']['effort'])
            max_effort_seen = max(max_effort_seen, effort)
            
            if effort >= max_effort:
                print("\nMax effort reached: {:.2f} Nm".format(effort))
                gripper_group.stop()
                rospy.sleep(0.5)
                return True, effort
            
            rospy.sleep(0.1)
        
        print("\nClosing completed. Max effort: {:.2f} Nm".format(max_effort_seen))
        return True, max_effort_seen
        
    except Exception as e:
        print("Error closing gripper: {}".format(e))
        return False, 0.0

def main():
    rospy.init_node('linear_pick', anonymous=True)
    
    # Initialize MoveIt
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    arm_group = moveit_commander.MoveGroupCommander("xarm6")
    gripper_group = moveit_commander.MoveGroupCommander("gripper")
    
    # Configure arm for better performance (exactly like xarm_move.py)
    arm_group.set_planner_id("RRTConnectkConfigDefault")
    arm_group.set_planning_time(10.0)
    arm_group.set_num_planning_attempts(5)
    arm_group.allow_replanning(True)
    arm_group.set_goal_position_tolerance(0.01)
    arm_group.set_goal_orientation_tolerance(0.05)
    arm_group.set_max_velocity_scaling_factor(0.3)
    arm_group.set_max_acceleration_scaling_factor(0.3)
    
    # Initialize effort monitoring
    effort_data = monitor_gripper_effort()
    
    rospy.sleep(1)
    
    # Show frame information like xarm_move.py does
    print("Planning frame:", arm_group.get_planning_frame())
    print("End effector link:", arm_group.get_end_effector_link())
    
    # IMPORTANT: Set reference frame to planning frame (like xarm_move.py line 40)
    arm_group.set_pose_reference_frame(arm_group.get_planning_frame())
    
    print("="*60)
    print("Linear Pick Demo")
    print("="*60)
    print("This will:")
    print("1. Open gripper")
    print("2. Approach target with linear motion")
    print("3. Close gripper with effort monitoring")
    print("4. Hold for 3 seconds")
    print("5. Open gripper")
    print("="*60)
    print("All poses are interpreted in frame: {}".format(arm_group.get_planning_frame()))
    
    # Open gripper
    if not open_gripper(gripper_group):
        print("Failed to open gripper")
        return
    
    # Get target from user
    print("\nEnter target pose:")
    try:
        x = float(input("X position (default 0.5): ") or "0.5")
        y = float(input("Y position (default 0.2): ") or "0.2")
        z = float(input("Z position (default 0.3): ") or "0.3")
        roll = float(input("Roll in degrees (default 0): ") or "0")
        pitch = float(input("Pitch in degrees (default 90): ") or "90")
        yaw = float(input("Yaw in degrees (default 0): ") or "0")
    except:
        x, y, z, roll, pitch, yaw = 0.5, 0.2, 0.3, 0, 90, 0
    
    # Create target pose
    target_pose = Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z
    
    qx, qy, qz, qw = quaternion_from_euler(
        roll * 3.14159 / 180.0,
        pitch * 3.14159 / 180.0,
        yaw * 3.14159 / 180.0
    )
    target_pose.orientation.x = qx
    target_pose.orientation.y = qy
    target_pose.orientation.z = qz
    target_pose.orientation.w = qw
    
    print("\nTarget pose: ({:.3f}, {:.3f}, {:.3f})".format(x, y, z))
    print("Target orientation: rpy({}, {}, {}) deg".format(roll, pitch, yaw))
    
    # Approach with linear motion
    success = approach_with_linear_motion(arm_group, target_pose, offset_z=0.05)
    
    if success:
        # Close gripper with effort monitoring
        gripper_success, final_effort = close_gripper(gripper_group, effort_data, max_effort=50.0)
        
        if gripper_success:
            print("\nGrasping successful! Max effort: {:.2f} Nm".format(final_effort))
            
            # Hold for 3 seconds
            print("\nHolding for 3 seconds...")
            for i in range(3):
                effort = effort_data['finger_joint']['effort']
                print("  {}s: Effort = {:.2f} Nm".format(i+1, effort))
                rospy.sleep(1.0)
            
            # Open gripper
            print("\nReleasing...")
            open_gripper(gripper_group)
            
            print("\n=== Demo completed! ===")
        else:
            print("Gripper closing failed")
    else:
        print("Linear approach failed")
    
    arm_group.stop()
    gripper_group.stop()
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print("Error: {}".format(e))

