#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler
import time

def ensure_gripper_open_state(gripper_group):
    """Ensure gripper is in open state before starting operations"""
    print("Ensuring gripper is in open state...")
    
    # Get current state
    current_joints = gripper_group.get_current_joint_values()
    print("Current gripper joints:", current_joints)
    
    # Check if gripper is already open (all joints close to 0)
    open_threshold = 0.05  # Stricter threshold
    is_already_open = all(abs(joint) < open_threshold for joint in current_joints)
    
    if is_already_open:
        print("Gripper is already in open state!")
        return True
    else:
        print("Moving gripper to open state...")
        try:
            gripper_group.set_start_state_to_current_state()
            gripper_group.set_named_target("open")
            plan = gripper_group.plan()
            
            if plan and plan.joint_trajectory.points:
                gripper_group.execute(plan, wait=True)
                print("Gripper moved to open state successfully!")
                return True
            else:
                print("Failed to plan gripper to open state!")
                return False
        except Exception as e:
            print("Error moving gripper to open state:", e)
            return False

def check_gripper_collision(gripper_group):
    """Check if gripper is in collision and try to resolve"""
    print("Checking gripper collision state...")
    
    # Get current state
    current_joints = gripper_group.get_current_joint_values()
    print("Current gripper joints:", current_joints)
    
    # Try to find a valid state near current state
    try:
        gripper_group.set_start_state_to_current_state()
        # Try a small movement to test collision
        test_joints = [j + 0.01 for j in current_joints]  # Small offset
        gripper_group.set_joint_value_target(test_joints)
        plan = gripper_group.plan()
        
        if plan and plan.joint_trajectory.points:
            print("Gripper collision check passed!")
            return True
        else:
            print("Gripper collision detected! Attempting to resolve...")
            # Try to move to a known safe state
            gripper_group.set_named_target("open")
            plan = gripper_group.plan()
            if plan and plan.joint_trajectory.points:
                gripper_group.execute(plan, wait=True)
                print("Gripper moved to safe state!")
                return True
            else:
                print("Could not resolve gripper collision!")
                return False
    except Exception as e:
        print("Error checking gripper collision:", e)
        return False

def get_goal_from_user():
    """Get goal position (m) and orientation (rpy in degrees) from user input"""
    print("Enter goal pose (position in meters, orientation in degrees):")
    try:
        x = float(input("X position: ") or 0.5)
        y = float(input("Y position: ") or 0.2)
        z = float(input("Z position: ") or 0.4)
        roll_deg = float(input("Roll (deg): ") or 0)
        pitch_deg = float(input("Pitch (deg): ") or 0)
        yaw_deg = float(input("Yaw (deg): ") or 0)
        return x, y, z, roll_deg, pitch_deg, yaw_deg
    except (ValueError, EOFError):
        print("Using defaults (0.5,0.2,0.4, rpy=0,0,0).")
        return 0.5, 0.2, 0.4, 0.0, 0.0, 0.0

def main():
    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_and_grip', anonymous=True)

    # Publisher for RViz visualization
    goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

    # Create robot and scene objects
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    
    # Initialize arm and gripper groups
    arm_group = moveit_commander.MoveGroupCommander("xarm6")
    gripper_group = moveit_commander.MoveGroupCommander("gripper")

    # Arm configuration for robustness
    print("Planning frame:", arm_group.get_planning_frame())
    print("End effector link:", arm_group.get_end_effector_link())
    arm_group.set_pose_reference_frame(arm_group.get_planning_frame())
    arm_group.set_planner_id("RRTConnectkConfigDefault")
    arm_group.set_planning_time(10.0)
    arm_group.set_num_planning_attempts(10)
    arm_group.allow_replanning(True)
    arm_group.set_goal_position_tolerance(0.01)
    arm_group.set_goal_orientation_tolerance(0.05)
    arm_group.set_max_velocity_scaling_factor(0.3)
    arm_group.set_max_acceleration_scaling_factor(0.3)

    # Gripper configuration
    gripper_group.set_planning_time(5.0)
    gripper_group.set_num_planning_attempts(5)
    gripper_group.allow_replanning(True)
    gripper_group.set_goal_joint_tolerance(0.1)

    # Allow some time for MoveIt to initialize
    rospy.sleep(2)

    print("=== Move and Grip Demo ===")
    print("This script will:")
    print("1. Ensure gripper is in open state")
    print("2. Move the arm to a target position")
    print("3. Close the gripper")
    print("4. Wait 2 seconds")
    print("5. Open the gripper")
    print()

    # Step 0: Ensure gripper is in open state
    print("\n=== Step 0: Ensuring gripper is in open state ===")
    if not ensure_gripper_open_state(gripper_group):
        print("Could not ensure gripper open state. Exiting...")
        return

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
    pose_stamped.header.frame_id = arm_group.get_planning_frame()
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose = target_pose
    goal_pub.publish(pose_stamped)
    rospy.sleep(0.5)  # allow RViz to update

    print("Planning to pose: x={}, y={}, z={}, rpy(deg)=({}, {}, {})".format(x, y, z, roll_deg, pitch_deg, yaw_deg))

    # Step 1: Move arm to target position
    print("\n=== Step 1: Moving arm to target position ===")
    arm_group.set_start_state_to_current_state()
    arm_group.set_pose_target(target_pose)
    plan = arm_group.plan()
    
    # Check if planning succeeded
    if plan and plan.joint_trajectory.points:
        arm_group.execute(plan, wait=True)
        print("Arm movement completed!")
        
        # Step 2: Close gripper
        print("\n=== Step 2: Closing gripper ===")
        
        # Debug: Show current gripper state before closing
        current_joints = gripper_group.get_current_joint_values()
        print("Current gripper joints before closing:", current_joints)
        
        gripper_group.set_start_state_to_current_state()
        # Use SRDF predefined "close" state
        gripper_group.set_named_target("close")
        gripper_plan = gripper_group.plan()
        
        if gripper_plan and gripper_plan.joint_trajectory.points:
            gripper_group.execute(gripper_plan, wait=True)
            print("Gripper closed!")
        else:
            print("Gripper closing planning failed! Trying alternative approach...")
            # Try with joint values as fallback
            gripper_joint_values = [0.4355565, 0.4355565, 0.5, 0.5]  # close state values
            gripper_group.set_joint_value_target(gripper_joint_values)
            gripper_plan = gripper_group.plan()
            if gripper_plan and gripper_plan.joint_trajectory.points:
                gripper_group.execute(gripper_plan, wait=True)
                print("Gripper closed!")
            else:
                print("Gripper closing failed completely!")
                return  # Exit if gripper closing fails completely
        
        # Step 3: Wait
        print("\n=== Step 3: Waiting 2 seconds ===")
        rospy.sleep(2.0)
        
        # Step 4: Open gripper
        print("\n=== Step 4: Opening gripper ===")
        
        # Check current gripper state
        current_joints = gripper_group.get_current_joint_values()
        print("Current gripper joint values:", current_joints)
        
        # Check if gripper is already open (all joints close to 0)
        open_threshold = 0.1
        is_already_open = all(abs(joint) < open_threshold for joint in current_joints)
        
        if is_already_open:
            print("Gripper is already open! Skipping opening step.")
            print("\n=== Demo completed successfully! ===")
        else:
            gripper_group.set_start_state_to_current_state()
            # Use named state instead of joint values
            gripper_group.set_named_target("open")
            gripper_plan = gripper_group.plan()
            
            if gripper_plan and gripper_plan.joint_trajectory.points:
                gripper_group.execute(gripper_plan, wait=True)
                print("Gripper opened!")
                print("\n=== Demo completed successfully! ===")
            else:
                print("Gripper opening planning failed! Trying alternative approach...")
                # Try with SRDF named state as fallback
                gripper_group.set_named_target("open")
                gripper_plan = gripper_group.plan()
                if gripper_plan and gripper_plan.joint_trajectory.points:
                    gripper_group.execute(gripper_plan, wait=True)
                    print("Gripper opened!")
                    print("\n=== Demo completed successfully! ===")
                else:
                    print("Gripper opening failed completely!")
        # else:
        #     print("Gripper closing planning failed!")
    else:
        print("Arm planning failed with RRTConnect. Retrying with BKPIECE and relaxed tolerances...")
        arm_group.set_planner_id("BKPIECEkConfigDefault")
        arm_group.set_goal_orientation_tolerance(0.15)
        arm_group.set_start_state_to_current_state()
        arm_group.set_pose_target(target_pose)
        plan_retry = arm_group.plan()
        if plan_retry and plan_retry.joint_trajectory.points:
            arm_group.execute(plan_retry, wait=True)
            print("Arm movement completed on retry!")
            
            # Continue with gripper operations...
            print("\n=== Step 2: Closing gripper ===")
            gripper_group.set_start_state_to_current_state()
            # Use SRDF predefined "close" state
            gripper_group.set_named_target("close")
            gripper_plan = gripper_group.plan()
            
            if gripper_plan and gripper_plan.joint_trajectory.points:
                gripper_group.execute(gripper_plan, wait=True)
                print("Gripper closed!")
                
                rospy.sleep(2.0)
                
                print("\n=== Step 3: Opening gripper ===")
                
                # Check current gripper state
                current_joints = gripper_group.get_current_joint_values()
                print("Current gripper joint values:", current_joints)
                
                # Check if gripper is already open (all joints close to 0)
                open_threshold = 0.1
                is_already_open = all(abs(joint) < open_threshold for joint in current_joints)
                
                if is_already_open:
                    print("Gripper is already open! Skipping opening step.")
                    print("\n=== Demo completed successfully! ===")
                else:
                    gripper_group.set_start_state_to_current_state()
                    # Use named state instead of joint values
                    gripper_group.set_named_target("open")
                    gripper_plan = gripper_group.plan()
                    
                    if gripper_plan and gripper_plan.joint_trajectory.points:
                        gripper_group.execute(gripper_plan, wait=True)
                        print("Gripper opened!")
                        print("\n=== Demo completed successfully! ===")
                    else:
                        print("Gripper opening planning failed! Trying alternative approach...")
                        # Try with SRDF named state as fallback
                        gripper_group.set_named_target("open")
                        gripper_plan = gripper_group.plan()
                        if gripper_plan and gripper_plan.joint_trajectory.points:
                            gripper_group.execute(gripper_plan, wait=True)
                            print("Gripper opened!")
                            print("\n=== Demo completed successfully! ===")
                        else:
                            print("Gripper opening failed completely!")
            else:
                print("Gripper closing planning failed!")
        else:
            print("Arm planning failed! Target may be unreachable or in collision.")

    # Clean shutdown
    arm_group.stop()
    arm_group.clear_pose_targets()
    gripper_group.stop()
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
