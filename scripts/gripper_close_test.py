#!/usr/bin/env python
"""
Simple script to test closing gripper with incremental steps
"""

import rospy
import moveit_commander
from sensor_msgs.msg import JointState
import sys
import time

def monitor_effort():
    """Monitor gripper effort"""
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

def main():
    print("Simple Gripper Close Test")
    print("="*60)
    
    # Initialize
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('gripper_close_test')
    
    gripper_group = moveit_commander.MoveGroupCommander("gripper")
    effort_data = monitor_effort()
    
    # Get current positions for all joints
    current_all = gripper_group.get_current_joint_values()
    current_finger = current_all[3] if len(current_all) >= 4 else current_all[0]
    print("Current joint positions: {}".format(current_all))
    print("Current finger_joint position: {:.4f}".format(current_finger))
    
    # Test incremental closing
    print("\nClosing in steps of 0.05...")
    
    step_size = 0.05
    target_steps = 12  # Close to 0.6 total
    close_target = 1.2  # Max safe limit from SRDF
    
    try:
        for i in range(target_steps):
            # Calculate target for finger_joint
            finger_target = current_finger + (i + 1) * step_size
            
            # Don't exceed the close target
            if finger_target > close_target:
                finger_target = close_target
            
            # Calculate other joints based on mimic relationships
            # Joint order in SRDF: right_proximal_phalanx, left_proximal_phalanx, right_crank, finger_joint
            # From SRDF close state: proximal ~ 0.4355565 when finger = 0.5
            # So ratio is 0.4355565/0.5 = 0.8711
            proximal_target = finger_target 
            right_crank_target = finger_target  # 1:1 mimic
            
            # Order: [right_proximal, left_proximal, right_crank, finger_joint]
            joint_targets = [proximal_target, proximal_target, right_crank_target, finger_target]
            
            print("\nStep {}: finger={:.4f}, proximal={:.4f}".format(i+1, finger_target, proximal_target))
            
            gripper_group.set_joint_value_target(joint_targets)
            plan = gripper_group.plan()
            
            if plan and plan.joint_trajectory.points:
                gripper_group.execute(plan, wait=True)
                rospy.sleep(0.2)
                
                effort = effort_data['finger_joint']['effort']
                actual_pos = gripper_group.get_current_joint_values()
                print("  Actual position: {:.4f}, Effort: {:.2f} Nm".format(
                    actual_pos[3] if len(actual_pos) >= 4 else actual_pos[0], effort))
            else:
                print("  Planning failed!")
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    final_positions = gripper_group.get_current_joint_values()
    print("\nFinal status:")
    print("  Joint positions: {}".format(final_positions))
    print("  Effort: {:.2f} Nm".format(effort_data['finger_joint']['effort']))
    
    gripper_group.stop()
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Error: {}".format(e))
