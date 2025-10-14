#!/usr/bin/env python3

"""
Simple Gripper Test Script
Tests the gripper integration with the fixed launch file
"""

import rospy
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def test_gripper():
    """Test gripper functionality"""
    rospy.init_node('gripper_test', anonymous=True)
    
    # Wait for system to be ready
    rospy.loginfo("Waiting for system to be ready...")
    rospy.sleep(3)
    
    # Check if gripper topics exist
    try:
        topics = rospy.get_published_topics()
        topic_names = [topic[0] for topic in topics]
        
        gripper_topics = [topic for topic in topic_names if 'gripper' in topic]
        if gripper_topics:
            rospy.loginfo(f"Found gripper topics: {gripper_topics}")
        else:
            rospy.logwarn("No gripper topics found")
            
    except Exception as e:
        rospy.logerr(f"Error checking topics: {e}")
        return
    
    # Test gripper command
    try:
        # Try to publish to fake_gripper_controller
        pub = rospy.Publisher('/fake_gripper_controller/command', JointTrajectory, queue_size=1)
        
        # Wait for publisher to be ready
        rospy.sleep(1)
        
        # Create gripper command
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = ['finger_joint']
        
        point = JointTrajectoryPoint()
        point.positions = [0.5]  # Half open
        point.time_from_start = rospy.Duration(1.0)
        
        msg.points = [point]
        
        # Publish command
        rospy.loginfo("Publishing gripper command...")
        pub.publish(msg)
        
        rospy.loginfo("Gripper test completed successfully!")
        
    except Exception as e:
        rospy.logerr(f"Error testing gripper: {e}")

def main():
    try:
        test_gripper()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test failed: {e}")

if __name__ == '__main__':
    main()
