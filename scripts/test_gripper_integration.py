#!/usr/bin/env python3

"""
Test script for Scout XArm6 Gripper Integration
This script tests the gripper controller integration with MoveIt
"""

import rospy
import sys
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from moveit_commander import MoveGroupCommander, RobotCommander
from moveit_commander import PlanningSceneInterface

class GripperTester:
    def __init__(self):
        rospy.init_node('gripper_tester', anonymous=True)
        
        # Initialize MoveIt commander
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        
        # Wait for MoveIt to be ready
        rospy.loginfo("Waiting for MoveIt to be ready...")
        rospy.sleep(2)
        
        # Initialize move group
        try:
            self.move_group = MoveGroupCommander("xarm6")
            rospy.loginfo("MoveGroup 'xarm6' found")
        except Exception as e:
            rospy.logerr(f"Failed to initialize MoveGroup: {e}")
            sys.exit(1)
        
        # Gripper publisher
        self.gripper_pub = rospy.Publisher('/gripper_controller/command', 
                                        JointTrajectory, queue_size=1)
        
        # Joint state subscriber
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, 
                                              self.joint_state_callback)
        self.current_joint_states = None
        
        rospy.loginfo("GripperTester initialized successfully")

    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        self.current_joint_states = msg

    def test_gripper_control(self):
        """Test gripper control functionality"""
        rospy.loginfo("Testing gripper control...")
        
        # Test gripper open
        self.control_gripper(0.0)  # Open
        rospy.sleep(2)
        
        # Test gripper close
        self.control_gripper(1.0)  # Close
        rospy.sleep(2)
        
        # Test gripper half open
        self.control_gripper(0.5)  # Half open
        rospy.sleep(2)
        
        rospy.loginfo("Gripper control test completed")

    def control_gripper(self, position):
        """Control gripper position"""
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = ['drive_joint']
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = rospy.Duration(1.0)
        
        msg.points = [point]
        
        rospy.loginfo(f"Sending gripper command: position = {position}")
        self.gripper_pub.publish(msg)

    def test_moveit_integration(self):
        """Test MoveIt integration with gripper"""
        rospy.loginfo("Testing MoveIt integration...")
        
        # Get current pose
        current_pose = self.move_group.get_current_pose().pose
        rospy.loginfo(f"Current pose: {current_pose}")
        
        # Get current joint values
        current_joints = self.move_group.get_current_joint_values()
        rospy.loginfo(f"Current joints: {current_joints}")
        
        # Test planning
        try:
            # Set a target pose (slightly different from current)
            target_pose = current_pose
            target_pose.position.z += 0.1  # Move up 10cm
            
            self.move_group.set_pose_target(target_pose)
            
            # Plan the motion
            plan = self.move_group.plan()
            
            if plan[0]:  # If planning succeeded
                rospy.loginfo("Planning succeeded!")
                # Uncomment to execute: self.move_group.execute(plan[1])
            else:
                rospy.logwarn("Planning failed!")
                
        except Exception as e:
            rospy.logerr(f"MoveIt test failed: {e}")

    def check_topics(self):
        """Check if required topics are available"""
        rospy.loginfo("Checking required topics...")
        
        topics = [
            '/gripper_controller/command',
            '/joint_states',
            '/move_group/display_planned_path',
            '/move_group/goal',
            '/move_group/result'
        ]
        
        for topic in topics:
            try:
                # Check if topic exists
                topic_info = rospy.get_published_topics()
                topic_names = [info[0] for info in topic_info]
                
                if topic in topic_names:
                    rospy.loginfo(f"✓ Topic {topic} is available")
                else:
                    rospy.logwarn(f"✗ Topic {topic} is not available")
            except Exception as e:
                rospy.logerr(f"Error checking topic {topic}: {e}")

    def run_tests(self):
        """Run all tests"""
        rospy.loginfo("Starting gripper integration tests...")
        
        # Wait for system to be ready
        rospy.sleep(3)
        
        # Check topics
        self.check_topics()
        
        # Test MoveIt integration
        self.test_moveit_integration()
        
        # Test gripper control
        self.test_gripper_control()
        
        rospy.loginfo("All tests completed!")

def main():
    try:
        tester = GripperTester()
        tester.run_tests()
        
        # Keep the node running
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by user")
    except Exception as e:
        rospy.logerr(f"Test failed: {e}")

if __name__ == '__main__':
    main()

