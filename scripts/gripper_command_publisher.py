#!/usr/bin/env python

"""
Simple Gripper Command Publisher
Publishes gripper commands to the fake_gripper_controller
"""

import rospy
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class GripperCommandPublisher:
    def __init__(self):
        rospy.init_node('gripper_command_publisher', anonymous=True)
        
        # Publisher for gripper commands
        self.gripper_pub = rospy.Publisher('/fake_gripper_controller/command', 
                                         JointTrajectory, queue_size=1)
        
        rospy.loginfo("Gripper Command Publisher initialized")
        rospy.loginfo("Commands:")
        rospy.loginfo("  'open' - Open gripper")
        rospy.loginfo("  'close' - Close gripper")
        rospy.loginfo("  'half' - Half open gripper")
        rospy.loginfo("  'quit' - Exit")

    def publish_gripper_command(self, position):
        """Publish gripper command"""
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = ['finger_joint']
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = rospy.Duration(1.0)
        
        msg.points = [point]
        
        rospy.loginfo("Publishing gripper command: position = {}".format(position))
        self.gripper_pub.publish(msg)

    def run_interactive(self):
        """Run interactive mode"""
        while not rospy.is_shutdown():
            try:
                command = input("Enter gripper command (open/close/half/quit): ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'open':
                    self.publish_gripper_command(0.0)
                elif command == 'close':
                    self.publish_gripper_command(1.0)
                elif command == 'half':
                    self.publish_gripper_command(0.5)
                else:
                    rospy.logwarn("Invalid command. Use: open, close, half, or quit")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break

    def run_demo(self):
        """Run demo sequence"""
        rospy.loginfo("Running gripper demo sequence...")
        
        # Wait for system to be ready
        rospy.sleep(2)
        
        # Demo sequence
        self.publish_gripper_command(0.0)  # Open
        rospy.sleep(3)
        
        self.publish_gripper_command(1.0)  # Close
        rospy.sleep(3)
        
        self.publish_gripper_command(0.5)  # Half open
        rospy.sleep(3)
        
        self.publish_gripper_command(0.0)  # Open
        rospy.sleep(3)
        
        rospy.loginfo("Demo sequence completed")

def main():
    try:
        publisher = GripperCommandPublisher()
        
        if len(sys.argv) > 1 and sys.argv[1] == 'demo':
            publisher.run_demo()
        else:
            publisher.run_interactive()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("Gripper command publisher interrupted")
    except Exception as e:
        rospy.logerr("Error: {}".format(e))

if __name__ == '__main__':
    main()
