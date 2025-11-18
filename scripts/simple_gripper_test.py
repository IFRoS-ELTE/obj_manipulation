#!/usr/bin/env python
"""
Simple gripper test - open/close and show effort
"""

import rospy
import moveit_commander
from sensor_msgs.msg import JointState
import sys
import time

class SimpleGripperTest:
    def __init__(self):
        """Initialize simple gripper tester"""
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('simple_gripper_test', anonymous=True)
        
        # Initialize MoveIt
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper")
        
        # Joint state subscriber
        self.joint_states = {'effort': 0.0, 'position': 0.0}
        
        def callback(msg):
            try:
                idx = msg.name.index('finger_joint')
                self.joint_states['effort'] = msg.effort[idx] if idx < len(msg.effort) else 0.0
                self.joint_states['position'] = msg.position[idx] if idx < len(msg.position) else 0.0
            except (ValueError, IndexError):
                pass
        
        rospy.Subscriber('/joint_states', JointState, callback)
        rospy.sleep(1)
        
        print("Simple Gripper Test Ready!")
    
    def show_status(self):
        """Show current gripper status"""
        print("\nCurrent Status:")
        print("  Effort: {:.2f} Nm".format(self.joint_states['effort']))
        print("  Position: {:.4f} rad".format(self.joint_states['position']))
    
    def open(self):
        """Open gripper"""
        print("\nOpening gripper...")
        try:
            self.gripper_group.set_named_target("open")
            plan = self.gripper_group.plan()
            if plan:
                self.gripper_group.execute(plan, wait=True)
                print("Gripper opened!")
                time.sleep(1)
                self.show_status()
                return True
        except Exception as e:
            print("Error: {}".format(e))
        return False
    
    def close(self):
        """Close gripper"""
        print("\nClosing gripper...")
        try:
            # Get current position first
            current_joints = self.gripper_group.get_current_joint_values()
            print("Current joints: {}".format(current_joints))
            
            # Try named target first
            self.gripper_group.set_start_state_to_current_state()
            self.gripper_group.set_named_target("close")
            plan = self.gripper_group.plan()
            
            if plan and plan.joint_trajectory.points:
                self.gripper_group.execute(plan, wait=True)
                print("Gripper closed!")
            else:
                # Fallback: try manual joint values
                print("Named target failed, trying manual values...")
                close_value = current_joints[0] + 0.1  # Close a bit
                self.gripper_group.set_joint_value_target([close_value])
                plan2 = self.gripper_group.plan()
                if plan2:
                    self.gripper_group.execute(plan2, wait=True)
                    print("Gripper partially closed!")
                else:
                    print("Failed to plan any motion")
                    return False
            
            time.sleep(1)
            self.show_status()
            return True
        except Exception as e:
            print("Error: {}".format(e))
        return False
    
    def monitor(self, duration=10):
        """Monitor effort for specified duration"""
        print("\nMonitoring effort for {} seconds...".format(duration))
        print("Press Ctrl+C to stop early\n")
        
        start = time.time()
        try:
            while time.time() - start < duration:
                sys.stdout.write("\rEffort: {:.2f} Nm  |  Position: {:.4f} rad".format(
                    self.joint_states['effort'], 
                    self.joint_states['position']
                ))
                sys.stdout.flush()
                rospy.sleep(0.1)
            sys.stdout.write("\n")
        except KeyboardInterrupt:
            print("\n")


def main():
    try:
        gripper = SimpleGripperTest()
        
        print("\n" + "="*60)
        print("Simple Gripper Test")
        print("="*60)
        print("Commands:")
        print("  1 - Open gripper")
        print("  2 - Close gripper")
        print("  3 - Show current status")
        print("  4 - Monitor effort (10s)")
        print("  5 - Exit")
        print("="*60)
        
        while not rospy.is_shutdown():
            try:
                choice = input("\nSelect (1-5): ").strip()
                
                if choice == "1":
                    gripper.open()
                elif choice == "2":
                    gripper.close()
                elif choice == "3":
                    gripper.show_status()
                elif choice == "4":
                    gripper.monitor()
                elif choice == "5":
                    break
                else:
                    print("Invalid choice")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print("Error: {}".format(e))
        
        gripper.gripper_group.stop()
        moveit_commander.roscpp_shutdown()
        
    except Exception as e:
        print("Error: {}".format(e))


if __name__ == '__main__':
    main()

