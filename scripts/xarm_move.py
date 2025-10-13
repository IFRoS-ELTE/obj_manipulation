#!/usr/bin/env python2
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler

# Global variable to store received goal
goal_received = None


def goal_callback(msg):
    global goal_received
    goal_received = msg
    rospy.loginfo("Received new goal pose from Noetic!")


def main():
    global goal_received

    # Initialize MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_arm_subscriber", anonymous=True)

    # Subscriber for goal pose
    rospy.Subscriber("/goal_pose", PoseStamped, goal_callback)

    # Robot and scene objects
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("xarm6")

    rospy.loginfo("Planning frame: %s", group.get_planning_frame())
    rospy.loginfo("End effector link: %s", group.get_end_effector_link())

    group.set_pose_reference_frame(group.get_planning_frame())
    group.set_planner_id("RRTConnectkConfigDefault")
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(10)
    group.allow_replanning(True)
    group.set_goal_position_tolerance(0.01)
    group.set_goal_orientation_tolerance(0.05)
    group.set_max_velocity_scaling_factor(0.3)
    group.set_max_acceleration_scaling_factor(0.3)

    rospy.loginfo("Waiting for goal poses from Noetic...")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if goal_received is not None:
            # Move robot to the received goal
            target_pose = goal_received.pose

            rospy.loginfo("Planning to received pose...")

            group.set_start_state_to_current_state()
            group.set_pose_target(target_pose)
            plan = group.plan()

            # Execute plan if successful
            if plan and plan.joint_trajectory.points:
                group.execute(plan, wait=True)
                rospy.loginfo("Movement completed!")
            else:
                rospy.logwarn("Planning failed! Retrying with BKPIECE...")
                group.set_planner_id("BKPIECEkConfigDefault")
                group.set_goal_orientation_tolerance(0.15)
                group.set_start_state_to_current_state()
                group.set_pose_target(target_pose)
                plan_retry = group.plan()
                if plan_retry and plan_retry.joint_trajectory.points:
                    group.execute(plan_retry, wait=True)
                    rospy.loginfo("Movement completed on retry!")
                else:
                    rospy.logerr("Planning failed! Target may be unreachable.")

            # Clear goal after execution
            goal_received = None

        rate.sleep()

    # Clean shutdown
    group.stop()
    group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
