#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import threading
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler

class MoveNode(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_node", anonymous=True)

        # MoveIt interfaces
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.MoveGroupCommander("xarm6")

        # Publishers + Subscribers
        self.goal_pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=10)
         
        rospy.Subscriber("/grasp_estimation_node/grasp_pose",
                         PoseStamped, self.plan_to_pose, queue_size=1)
        rospy.Subscriber("/move_node/allow_execution",
                         Bool, self.allow_execution_cb, queue_size=1)

        # MoveIt config
        print("Planning frame:", self.arm_group.get_planning_frame())
        print("End effector link:", self.arm_group.get_end_effector_link())
        self.arm_group.set_pose_reference_frame(self.arm_group.get_planning_frame())
        self.arm_group.set_planner_id("RRTConnectkConfigDefault")
        self.arm_group.set_planning_time(5.0)
        self.arm_group.set_num_planning_attempts(5)
        self.arm_group.allow_replanning(True)
        self.arm_group.set_goal_position_tolerance(0.01)
        self.arm_group.set_goal_orientation_tolerance(0.05)
        self.arm_group.set_max_velocity_scaling_factor(0.3)
        self.arm_group.set_max_acceleration_scaling_factor(0.3)

        # Threading 
        self._lock = threading.Lock()
        self._latest_plan = None

        rospy.loginfo("Move node ready, waiting for grasp poses and execution trigger.")

    # --------------------------- CALLBACKS ---------------------------
    def plan_to_pose(self, msg):
        """Receives a PoseStamped from grasp_estimation_node and plans a trajectory."""
        if not self._lock.acquire(False):
            rospy.logwarn("Planning/execution in progress, skipping pose.")
            return

        try:
            # Republish goal for RViz
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.arm_group.get_planning_frame()
            pose_stamped.header.stamp = rospy.Time.now()
            target_pose = msg.pose
            pose_stamped.pose = target_pose
            self.goal_pub.publish(pose_stamped)

            rospy.loginfo(
                "Planning to x=%.3f y=%.3f z=%.3f",
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z,
            )

            self._plan_with_rrtconnect(target_pose)

            
        finally:
            self._lock.release()

    def allow_execution_cb(self, msg):
        """Executes the last computed plan when Bool(data=True) is received."""
        if not msg.data:
            rospy.loginfo("allow_execution=False -> ignoring.")
            return

        if self._latest_plan is None:
            rospy.logwarn("Execution requested but no plan is available.")
            return

        if not self._lock.acquire(False):
            rospy.logwarn("Controller busy -> cannot execute trajectory.")
            return

        try:
            rospy.loginfo("Executing trajectory...")
            self.arm_group.execute(self._latest_plan, wait=True)
            rospy.loginfo("Execution complete.")
            self._latest_plan = None
        finally:
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            self._lock.release()

    # --------------------------- PLANNERS ---------------------------
    def _plan_with_rrtconnect(self, target_pose):
        self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_pose_target(target_pose)

        plan = self.arm_group.plan()

        if plan and plan.joint_trajectory.points:
            rospy.loginfo("Primary planner succeeded.")
            self._latest_plan = plan
        else:
            rospy.logwarn("Primary planner failed")

 
if __name__ == "__main__":
    node = MoveNode()
    rospy.spin()
    moveit_commander.roscpp_shutdown()
