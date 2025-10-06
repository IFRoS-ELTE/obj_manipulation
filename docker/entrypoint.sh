#!/bin/bash

# Source the ROS environment
source /opt/ros/noetic/setup.bash

# Compile the source code
catkin_make
source /catkin_ws/devel/setup.bash

# Launch terminal to execute commands
exec bash