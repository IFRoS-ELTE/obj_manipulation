#!/bin/bash

# Source the ROS environment
source /opt/ros/melodic/setup.bash

# Compile the source code
catkin_make
# source /stonefish_ws/devel/setup.bash

# Start the ROS master
# roscore &

# Launch terminal to execute commands
exec bash

# Use to force the use of the dedicated GPU
# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia