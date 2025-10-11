#!/bin/bash

# Detect ROS distribution and source accordingly
if [ -f /opt/ros/noetic/setup.bash ]; then
    echo "Detected ROS Noetic"
    source /opt/ros/noetic/setup.bash
elif [ -f /opt/ros/melodic/setup.bash ]; then
    echo "Detected ROS Melodic"
    source /opt/ros/melodic/setup.bash
else
    echo "No ROS distribution found!"
fi

# Compile the source code if catkin workspace exists
if [ -d /catkin_ws/src ]; then
    echo "Compiling catkin workspace..."
    cd /catkin_ws
    
    # Install dependencies
    echo "Installing dependencies..."
    rosdep install --from-paths src --ignore-src -r -y
    
    # Build workspace
    echo "Building workspace..."
    catkin_make
    
    # Source the workspace
    echo "Sourcing workspace..."
    source devel/setup.bash
    
    echo "Catkin workspace built and sourced successfully!"
else
    echo "No catkin workspace found at /catkin_ws/src"
fi

# Launch terminal to execute commands
exec bash