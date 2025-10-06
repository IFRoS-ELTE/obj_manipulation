#!/bin/bash

# For GPU use inside docker containers
xhost +local:docker

# Start the docker image
docker run -it --net=host --privileged --gpus all \
    --device /dev/dri \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$(pwd):/catkin_ws/src/obj_manipulation" \
    robot \
    bash