#!/bin/bash
# Build both docker images if any changes have been applied
docker build --network=host -f docker/Dockerfile_nvidia_noetic -t nvidia_noetic .
docker build --network=host -f docker/Dockerfile_robot -t robot .
docker build --network=host -f docker/Dockerfile_nvidia_melodic -t nvidia_melodic .