#!/bin/bash

# Build both docker images if any changes have been applied
docker build --network=host -f docker/Dockerfile_nvidia_melodic -t nvidia_melodic .
# docker build -f Dockerfile_stonefish -t stonefish ../../