# Grasp Anything with UR3  

**Team:** Faran, Sherif, Pravin, Jim  

## Objective  
The goal of this project is to enable a UR3 robotic arm to autonomously **detect, grasp, and place previously unseen rigid objects** into a basket. Using RGB-D perception, grasp planning, motion planning, and feedback control, the system aims to generalize across a wide variety of object shapes, sizes, and materials.  

## System Overview  
The robotic pipeline integrates the following modules:  

1. **Perception** – Object detection and 6D pose estimation from RGB-D input (e.g., Intel RealSense).  
2. **Grasp Planning** – Candidate grasp generation (PCA-based and/or Contact-GraspNet) and stability evaluation.  
3. **Motion Planning** – ROS + MoveIt! trajectory generation for UR3 manipulator.  
4. **Execution** – Pick-and-place actions with collision-aware planning.  
5. **Feedback** – Visual or gripper-based monitoring, with retry mechanisms for grasp failure recovery.  

## Methodology  
- **Perception:** Mask R-CNN / YOLOv8-Seg for segmentation or point cloud clustering. Pose estimation via centroid + PCA orientation.  
- **Grasp Planning:** Generate and evaluate candidate grasps, or leverage pretrained grasp-affordance models (e.g., Contact-GraspNet).  
- **Motion Execution:** Pre-grasp → approach → grasp → lift → place in basket.  
- **Feedback:** Visual alignment correction and gripper-based failure detection.  

## Implementation Plan  
- **Week 1:** Setup UR3, RGB-D camera, ROS workspace.  
- **Week 2:** Perception module (segmentation + pose estimation).  
- **Week 3:** Grasp planning implementation.  
- **Week 4:** Motion planning and execution (MoveIt!).  
- **Week 5:** Feedback mechanisms.  
- **Week 6:** Testing with diverse unseen objects.  
- **Week 7:** Final integration, demo, and report.  

## Expected Outcome  
- Robust grasping of arbitrary objects with the UR3 manipulator.  
- Generalizable and modular pipeline adaptable to other robots and tasks.  
- Quantitative evaluation of grasp success rate, task time, and recovery ability.  




# Docker Setup

This project uses Docker with NVIDIA GPU support for ROS1 Melodic development.

### Prerequisites
- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU drivers
- X11 forwarding support (for GUI applications)

### Building the Docker Image

1. **Build the ROS1 Melodic image:**
   ```bash
   chmod +x ./docker/build_image.sh
   ./docker/build_image.sh
   ```
   This creates a Docker image named `nvidia_melodic` with:
   - Ubuntu 18.04 base
   - NVIDIA CUDA 10.1 support
   - ROS1 Melodic desktop full
   - Build tools (gcc, cmake, etc.)
   - Python3 and ROS development tools

### Running the Docker Container

1. **Deploy and enter the container:**
   ```bash
   chmod +x ./docker/deploy_image.sh
   ./docker/deploy_image.sh
   ```

2. **What happens when you run the container:**
   - Mounts your current project directory to `/catkin_ws/src/obj_manipulation`
   - Sets up GPU access for CUDA applications
   - Enables X11 forwarding for GUI applications
   - Sources ROS Melodic environment
   - Builds the catkin workspace with `catkin_make`
   - Starts an interactive bash shell
### Opening new terminal in same container
   ```bash
   sudo docker ps
   sudo docker exec -it <container ID> bash
   ```
### Container Environment

**Working Directory:** `/catkin_ws`  
**Project Location:** `/catkin_ws/src/obj_manipulation`  
**ROS Environment:** Automatically sourced from `/opt/ros/melodic/setup.bash`
 
