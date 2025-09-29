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

## References  
- [Grasp Anything Documentation](https://airvlab.github.io/grasp-anything/docs/grasp-anything/)  
- [Contact-GraspNet (NVlabs)](https://github.com/NVlabs/contact_graspnet)  
