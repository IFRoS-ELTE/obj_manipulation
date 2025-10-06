 
# Object Manipulation with XArm6 on Scout
**Team:** Muhammad Faran Akram, Sherif Sameh, Pravin Oli, Jamin Rahman Jim  

## Objective  
The goal of this project is to enable a UR3 robotic arm to autonomously **detect, grasp, and place previously unseen rigid objects** into a basket. Using RGB-D perception, grasp planning, motion planning, and feedback control, the system aims to generalize across a wide variety of object shapes, sizes, and materials.  

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
 
## Build workspace

```bash
cd /catkin_ws
catkin_make
source /catkin_ws/devel/setup.bash
```

## Launching robot

```bash
roslaunch obj_manipulation scout_xarm_moveit.launch
# With Gazebo simulation
roslaunch obj_manipulation scout_xarm_moveit.launch gazebo:=true

# With GPU Gazebo simulation
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia roslaunch obj_manipulation scout_xarm_moveit.launch gazebo:=true

# # With joint state publisher GUI
# roslaunch obj_manipulation scout_xarm_moveit.launch use_gui:=true

# With real hardware (when available)
roslaunch obj_manipulation scout_xarm_moveit.launch use_real_hardware:=true robot_ip:=192.168.1.102
```

## Python MoveIt Commander
```bash
source /catkin_ws/devel/setup.bash
python src/obj_manipulation/scripts/xarm_move.py
```

## On Real Robot
```bash
roslaunch agx_xarm_bringup scout_xarm_moveit.launch use_real_hardware:=true
```
## On local machine
```bash
source devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.102:11311  # robot IP
export ROS_IP=<local machine ip>
```
ensure roscore is visible:
```bash
rostopic list
```
launch python test script:
```bash 
python src/obj_manipulation/scripts/xarm_move.py
```


High-level flow:
```
Start
 ├─ ROS init (node, MoveIt)
 ├─ Configure group (frame, tolerances, time, attempts, scaling)
 ├─ Prompt user → (x,y,z)
 ├─ Build target pose + publish RViz marker
 ├─ Set start state → set pose target
 │   └─ OMPL plan
 │       ├─ Success → execute
 │       └─ Fail → Cartesian path
 │           ├─ fraction > 0.7 → execute
 │           └─ else → report failure
 └─ Stop, clear, shutdown
```

## Troubleshooting
- If RViz/MoveIt cannot load gripper meshes like `package://dh_robotics_ag95_model/...`:
  - Ensure the terminal running the launch has the workspace sourced:
    ```bash
    source /opt/ros/melodic/setup.bash
    source /catkin_ws/devel/setup.bash
    rospack find dh_robotics_ag95_model
    ```
  - Rebuild if needed: `cd /catkin_ws && catkin_make`
