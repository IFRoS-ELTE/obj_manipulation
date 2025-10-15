 
# Object Manipulation with XArm6 on Scout
**Team:** Muhammad Faran Akram, Sherif Sameh, Pravin Oli, Jamin Rahman Jim  

## Dual-container ROS1 (Melodic + Noetic) setup

Run Melodic (MoveIt + drivers, Python 2) and Noetic (Python 3 workspace) side-by-side using Docker with shared ROS1 networking.

### Build images
```bash
cd docker
./build_image.sh
```

### Run a container
Use a single script with a distro argument:
```bash
cd docker
# Melodic (ROS master + MoveIt/drivers)
./deploy_image.sh melodic

# Noetic (Python 3 workspace)
./deploy_image.sh noetic
```

Inside the Melodic container, start roscore or your MoveIt/driver launch. In the Noetic container, `rostopic list` should show topics from Melodic.

Both containers use host networking and share `ROS_MASTER_URI` (default `http://127.0.0.1:11311`). Override if needed:
```bash
ROS_MASTER_URI=http://<host_ip>:11311 ./deploy_image.sh noetic
```

Notes:
- Start Melodic roscore before Noetic tools.
- Message/service definitions must match across Melodic/Noetic.
- GUI/RViz supported via `--gpus all` and X11 mount.

## Objective  
The goal of this project is to enable a UR3 robotic arm to autonomously **detect, grasp, and place previously unseen rigid objects** into a basket. Using RGB-D perception, grasp planning, motion planning, and feedback control, the system aims to generalize across a wide variety of object shapes, sizes, and materials.  

# Docker Setup
This project uses Docker with NVIDIA GPU support for ROS1 Noetic development.

### Prerequisites
- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU drivers
- X11 forwarding support (for GUI applications)

### Building the Docker Image

1. **Build the ROS1 Noetic image:**
   ```bash
   chmod +x ./docker/build_image.sh
   ./docker/build_image.sh
   ```
   This creates a Docker image named `robot` with:
   - Ubuntu 20.04 base
   - NVIDIA CUDA 11.4 support
   - ROS1 Noetic desktop full
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
   - Sources ROS Noetic environment
   - Builds the catkin workspace with `catkin_make`
   - Starts an interactive bash shell

### Opening another terminal in same container
   ```bash
   sudo docker ps
   sudo docker exec -it <container ID> bash
   ```

### Docker Compose Setup

Run the dual-container setup for object manipulation:

```bash
cd docker
docker compose build --no-cache
```

**Typical workflow for object manipulation:**
```bash
# Start containers
cd docker/
docker compose up -d

# or from root:
docker compose -f docker/docker-compose.yml up -d


# Terminal 1: Launch robot simulation
docker compose exec -it melodic bash #melodic is name of container

# Terminal 2: Run segmentation and manipulation code
docker compose exec -it noetic bash #noetic is name of container 
```

**Container usage:**
- **Melodic container**: Scout XArm6 robot, MoveIt planning, Gazebo simulation(#TODO))
- **Noetic container**: Instance segmentation, Python 3 manipulation scripts, object detection

**Quick commands:**
```bash
# Access robot container
docker compose exec -it <Container Name> bash

# Stop everything
docker compose down
```

### Container Environment

**Working Directory:** `/catkin_ws`  
**Project Location:** `/catkin_ws/src/obj_manipulation`  
**ROS Environment:** Automatically sourced from `/opt/ros/noetic/setup.bash`
 
## Build workspace

```bash
cd /catkin_ws
catkin_make
source /catkin_ws/devel/setup.bash
```

## Testing noetic melodic connection

```bash
#In melodic docker (Terminal-1)
cd /catkin_ws
catkin_make
source /catkin_ws/devel/setup.bash

roslaunch obj_manipulation scout_xarm_moveit.launch 

#In melodic docker (Terminal-2)
cd /catkin_ws
catkin_make
source /catkin_ws/devel/setup.bash

rosrun obj_manipulation xarm_move.py 

#In noetic docker (Terminal-1)
cd /catkin_ws
catkin_make
source /catkin_ws/devel/setup.bash

rosrun obj_manipulation xarm_moveit_noetic.py

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

## Unseen Object Instance Segmentation
To use the instance segmentation module, follow the instructions given inside the following [`README.md`](./obj_manipulation/segment/models/README.md) to download its pre-trained weights.
Then, follow the instructions inside the following [`README.md`](./tests/segment/examples/README.md) to verify that it works as expected.

## Grasp Estimation using Contact-GraspNet
Similar to the instance segmentation module, follow the instructions given inside the following [`README.md`](./obj_manipulation/grasp/models/README.md) to download its pre-trained weights.
Then, follow the instructions inside the following [`README.md`](./tests/grasp/examples/README.md) to verify that it works as expected.

## Troubleshooting
- If RViz/MoveIt cannot load gripper meshes like `package://dh_robotics_ag95_model/...`:
  - Ensure the terminal running the launch has the workspace sourced:
    ```bash
    source /opt/ros/melodic/setup.bash
    source /catkin_ws/devel/setup.bash
    rospack find dh_robotics_ag95_model
    ```
  - Rebuild if needed: `cd /catkin_ws && catkin_make`
