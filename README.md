 
# Object Manipulation with XArm6 on Scout
**Team:** Muhammad Faran Akram, Sherif Sameh, Pravin Oli, Jamin Rahman Jim  

### Dual-container ROS1 (Melodic + Noetic) setup

Run Melodic (MoveIt + drivers, Python 2) and Noetic (Python 3 workspace) side-by-side using Docker with shared ROS1 networking.
 
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

# With real hardware (Only in user, not in Robot)
roslaunch obj_manipulation scout_xarm_grip_moveit.launch use_real_hardware:=true robot_ip:=192.168.1.102
```

## On Real Robot Silvanus
```bash
roslaunch agx_xarm_bringup scout_xarm_moveit.launch use_real_hardware:=true
```

## Use gripper:
```bash 
rosrun agx_xarm_pick control_gripper
```
## Opening Rviz in docker:
```bash 
rviz -d src/obj_manipulation/rviz/silvanus.rviz
```

## Python MoveIt Commander
```bash
source /catkin_ws/devel/setup.bash
python src/obj_manipulation/scripts/xarm_move.py
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


## High-level flow:
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
## Using move_node
move_node will take the pose from grasp_estimation_node through `/grasp_estimation_node/grasp_pose` topic, show the planned trajectory and wait for execution signal on `/move_node/allow_execution`.

Make sure move group is running (either on real hardware or simulation).
1. Make sure executable is created, Inside docker:
```bash
chmod +x src/obj_manipulation/scripts/move_node.py 
``` 
2. run node (#TODO: Add node to launch file): 
```bash 
rosrun obj_manipulation move_node
```
 
**Subscribed Topics:**
- `/grasp_estimation_node/grasp_pose` - Receives target grasp poses (x, y, z, quaternion)
- `/move_node/allow_execution` - Receives execution trigger signal
 
**Execution Signal Topic:** `/move_node/allow_execution`
- **Message Type:** `std_msgs/Bool`
- **Format:** `data: true` to execute
- **To allow path execution:**
  ```bash
  rostopic pub /move_node/allow_execution std_msgs/Bool "data: true"
  ```

## Unseen Object Instance Segmentation
To use the instance segmentation module, follow the instructions given inside the following [`README.md`](./obj_manipulation/segment/models/README.md) to download its pre-trained weights.
Afterwards, you can follow the instructions inside the following [`README.md`](./tests/segment/examples/README.md) to verify that it works as expected.

Typically during normal operation, the instance segmentation module's output is used internally by the grasp predition node and is not exposed to the user. However, for debugging, you can launch the following node which performs instance segmentation and publishes the segmented image result to the topic `/instance_segmentation/seg_mask` as a `sensor_msgs/Image` ROS message.

```bash
rosrun obj_manipulation segmentation_node.py
```

## Grasp Estimation using Contact-GraspNet
Similar to the instance segmentation module, follow the instructions given inside the following [`README.md`](./obj_manipulation/grasp/models/README.md) to download its pre-trained weights.
Afterwards, you can follow the instructions inside the following [`README.md`](./tests/grasp/examples/README.md) to verify that it works as expected.

To use the grasp estimation module with real live data, launch the following node.

```bash
rosrun obj_manipulation grasp_estimation_node.py
```

The grasp estimation node always stores the latest received RGB and Depth images but does not perform grasp prediction until it is explicity triggered.
This design choice was taken since grasp prediction should not be a repeating operation that takes place at the same update rate of the sensors.
To trigger a grasp predicition attempt using the latest stored images, use the following command or explicity publish to this topic from another ROS node.

```bash
rostopic pub /grasp_estimation_node/trigger std_msgs/Bool "data: true"
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
