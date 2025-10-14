# Scout XArm6 Gripper MoveIt Launch Files

This directory contains launch files that integrate the `xarm6_gripper_moveit_config` with your scout_xarm setup, providing enhanced gripper control capabilities.

## Available Launch Files

### 1. `scout_xarm6_gripper_demo.launch`
**Purpose**: Quick demo with fake execution (simulation only)
**Use Case**: Testing and development without real hardware

```bash
roslaunch obj_manipulation scout_xarm6_gripper_demo.launch
```

**Arguments**:
- `debug`: Enable debug mode (default: false)
- `use_gui`: Use joint state publisher GUI (default: false)
- `use_rviz`: Launch RViz (default: true)
- `limited`: Limit joints to [-PI, PI] (default: false)

### 2. `scout_xarm6_gripper_real.launch`
**Purpose**: Real hardware integration with gripper support
**Use Case**: Controlling actual robot with gripper

```bash
roslaunch obj_manipulation scout_xarm6_gripper_real.launch robot_ip:=192.168.1.219
```

**Arguments**:
- `robot_ip`: Robot IP address (default: 192.168.1.219)
- `debug`: Enable debug mode (default: false)
- `use_rviz`: Launch RViz (default: true)
- `limited`: Limit joints to [-PI, PI] (default: false)
- `gripper_controller`: Enable gripper controller (default: true)

### 3. `scout_xarm6_gripper_gazebo.launch`
**Purpose**: Gazebo simulation with gripper
**Use Case**: Testing in Gazebo environment

```bash
roslaunch obj_manipulation scout_xarm6_gripper_gazebo.launch
```

**Arguments**:
- `debug`: Enable debug mode (default: false)
- `use_rviz`: Launch RViz (default: true)
- `limited`: Limit joints to [-PI, PI] (default: false)
- `gripper_controller`: Enable gripper controller (default: true)
- `gazebo_gui`: Gazebo GUI (default: true)
- `paused`: Start Gazebo paused (default: false)
- `world_name`: Gazebo world file (default: worlds/empty.world)

### 4. `scout_xarm6_gripper_moveit.launch`
**Purpose**: Comprehensive launch file with all options
**Use Case**: Advanced usage with multiple configuration options

```bash
# Simulation
roslaunch obj_manipulation scout_xarm6_gripper_moveit.launch

# Real hardware
roslaunch obj_manipulation scout_xarm6_gripper_moveit.launch use_real_hardware:=true robot_ip:=192.168.1.219

# Gazebo simulation
roslaunch obj_manipulation scout_xarm6_gripper_moveit.launch gazebo:=true
```

## Key Features

### Gripper Controller Integration
- **Arm Controller**: Controls joints 1-6 (`joint1` to `joint6`)
- **Gripper Controller**: Controls gripper joint (`drive_joint`)
- **Action Server**: `FollowJointTrajectory` for both arm and gripper

### MoveIt Configuration
- Uses `xarm6_gripper_moveit_config` for enhanced gripper support
- Includes proper SRDF configuration for gripper
- Supports both fake execution and real hardware control

### Controller Management
- **Fake Execution**: For simulation and testing
- **Real Execution**: For actual hardware control
- **Gazebo Integration**: For simulation environment

## Testing the Integration

### 1. Test with Demo Launch
```bash
roslaunch obj_manipulation scout_xarm6_gripper_demo.launch
```

### 2. Verify Gripper Topics
```bash
# Check gripper topics
rostopic list | grep gripper

# Monitor gripper commands
rostopic echo /gripper_controller/command

# Check joint states
rostopic echo /joint_states
```

### 3. Test Gripper Control
```bash
# Send gripper command
rostopic pub /gripper_controller/command trajectory_msgs/JointTrajectory "
header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
joint_names: ['drive_joint']
points:
- positions: [0.0]
  velocities: [0.0]
  accelerations: [0.0]
  effort: [0.0]
  time_from_start:
    secs: 1
    nsecs: 0"
```

## Troubleshooting

### Common Issues

1. **Gripper not responding**
   - Check if `gripper_controller` is enabled
   - Verify gripper driver is running
   - Check joint state publisher

2. **MoveIt planning fails**
   - Ensure robot description is loaded
   - Check SRDF configuration
   - Verify joint limits

3. **Real hardware connection issues**
   - Check robot IP address
   - Verify network connectivity
   - Check gripper driver status

### Debug Mode
Enable debug mode for detailed logging:
```bash
roslaunch obj_manipulation scout_xarm6_gripper_demo.launch debug:=true
```

## Integration with Your Code

### Python Example
```python
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def control_gripper(position):
    pub = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=1)
    
    msg = JointTrajectory()
    msg.joint_names = ['drive_joint']
    
    point = JointTrajectoryPoint()
    point.positions = [position]
    point.time_from_start = rospy.Duration(1.0)
    
    msg.points = [point]
    pub.publish(msg)
```

### C++ Example
```cpp
#include <trajectory_msgs/JointTrajectory.h>
#include <ros/ros.h>

void controlGripper(double position) {
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<trajectory_msgs::JointTrajectory>("/gripper_controller/command", 1);
    
    trajectory_msgs::JointTrajectory msg;
    msg.joint_names = {"drive_joint"};
    
    trajectory_msgs::JointTrajectoryPoint point;
    point.positions = {position};
    point.time_from_start = ros::Duration(1.0);
    
    msg.points = {point};
    pub.publish(msg);
}
```

## Next Steps

1. **Test the demo launch** to verify basic functionality
2. **Configure your robot IP** for real hardware testing
3. **Integrate gripper control** into your manipulation code
4. **Test with Gazebo** for simulation validation
5. **Deploy to real hardware** when ready

## Support

For issues or questions:
- Check ROS logs: `roslog`
- Verify topics: `rostopic list`
- Check parameters: `rosparam list`
- Monitor joint states: `rostopic echo /joint_states`
