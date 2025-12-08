# Phase 2: RViz Visualization

## Overview

This phase sets up RViz2 to visualize the ROS2 topics published from Isaac Sim. By the end of this phase, you will be able to see the robot model, camera feeds, depth data, odometry trail, and TF tree in real-time.

## Prerequisites

- Phase 1 completed: ROS2 bridge publishing topics
- All topics verified with `ros2 topic list`
- TF tree publishing correctly

## Goals

- [ ] Install RViz2 and visualization plugins
- [ ] Create robot URDF for RViz visualization
- [ ] Configure RViz displays for camera feeds
- [ ] Visualize depth as point cloud
- [ ] Display odometry trail
- [ ] Show TF coordinate frames
- [ ] Create saved RViz configuration
- [ ] Create launch file for easy startup

## Step 1: Install RViz2 and Dependencies

```bash
source /opt/ros/humble/setup.bash

# Install RViz2 and plugins
sudo apt install -y \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    ros-humble-rqt-image-view \
    ros-humble-rqt-tf-tree \
    ros-humble-image-transport-plugins \
    ros-humble-depth-image-proc \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui
```

## Step 2: Create Jetbot URDF

Create `ros2_ws/src/jetbot_description/urdf/jetbot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="jetbot">

  <!-- Properties -->
  <xacro:property name="chassis_length" value="0.14"/>
  <xacro:property name="chassis_width" value="0.10"/>
  <xacro:property name="chassis_height" value="0.05"/>
  <xacro:property name="wheel_radius" value="0.03"/>
  <xacro:property name="wheel_width" value="0.02"/>
  <xacro:property name="wheel_base" value="0.1125"/>
  <xacro:property name="caster_radius" value="0.01"/>

  <!-- Base Link -->
  <link name="base_link"/>

  <!-- Chassis -->
  <link name="chassis">
    <visual>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
      <material name="dark_grey">
        <color rgba="0.3 0.3 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
    </collision>
  </link>

  <joint name="base_to_chassis" type="fixed">
    <parent link="base_link"/>
    <child link="chassis"/>
    <origin xyz="0 0 ${wheel_radius + chassis_height/2}" rpy="0 0 0"/>
  </joint>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="chassis"/>
    <child link="left_wheel"/>
    <origin xyz="0 ${wheel_base/2} ${-chassis_height/2}" rpy="${-pi/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="chassis"/>
    <child link="right_wheel"/>
    <origin xyz="0 ${-wheel_base/2} ${-chassis_height/2}" rpy="${-pi/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Caster Wheel (for stability) -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="${caster_radius}"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${caster_radius}"/>
      </geometry>
    </collision>
  </link>

  <joint name="caster_joint" type="fixed">
    <parent link="chassis"/>
    <child link="caster_wheel"/>
    <origin xyz="${-chassis_length/2 + caster_radius} 0 ${-chassis_height/2 - wheel_radius + caster_radius}" rpy="0 0 0"/>
  </joint>

  <!-- Camera Mount -->
  <link name="camera_mount">
    <visual>
      <geometry>
        <box size="0.02 0.04 0.02"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 0.8 1.0"/>
      </material>
    </visual>
  </link>

  <joint name="camera_mount_joint" type="fixed">
    <parent link="chassis"/>
    <child link="camera_mount"/>
    <origin xyz="${chassis_length/2 - 0.01} 0 ${chassis_height/2 + 0.01}" rpy="0 0 0"/>
  </joint>

  <!-- Camera Link (optical frame) -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.01 0.03 0.03"/>
      </geometry>
      <material name="black"/>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="camera_mount"/>
    <child link="camera_link"/>
    <origin xyz="0.015 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Camera Optical Frame (Z forward, X right, Y down) -->
  <link name="camera_optical_frame"/>

  <joint name="camera_optical_joint" type="fixed">
    <parent link="camera_link"/>
    <child link="camera_optical_frame"/>
    <origin xyz="0 0 0" rpy="${-pi/2} 0 ${-pi/2}"/>
  </joint>

</robot>
```

## Step 3: Create ROS2 Package Structure

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create jetbot_description package
ros2 pkg create --build-type ament_cmake jetbot_description

# Create directories
mkdir -p jetbot_description/urdf
mkdir -p jetbot_description/launch
mkdir -p jetbot_description/rviz
mkdir -p jetbot_description/config
```

Create `ros2_ws/src/jetbot_description/launch/view_robot.launch.py`:

```python
"""Launch file for viewing Jetbot in RViz with Isaac Sim data."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package path
    pkg_path = get_package_share_directory('jetbot_description')

    # URDF file
    urdf_file = os.path.join(pkg_path, 'urdf', 'jetbot.urdf.xacro')

    # RViz config
    rviz_config = os.path.join(pkg_path, 'rviz', 'jetbot_view.rviz')

    # Robot description
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )

    return LaunchDescription([
        # Use sim time argument
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time from /clock'
        ),

        # Robot State Publisher (publishes robot model to /robot_description)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }]
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }]
        ),

        # Depth to PointCloud2 conversion
        Node(
            package='depth_image_proc',
            executable='point_cloud_xyz_node',
            name='depth_to_pointcloud',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            remappings=[
                ('camera_info', '/jetbot/camera/camera_info'),
                ('image_rect', '/jetbot/camera/depth/image_raw'),
                ('points', '/jetbot/camera/depth/points'),
            ]
        ),
    ])
```

## Step 4: Create RViz Configuration

Create `ros2_ws/src/jetbot_description/rviz/jetbot_view.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /TF1
        - /RobotModel1
        - /Camera RGB1
        - /Camera Depth1
        - /Odometry1
      Splitter Ratio: 0.5
  - Class: rviz_common/Views
    Name: Views

Visualization Manager:
  Class: ""
  Displays:
    # Grid
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Line Style:
        Line Width: 0.03
        Value: Lines
      Normal Cell Count: 0
      Cell Size: 1
      Color: 160; 160; 164
      Plane: XY
      Plane Cell Count: 20
      Reference Frame: odom

    # TF Frames
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 0.3
      Show Arrows: true
      Show Axes: true
      Show Names: true
      Update Interval: 0

    # Robot Model
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Enabled: true
      Description Topic:
        Value: /robot_description
      Alpha: 1

    # RGB Camera Image
    - Class: rviz_default_plugins/Image
      Name: Camera RGB
      Enabled: true
      Topic:
        Value: /jetbot/camera/rgb/image_raw
        Depth: 5
        Durability Policy: Volatile
        Reliability Policy: Best Effort
      Normalize Range: true
      Max Value: 1
      Min Value: 0
      Median window: 5

    # Depth Camera Image
    - Class: rviz_default_plugins/Image
      Name: Camera Depth
      Enabled: true
      Topic:
        Value: /jetbot/camera/depth/image_raw
        Depth: 5
        Durability Policy: Volatile
        Reliability Policy: Best Effort
      Normalize Range: true
      Max Value: 10
      Min Value: 0
      Median window: 5

    # Depth Point Cloud
    - Class: rviz_default_plugins/PointCloud2
      Name: Depth Points
      Enabled: true
      Topic:
        Value: /jetbot/camera/depth/points
        Depth: 5
        Durability Policy: Volatile
        Reliability Policy: Best Effort
      Size (Pixels): 2
      Size (m): 0.01
      Style: Points
      Color Transformer: AxisColor
      Axis: Z
      Use Fixed Frame: true

    # Odometry
    - Class: rviz_default_plugins/Odometry
      Name: Odometry
      Enabled: true
      Topic:
        Value: /jetbot/odom
        Depth: 5
        Durability Policy: Volatile
        Reliability Policy: Reliable
      Position Tolerance: 0.1
      Angle Tolerance: 0.1
      Keep: 100
      Shape:
        Alpha: 1
        Axes Length: 0.1
        Axes Radius: 0.01
        Color: 255; 25; 0
        Head Length: 0.03
        Head Radius: 0.01
        Shaft Length: 0.1
        Shaft Radius: 0.005
        Value: Arrow

    # Path (Odometry Trail)
    - Class: rviz_default_plugins/Path
      Name: Odom Path
      Enabled: true
      Topic:
        Value: /jetbot/path
      Color: 25; 255; 0
      Line Style: Lines
      Line Width: 0.03

  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: odom
    Frame Rate: 30

  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
    - Class: rviz_default_plugins/SetGoal
    - Class: rviz_default_plugins/PublishPoint

  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 3
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.5
      Target Frame: base_link
      Yaw: 0.8

Window Geometry:
  Height: 1000
  Width: 1500
  X: 100
  Y: 100
  Displays:
    collapsed: false
  Camera RGB:
    collapsed: false
  Camera Depth:
    collapsed: false
```

## Step 5: Update CMakeLists.txt

Update `ros2_ws/src/jetbot_description/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(jetbot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY
  urdf
  launch
  rviz
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
```

Update `ros2_ws/src/jetbot_description/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>jetbot_description</name>
  <version>0.1.0</version>
  <description>Jetbot robot description for RViz visualization</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>rviz2</exec_depend>
  <exec_depend>xacro</exec_depend>
  <exec_depend>depth_image_proc</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Step 6: Build and Source Workspace

```bash
cd ~/ros2_ws
colcon build --packages-select jetbot_description
source install/setup.bash

# Add to bashrc for persistence
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Step 7: Create Convenience Launch Script

Create `launch_rviz.sh` in your project root:

```bash
#!/bin/bash
# Launch RViz2 for Jetbot visualization

# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Launch RViz with Jetbot configuration
ros2 launch jetbot_description view_robot.launch.py use_sim_time:=true
```

## Step 8: Running the Visualization

### Terminal 1: Isaac Sim with ROS2 Bridge

```bash
source ~/setup_ros2_isaacsim.sh
./run_ros2.sh
```

### Terminal 2: RViz Visualization

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch jetbot_description view_robot.launch.py
```

## RViz Display Overview

| Display | Description |
|---------|-------------|
| **Grid** | Reference grid in odom frame |
| **TF** | Coordinate frames (odom, base_link, camera_link) |
| **RobotModel** | 3D Jetbot model from URDF |
| **Camera RGB** | Live RGB camera feed |
| **Camera Depth** | Live depth camera feed (normalized) |
| **Depth Points** | 3D point cloud from depth camera |
| **Odometry** | Odometry arrows showing robot motion |
| **Odom Path** | Trail of robot positions |

## Verification Checklist

- [ ] RViz opens without errors
- [ ] Robot model displays correctly
- [ ] TF frames visible (odom, base_link, camera_link)
- [ ] RGB camera image displays
- [ ] Depth camera image displays
- [ ] Point cloud visible in front of robot
- [ ] Odometry arrows update when robot moves
- [ ] Fixed frame set to "odom"

## Troubleshooting

### Robot Model Not Visible

1. Check robot_description topic:
   ```bash
   ros2 topic echo /robot_description --once
   ```

2. Verify URDF syntax:
   ```bash
   check_urdf $(ros2 pkg prefix jetbot_description)/share/jetbot_description/urdf/jetbot.urdf.xacro
   ```

### Camera Images Black/Missing

1. Ensure simulation is playing in Isaac Sim
2. Check image encoding:
   ```bash
   ros2 topic echo /jetbot/camera/rgb/image_raw --field encoding --once
   ```
3. Try `rqt_image_view`:
   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```

### Point Cloud Not Visible

1. Verify depth_image_proc node is running:
   ```bash
   ros2 node list | grep depth
   ```

2. Check point cloud topic:
   ```bash
   ros2 topic echo /jetbot/camera/depth/points --once
   ```

3. Ensure camera_info topic has valid intrinsics:
   ```bash
   ros2 topic echo /jetbot/camera/camera_info --once
   ```

### TF Frames Missing

1. View TF tree:
   ```bash
   ros2 run tf2_tools view_frames
   ```

2. Check specific transform:
   ```bash
   ros2 run tf2_ros tf2_echo odom base_link
   ```

## Success Criteria

- [ ] RViz displays robot model correctly
- [ ] All TF frames visible and connected
- [ ] RGB camera feed at ~30 fps
- [ ] Depth camera feed updating
- [ ] Point cloud rendered in 3D view
- [ ] Odometry trail follows robot movement
- [ ] No TF warnings in RViz

## Next Phase

Once RViz visualization is working correctly, proceed to [Phase 3: NVIDIA cuVSLAM Integration](./phase_3_cuvslam.md).

## References

- [RViz2 User Guide](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)
- [URDF Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)
- [Robot State Publisher](https://github.com/ros/robot_state_publisher)
- [depth_image_proc](https://github.com/ros-perception/image_pipeline/tree/humble/depth_image_proc)
