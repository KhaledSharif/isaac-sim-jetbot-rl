# Phase 3: NVIDIA cuVSLAM Integration

## Overview

This phase integrates NVIDIA's GPU-accelerated cuVSLAM (CUDA Visual SLAM) for real-time 3D mapping and localization. cuVSLAM provides high-performance visual odometry and SLAM using stereo cameras and optional IMU data.

## Prerequisites

- Phase 1 completed: ROS2 bridge publishing camera/odometry topics
- Phase 2 completed: RViz visualization working
- NVIDIA GPU with CUDA support (compute capability 7.0+)
- NVIDIA drivers 525+ installed
- Isaac ROS compatible with your system

## Goals

- [ ] Install Isaac ROS Visual SLAM (cuVSLAM)
- [ ] Configure stereo camera setup for cuVSLAM
- [ ] Launch cuVSLAM with Isaac Sim data
- [ ] Visualize SLAM output in RViz (landmarks, trajectory)
- [ ] Save and reload maps
- [ ] Achieve real-time 3D mapping while navigating

## cuVSLAM Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Isaac Sim                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Jetbot with Stereo Camera                  │    │
│  │    ┌──────────┐              ┌──────────┐              │    │
│  │    │  Left    │              │  Right   │              │    │
│  │    │  Camera  │              │  Camera  │              │    │
│  │    └────┬─────┘              └────┬─────┘              │    │
│  └─────────┼────────────────────────┼──────────────────────┘    │
│            │                        │                           │
│            ▼                        ▼                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   ROS2 Bridge                            │   │
│  │  /camera/left/image_raw    /camera/right/image_raw      │   │
│  │  /camera/left/camera_info  /camera/right/camera_info    │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Isaac ROS cuVSLAM                            │
│  ┌─────────────────┐     ┌─────────────────────────────────┐   │
│  │  Visual         │     │  Loop Closure &                  │   │
│  │  Odometry       │────▶│  Graph Optimization              │   │
│  │  (GPU)          │     │  (GPU)                           │   │
│  └─────────────────┘     └─────────────────────────────────┘   │
│           │                           │                         │
│           ▼                           ▼                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Published Topics:                                       │   │
│  │  - /visual_slam/tracking/odometry                        │   │
│  │  - /visual_slam/vis/landmarks_cloud                      │   │
│  │  - /visual_slam/vis/trajectory                           │   │
│  │  - /tf (map -> odom)                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RViz2                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  3D Map      │  │  Trajectory  │  │  Live Camera Feed    │  │
│  │  (Landmarks) │  │  Path        │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Install Isaac ROS

### Option A: Using Docker (Recommended)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Clone Isaac ROS Common
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git

# Clone Isaac ROS Visual SLAM
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git

# Run the Isaac ROS dev container
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common
./scripts/run_dev.sh
```

### Option B: Native Installation

```bash
# Add Isaac ROS apt repository
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:nvidia-isaac-ros/release

# Install Isaac ROS Visual SLAM
sudo apt install -y \
    ros-humble-isaac-ros-visual-slam \
    ros-humble-isaac-ros-stereo-image-proc \
    ros-humble-isaac-ros-image-proc
```

## Step 2: Add Stereo Camera to Jetbot

cuVSLAM requires stereo cameras. Update the ROS2 bridge to publish stereo image pairs.

Create `src/stereo_camera_setup.py`:

```python
"""Stereo camera setup for cuVSLAM integration.

This module adds a stereo camera rig to the Jetbot for use with
NVIDIA cuVSLAM visual SLAM.
"""

from pxr import Gf, UsdGeom
import omni.graph.core as og


class StereoCameraSetup:
    """Sets up stereo cameras for visual SLAM."""

    # Stereo camera configuration
    BASELINE = 0.06  # 6cm baseline (distance between cameras)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    CAMERA_FOV = 90  # degrees

    # Topic names
    LEFT_IMAGE_TOPIC = "/camera/left/image_raw"
    LEFT_INFO_TOPIC = "/camera/left/camera_info"
    RIGHT_IMAGE_TOPIC = "/camera/right/image_raw"
    RIGHT_INFO_TOPIC = "/camera/right/camera_info"

    def __init__(self, world, robot_prim_path: str = "/World/Jetbot"):
        """Initialize stereo camera setup.

        Args:
            world: Isaac Sim World object
            robot_prim_path: Prim path of the robot
        """
        self.world = world
        self.robot_prim_path = robot_prim_path
        self.stage = world.stage

        # Camera paths
        self.left_camera_path = f"{robot_prim_path}/chassis/stereo_camera/left_camera"
        self.right_camera_path = f"{robot_prim_path}/chassis/stereo_camera/right_camera"

    def create_stereo_cameras(self) -> bool:
        """Create stereo camera rig on the Jetbot.

        Returns:
            True if cameras created successfully
        """
        try:
            # Create stereo camera mount
            mount_path = f"{self.robot_prim_path}/chassis/stereo_camera"
            mount_xform = UsdGeom.Xform.Define(self.stage, mount_path)

            # Position mount at front of robot, above chassis
            mount_xform.AddTranslateOp().Set(Gf.Vec3d(0.07, 0, 0.04))

            # Create left camera
            left_camera = UsdGeom.Camera.Define(self.stage, self.left_camera_path)
            left_xform = UsdGeom.Xformable(left_camera.GetPrim())
            left_xform.AddTranslateOp().Set(Gf.Vec3d(0, self.BASELINE / 2, 0))
            left_xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))

            # Set left camera properties
            left_camera.GetFocalLengthAttr().Set(1.93)  # Approximate for 90 deg FOV
            left_camera.GetHorizontalApertureAttr().Set(3.6)
            left_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

            # Create right camera
            right_camera = UsdGeom.Camera.Define(self.stage, self.right_camera_path)
            right_xform = UsdGeom.Xformable(right_camera.GetPrim())
            right_xform.AddTranslateOp().Set(Gf.Vec3d(0, -self.BASELINE / 2, 0))
            right_xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))

            # Set right camera properties (same as left)
            right_camera.GetFocalLengthAttr().Set(1.93)
            right_camera.GetHorizontalApertureAttr().Set(3.6)
            right_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

            print(f"[StereoCameraSetup] Created stereo cameras:")
            print(f"  Left:  {self.left_camera_path}")
            print(f"  Right: {self.right_camera_path}")
            print(f"  Baseline: {self.BASELINE}m")

            return True

        except Exception as e:
            print(f"[StereoCameraSetup] Failed to create stereo cameras: {e}")
            return False

    def create_ros2_publishers(self) -> bool:
        """Create OmniGraph nodes for stereo image publishing.

        Returns:
            True if publishers created successfully
        """
        try:
            keys = og.Controller.Keys

            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": "/ROS2_Stereo_Graph", "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        # Execution
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

                        # Left camera
                        ("LeftRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("PublishLeftImage", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("PublishLeftInfo", "isaacsim.ros2.bridge.ROS2CameraHelper"),

                        # Right camera
                        ("RightRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("PublishRightImage", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("PublishRightInfo", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    keys.CONNECT: [
                        # Execution flow
                        ("OnPlaybackTick.outputs:tick", "LeftRenderProduct.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "RightRenderProduct.inputs:execIn"),

                        # Left camera chain
                        ("LeftRenderProduct.outputs:execOut", "PublishLeftImage.inputs:execIn"),
                        ("LeftRenderProduct.outputs:execOut", "PublishLeftInfo.inputs:execIn"),
                        ("LeftRenderProduct.outputs:renderProductPath", "PublishLeftImage.inputs:renderProductPath"),
                        ("LeftRenderProduct.outputs:renderProductPath", "PublishLeftInfo.inputs:renderProductPath"),
                        ("ROS2Context.outputs:context", "PublishLeftImage.inputs:context"),
                        ("ROS2Context.outputs:context", "PublishLeftInfo.inputs:context"),

                        # Right camera chain
                        ("RightRenderProduct.outputs:execOut", "PublishRightImage.inputs:execIn"),
                        ("RightRenderProduct.outputs:execOut", "PublishRightInfo.inputs:execIn"),
                        ("RightRenderProduct.outputs:renderProductPath", "PublishRightImage.inputs:renderProductPath"),
                        ("RightRenderProduct.outputs:renderProductPath", "PublishRightInfo.inputs:renderProductPath"),
                        ("ROS2Context.outputs:context", "PublishRightImage.inputs:context"),
                        ("ROS2Context.outputs:context", "PublishRightInfo.inputs:context"),
                    ],
                    keys.SET_VALUES: [
                        # Left camera
                        ("LeftRenderProduct.inputs:cameraPrim", self.left_camera_path),
                        ("LeftRenderProduct.inputs:width", self.CAMERA_WIDTH),
                        ("LeftRenderProduct.inputs:height", self.CAMERA_HEIGHT),
                        ("PublishLeftImage.inputs:type", "rgb"),
                        ("PublishLeftImage.inputs:topicName", self.LEFT_IMAGE_TOPIC),
                        ("PublishLeftImage.inputs:frameId", "left_camera_optical"),
                        ("PublishLeftInfo.inputs:type", "camera_info"),
                        ("PublishLeftInfo.inputs:topicName", self.LEFT_INFO_TOPIC),
                        ("PublishLeftInfo.inputs:frameId", "left_camera_optical"),

                        # Right camera
                        ("RightRenderProduct.inputs:cameraPrim", self.right_camera_path),
                        ("RightRenderProduct.inputs:width", self.CAMERA_WIDTH),
                        ("RightRenderProduct.inputs:height", self.CAMERA_HEIGHT),
                        ("PublishRightImage.inputs:type", "rgb"),
                        ("PublishRightImage.inputs:topicName", self.RIGHT_IMAGE_TOPIC),
                        ("PublishRightImage.inputs:frameId", "right_camera_optical"),
                        ("PublishRightInfo.inputs:type", "camera_info"),
                        ("PublishRightInfo.inputs:topicName", self.RIGHT_INFO_TOPIC),
                        ("PublishRightInfo.inputs:frameId", "right_camera_optical"),
                    ],
                }
            )

            print("[StereoCameraSetup] Created stereo camera ROS2 publishers")
            return True

        except Exception as e:
            print(f"[StereoCameraSetup] Failed to create ROS2 publishers: {e}")
            return False
```

## Step 3: Create cuVSLAM Launch File

Create `ros2_ws/src/jetbot_slam/launch/cuvslam.launch.py`:

```python
"""Launch file for NVIDIA cuVSLAM with Isaac Sim Jetbot."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap
from launch.conditions import IfCondition


def generate_launch_description():
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'enable_slam',
            default_value='true',
            description='Enable SLAM (mapping). Set to false for localization only.'
        ),
        DeclareLaunchArgument(
            'enable_imu',
            default_value='false',
            description='Enable IMU fusion'
        ),
        DeclareLaunchArgument(
            'map_path',
            default_value='',
            description='Path to load map from (empty for new map)'
        ),

        # cuVSLAM Node
        Node(
            package='isaac_ros_visual_slam',
            executable='visual_slam_node',
            name='visual_slam',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),

                # Frame configuration
                'map_frame': 'map',
                'odom_frame': 'odom',
                'base_frame': 'base_link',

                # Camera configuration
                'num_cameras': 2,
                'enable_rectified_pose': True,
                'enable_imu_fusion': LaunchConfiguration('enable_imu'),

                # SLAM configuration
                'enable_slam_visualization': True,
                'enable_landmarks_view': True,
                'enable_observations_view': True,

                # Performance tuning
                'rectified_images': False,  # Isaac Sim provides raw images
                'enable_debug_mode': False,
                'debug_dump_path': '/tmp/cuvslam_debug',

                # Visual odometry parameters
                'gyro_noise_density': 0.000244,
                'gyro_random_walk': 0.000019393,
                'accel_noise_density': 0.001862,
                'accel_random_walk': 0.003,
                'calibration_frequency': 200.0,

                # Image processing
                'image_jitter_threshold_ms': 34.0,  # ~30fps tolerance
                'img_height': 480,
                'img_width': 640,
            }],
            remappings=[
                # Stereo camera topics from Isaac Sim
                ('visual_slam/image_0', '/camera/left/image_raw'),
                ('visual_slam/camera_info_0', '/camera/left/camera_info'),
                ('visual_slam/image_1', '/camera/right/image_raw'),
                ('visual_slam/camera_info_1', '/camera/right/camera_info'),

                # IMU (if enabled)
                ('visual_slam/imu', '/jetbot/imu'),
            ]
        ),

        # Static TF: map -> odom (initial, cuVSLAM will update)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }]
        ),
    ])
```

## Step 4: Create cuVSLAM RViz Configuration

Create `ros2_ws/src/jetbot_slam/rviz/cuvslam_view.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays

Visualization Manager:
  Class: ""
  Displays:
    # Grid
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Cell Size: 1
      Color: 160; 160; 164
      Plane: XY
      Reference Frame: map

    # TF Frames
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true
      Marker Scale: 0.3
      Show Names: true
      Frames:
        map:
          Value: true
        odom:
          Value: true
        base_link:
          Value: true
        left_camera_optical:
          Value: true
        right_camera_optical:
          Value: true

    # Robot Model
    - Class: rviz_default_plugins/RobotModel
      Name: Robot Model
      Enabled: true
      Description Topic:
        Value: /robot_description

    # cuVSLAM Landmarks (3D Map)
    - Class: rviz_default_plugins/PointCloud2
      Name: SLAM Landmarks
      Enabled: true
      Topic:
        Value: /visual_slam/vis/landmarks_cloud
      Size (m): 0.02
      Color Transformer: FlatColor
      Color: 0; 255; 0  # Green
      Style: Spheres

    # cuVSLAM Observations (current frame features)
    - Class: rviz_default_plugins/PointCloud2
      Name: Observations
      Enabled: true
      Topic:
        Value: /visual_slam/vis/observations_cloud
      Size (m): 0.01
      Color Transformer: FlatColor
      Color: 255; 255; 0  # Yellow
      Style: Points

    # cuVSLAM Trajectory
    - Class: rviz_default_plugins/Path
      Name: SLAM Trajectory
      Enabled: true
      Topic:
        Value: /visual_slam/tracking/slam_path
      Color: 255; 0; 255  # Magenta
      Line Style: Lines
      Line Width: 0.02

    # Visual Odometry Path
    - Class: rviz_default_plugins/Path
      Name: VO Path
      Enabled: true
      Topic:
        Value: /visual_slam/tracking/vo_path
      Color: 0; 255; 255  # Cyan
      Line Style: Lines
      Line Width: 0.01

    # Loop Closure Markers
    - Class: rviz_default_plugins/MarkerArray
      Name: Loop Closures
      Enabled: true
      Topic:
        Value: /visual_slam/vis/loop_closure_cloud

    # Left Camera Feed
    - Class: rviz_default_plugins/Image
      Name: Left Camera
      Enabled: true
      Topic:
        Value: /camera/left/image_raw
      Normalize Range: true

    # Right Camera Feed
    - Class: rviz_default_plugins/Image
      Name: Right Camera
      Enabled: true
      Topic:
        Value: /camera/right/image_raw
      Normalize Range: true

    # Pose with Covariance
    - Class: rviz_default_plugins/PoseWithCovariance
      Name: SLAM Pose
      Enabled: true
      Topic:
        Value: /visual_slam/tracking/odometry
      Color: 255; 100; 0
      Shape: Arrow

  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
    Frame Rate: 30

  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
      Topic: /initialpose

  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 8
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Pitch: 0.7
      Yaw: 0.5
      Name: Orbit View
```

## Step 5: Create Complete Launch Script

Create `launch_cuvslam.sh`:

```bash
#!/bin/bash
# Launch complete cuVSLAM system with Isaac Sim

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NVIDIA cuVSLAM + Isaac Sim Jetbot ===${NC}"

# Check NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA GPU not detected. cuVSLAM requires NVIDIA GPU.${NC}"
    exit 1
fi

echo -e "${GREEN}GPU detected:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Source ROS2
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Set environment
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml

echo ""
echo -e "${YELLOW}Launch order:${NC}"
echo "1. Terminal 1: Isaac Sim with stereo cameras"
echo "   ./run_ros2.sh --enable-stereo"
echo ""
echo "2. Terminal 2: cuVSLAM"
echo "   ros2 launch jetbot_slam cuvslam.launch.py"
echo ""
echo "3. Terminal 3: RViz"
echo "   ros2 launch jetbot_slam cuvslam_rviz.launch.py"
echo ""

# Parse arguments
ACTION=${1:-"help"}

case $ACTION in
    "cuvslam")
        echo -e "${GREEN}Launching cuVSLAM...${NC}"
        ros2 launch jetbot_slam cuvslam.launch.py
        ;;
    "rviz")
        echo -e "${GREEN}Launching RViz for cuVSLAM...${NC}"
        rviz2 -d $(ros2 pkg prefix jetbot_slam)/share/jetbot_slam/rviz/cuvslam_view.rviz
        ;;
    "topics")
        echo -e "${GREEN}Checking cuVSLAM topics...${NC}"
        ros2 topic list | grep -E "(visual_slam|camera)"
        ;;
    *)
        echo "Usage: $0 {cuvslam|rviz|topics}"
        echo ""
        echo "Commands:"
        echo "  cuvslam  - Launch cuVSLAM node"
        echo "  rviz     - Launch RViz with cuVSLAM config"
        echo "  topics   - List cuVSLAM related topics"
        ;;
esac
```

## Step 6: Map Saving and Loading

### Save Current Map

```bash
# Call the SaveMap service
ros2 service call /visual_slam/save_map isaac_ros_visual_slam_interfaces/srv/SaveMap \
    "{map_path: '/home/user/maps/jetbot_map'}"
```

### Load Existing Map

```bash
# Launch with map path
ros2 launch jetbot_slam cuvslam.launch.py map_path:=/home/user/maps/jetbot_map
```

### Localize in Map

```bash
# Set initial pose estimate
ros2 topic pub /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"

# Call localization service
ros2 service call /visual_slam/localize_in_map isaac_ros_visual_slam_interfaces/srv/LocalizeInMap \
    "{map_path: '/home/user/maps/jetbot_map', pose: {position: {x: 0, y: 0, z: 0}, orientation: {w: 1}}}"
```

## Running the Complete System

### Terminal 1: Isaac Sim

```bash
source ~/setup_ros2_isaacsim.sh
./run_ros2.sh --enable-stereo
```

### Terminal 2: cuVSLAM

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch jetbot_slam cuvslam.launch.py
```

### Terminal 3: RViz

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
./launch_cuvslam.sh rviz
```

## cuVSLAM Topics Reference

| Topic | Type | Description |
|-------|------|-------------|
| `/visual_slam/tracking/odometry` | `nav_msgs/Odometry` | Visual odometry output |
| `/visual_slam/tracking/vo_pose` | `geometry_msgs/PoseStamped` | VO pose |
| `/visual_slam/tracking/slam_path` | `nav_msgs/Path` | Full SLAM trajectory |
| `/visual_slam/tracking/vo_path` | `nav_msgs/Path` | VO trajectory |
| `/visual_slam/vis/landmarks_cloud` | `sensor_msgs/PointCloud2` | 3D landmark map |
| `/visual_slam/vis/observations_cloud` | `sensor_msgs/PointCloud2` | Current observations |
| `/visual_slam/status` | `isaac_ros_visual_slam_interfaces/VisualSlamStatus` | SLAM status |

## Verification Checklist

- [ ] cuVSLAM node starts without errors
- [ ] `/visual_slam/tracking/odometry` publishing
- [ ] Landmarks visible in RViz (green points)
- [ ] Trajectory path growing as robot moves
- [ ] TF tree: map -> odom -> base_link
- [ ] Loop closures detected (if revisiting areas)
- [ ] Map can be saved and loaded

## Troubleshooting

### cuVSLAM Not Starting

1. Check GPU memory:
   ```bash
   nvidia-smi
   ```

2. Verify stereo topics:
   ```bash
   ros2 topic hz /camera/left/image_raw
   ros2 topic hz /camera/right/image_raw
   ```

3. Check camera_info has valid intrinsics:
   ```bash
   ros2 topic echo /camera/left/camera_info --once
   ```

### No Landmarks Generated

1. Ensure sufficient texture in scene (add more objects)
2. Check image quality in RViz
3. Move robot slowly for initial mapping
4. Verify stereo baseline is correct

### Tracking Lost

1. Move robot back to mapped area
2. Reduce motion speed
3. Check for motion blur in images
4. Add more lighting to scene

### Performance Issues

1. Reduce image resolution (320x240 for testing)
2. Disable visualization topics:
   ```python
   'enable_landmarks_view': False,
   'enable_observations_view': False,
   ```
3. Use dedicated GPU for cuVSLAM (if multi-GPU)

## Success Criteria

- [ ] cuVSLAM tracks robot pose in real-time
- [ ] 3D landmark map builds as robot explores
- [ ] Trajectory path visible in RViz
- [ ] Loop closures correct drift when revisiting areas
- [ ] Map can be saved and reloaded
- [ ] Localization works in saved map
- [ ] Frame rate > 20 fps

## Performance Metrics

Target performance on RTX 3080:
- Visual odometry: 30+ fps
- SLAM update: 10+ Hz
- Landmark density: 1000+ points per cubic meter
- Loop closure detection: < 100ms

## References

- [Isaac ROS Visual SLAM Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/index.html)
- [cuVSLAM Isaac Sim Tutorial](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_isaac_sim.html)
- [cuVSLAM Architecture](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html)
- [Isaac ROS GitHub Repository](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)
