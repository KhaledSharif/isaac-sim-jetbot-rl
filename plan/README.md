# ROS Visual SLAM Integration Plan

This directory contains the multi-phase implementation plan for integrating ROS2 and NVIDIA cuVSLAM with the Isaac Sim Jetbot project.

## Overview

```
Phase 1: ROS2 Integration      Phase 2: RViz Visualization     Phase 3: cuVSLAM
┌─────────────────────┐       ┌─────────────────────────┐     ┌────────────────────────┐
│ • Install ROS2      │       │ • Install RViz2         │     │ • Install Isaac ROS    │
│ • Configure DDS     │  ──▶  │ • Create robot URDF     │ ──▶ │ • Add stereo cameras   │
│ • ROS2 Bridge       │       │ • Visualize topics      │     │ • cuVSLAM mapping      │
│ • Publish topics    │       │ • Point cloud view      │     │ • 3D map in RViz       │
└─────────────────────┘       └─────────────────────────┘     └────────────────────────┘
```

## Phases

| Phase | Document | Description | Estimated Effort |
|-------|----------|-------------|------------------|
| 1 | [phase_1_ros.md](./phase_1_ros.md) | ROS2 Humble installation and Isaac Sim bridge setup | 2-4 hours |
| 2 | [phase_2_rviz.md](./phase_2_rviz.md) | RViz2 visualization of camera, depth, and TF | 1-2 hours |
| 3 | [phase_3_cuvslam.md](./phase_3_cuvslam.md) | NVIDIA cuVSLAM integration for 3D mapping | 3-5 hours |

## Prerequisites

- Isaac Sim 5.0.0 installed and working
- Ubuntu 22.04 (recommended)
- NVIDIA GPU with CUDA support (compute capability 7.0+)
- NVIDIA drivers 525+
- 16GB+ RAM recommended

## Quick Start

After completing all phases, the typical workflow is:

```bash
# Terminal 1: Isaac Sim with ROS2 bridge
source ~/setup_ros2_isaacsim.sh
./run_ros2.sh --enable-stereo

# Terminal 2: cuVSLAM
source /opt/ros/humble/setup.bash
ros2 launch jetbot_slam cuvslam.launch.py

# Terminal 3: RViz visualization
source /opt/ros/humble/setup.bash
rviz2 -d ~/ros2_ws/src/jetbot_slam/rviz/cuvslam_view.rviz
```

## Expected Results

After completing all phases:

1. **Real-time 3D mapping**: Drive the Jetbot around and watch a 3D point cloud map build in RViz
2. **Visual odometry**: Accurate pose tracking using stereo cameras
3. **Loop closure**: Automatic drift correction when revisiting areas
4. **Map persistence**: Save and reload maps for future sessions

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Isaac Sim 5.0.0                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Jetbot Robot                                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │ │
│  │  │ Keyboard │  │  Stereo  │  │  Depth   │  │   Odometry   │   │ │
│  │  │ Control  │  │ Cameras  │  │  Camera  │  │   (wheels)   │   │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    ROS2 Bridge (OmniGraph)                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ ROS2 Topics
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           ROS2 Humble                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  /camera/left/* │  │ /camera/right/* │  │  /tf, /odom, /clock │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
│           │                    │                      │              │
│           └────────────────────┼──────────────────────┘              │
│                                ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    NVIDIA cuVSLAM (GPU)                         │ │
│  │  • Visual Odometry    • Loop Closure    • Graph Optimization   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                │                                     │
│                                ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                         RViz2                                   │ │
│  │  • 3D Landmark Map    • Trajectory Path    • Camera Feeds      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## ROS2 Topics Published

| Topic | Type | Source |
|-------|------|--------|
| `/jetbot/camera/rgb/image_raw` | `sensor_msgs/Image` | Phase 1 |
| `/jetbot/camera/depth/image_raw` | `sensor_msgs/Image` | Phase 1 |
| `/jetbot/camera/camera_info` | `sensor_msgs/CameraInfo` | Phase 1 |
| `/jetbot/odom` | `nav_msgs/Odometry` | Phase 1 |
| `/tf` | `tf2_msgs/TFMessage` | Phase 1 |
| `/camera/left/image_raw` | `sensor_msgs/Image` | Phase 3 |
| `/camera/right/image_raw` | `sensor_msgs/Image` | Phase 3 |
| `/visual_slam/vis/landmarks_cloud` | `sensor_msgs/PointCloud2` | Phase 3 |
| `/visual_slam/tracking/slam_path` | `nav_msgs/Path` | Phase 3 |

## Files Created

After completing all phases, the following files will be added:

```
isaac-sim-jetbot-keyboard/
├── src/
│   ├── ros2_bridge.py              # Phase 1: ROS2 OmniGraph setup
│   └── stereo_camera_setup.py      # Phase 3: Stereo camera rig
├── run_ros2.sh                      # Phase 1: ROS2-enabled launch script
├── launch_rviz.sh                   # Phase 2: RViz launch helper
├── launch_cuvslam.sh                # Phase 3: cuVSLAM launch helper
└── plan/
    ├── README.md                    # This file
    ├── phase_1_ros.md
    ├── phase_2_rviz.md
    └── phase_3_cuvslam.md

~/ros2_ws/src/
├── jetbot_description/
│   ├── urdf/jetbot.urdf.xacro      # Phase 2: Robot model
│   ├── launch/view_robot.launch.py
│   └── rviz/jetbot_view.rviz
└── jetbot_slam/
    ├── launch/cuvslam.launch.py    # Phase 3: cuVSLAM launch
    └── rviz/cuvslam_view.rviz
```

## Troubleshooting

See individual phase documents for detailed troubleshooting guides.

### Common Issues

1. **No ROS2 topics**: Ensure Isaac Sim ROS2 bridge extension is enabled
2. **TF errors in RViz**: Check that simulation is playing
3. **cuVSLAM not tracking**: Ensure sufficient texture in scene, move slowly
4. **Performance issues**: Reduce image resolution, disable visualization topics

## References

- [Isaac Sim ROS2 Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/ros2_tutorials/ros2_landing_page.html)
- [NVIDIA Isaac ROS](https://nvidia-isaac-ros.github.io/)
- [cuVSLAM Documentation](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
