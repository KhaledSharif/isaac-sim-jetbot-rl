# Phase 1: ROS2 Integration with Isaac Sim

## Overview

This phase establishes the foundational ROS2 bridge between Isaac Sim 5.0.0 and the ROS2 ecosystem. By the end of this phase, the Jetbot will publish essential sensor data and transforms to ROS2 topics.

## Prerequisites

- Isaac Sim 5.0.0 installed and working
- Ubuntu 22.04 (recommended for ROS2 Humble)
- NVIDIA GPU with updated drivers

## Goals

- [ ] Install ROS2 Humble
- [ ] Configure Fast DDS for Isaac Sim communication
- [ ] Enable Isaac Sim ROS2 Bridge extension
- [ ] Publish RGB camera images to ROS2
- [ ] Publish depth camera images to ROS2
- [ ] Publish camera intrinsics (CameraInfo)
- [ ] Publish TF transforms (odom -> base_link -> camera_link)
- [ ] Publish odometry messages
- [ ] Verify all topics with `ros2 topic list` and `ros2 topic echo`

## Step 1: Install ROS2 Humble

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS2 apt repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list'

# Install ROS2 Humble Desktop
sudo apt update
sudo apt install -y ros-humble-desktop

# Install additional packages
sudo apt install -y \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-tf2-tools \
    ros-humble-tf2-ros \
    python3-colcon-common-extensions \
    python3-rosdep

# Initialize rosdep
sudo rosdep init
rosdep update

# Add to bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Step 2: Configure Fast DDS for Isaac Sim

Isaac Sim requires specific DDS configuration for reliable communication.

```bash
# Create ROS directory
mkdir -p ~/.ros

# Create Fast DDS configuration file
cat > ~/.ros/fastdds.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>UdpTransport</transport_id>
            <type>UDPv4</type>
        </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="participant_profile" is_default_profile="true">
        <rtps>
            <userTransports>
                <transport_id>UdpTransport</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
        </rtps>
    </participant>
</profiles>
EOF

# Create environment setup script
cat > ~/setup_ros2_isaacsim.sh << 'EOF'
#!/bin/bash
# Environment setup for ROS2 + Isaac Sim

# Source ROS2
source /opt/ros/humble/setup.bash

# Set Fast DDS as RMW implementation
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml

# Unset LD_LIBRARY_PATH to avoid conflicts with Isaac Sim
# (Isaac Sim will set its own paths)
unset LD_LIBRARY_PATH

echo "ROS2 + Isaac Sim environment configured"
echo "RMW: $RMW_IMPLEMENTATION"
EOF

chmod +x ~/setup_ros2_isaacsim.sh
```

## Step 3: Create ROS2 Bridge Module

Create `src/ros2_bridge.py`:

```python
"""ROS2 Bridge for Isaac Sim Jetbot.

This module sets up OmniGraph nodes to publish sensor data to ROS2 topics:
- /jetbot/camera/rgb/image_raw (sensor_msgs/Image)
- /jetbot/camera/depth/image_raw (sensor_msgs/Image)
- /jetbot/camera/camera_info (sensor_msgs/CameraInfo)
- /jetbot/odom (nav_msgs/Odometry)
- /tf (tf2_msgs/TFMessage)
"""

import omni.graph.core as og
from typing import Optional


class ROS2Bridge:
    """Manages ROS2 OmniGraph nodes for Isaac Sim Jetbot."""

    # Default topic names (namespaced under /jetbot)
    NAMESPACE = "jetbot"
    RGB_TOPIC = f"/{NAMESPACE}/camera/rgb/image_raw"
    DEPTH_TOPIC = f"/{NAMESPACE}/camera/depth/image_raw"
    CAMERA_INFO_TOPIC = f"/{NAMESPACE}/camera/camera_info"
    ODOM_TOPIC = f"/{NAMESPACE}/odom"

    # Frame IDs
    BASE_FRAME = "base_link"
    ODOM_FRAME = "odom"
    CAMERA_FRAME = "camera_link"

    def __init__(self, robot_prim_path: str = "/World/Jetbot"):
        """Initialize ROS2 Bridge.

        Args:
            robot_prim_path: USD prim path of the Jetbot robot
        """
        self.robot_prim_path = robot_prim_path
        self.camera_prim_path = f"{robot_prim_path}/chassis/rgb_camera/jetbot_camera"
        self.chassis_prim_path = f"{robot_prim_path}/chassis"

        self.graph_path = "/ROS2Bridge"
        self._graph_created = False

    def create_ros2_graph(self) -> bool:
        """Create the OmniGraph for ROS2 publishing.

        Returns:
            True if graph created successfully
        """
        if self._graph_created:
            print("[ROS2Bridge] Graph already created")
            return True

        try:
            keys = og.Controller.Keys

            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": self.graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        # Tick source
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),

                        # ROS2 context (manages ROS2 node lifecycle)
                        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),

                        # Simulation time reader
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

                        # Clock publisher (required for use_sim_time)
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),

                        # Camera render product (captures camera output)
                        ("CameraRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),

                        # RGB image publisher
                        ("PublishRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),

                        # Depth image publisher
                        ("PublishDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),

                        # Camera info publisher
                        ("PublishCameraInfo", "isaacsim.ros2.bridge.ROS2CameraHelper"),

                        # Odometry computation
                        ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),

                        # Odometry publisher
                        ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),

                        # TF publisher for robot links
                        ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),

                        # TF publisher for odom -> base_link
                        ("PublishOdomTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    ],
                    keys.CONNECT: [
                        # Execution flow - main tick
                        ("OnPlaybackTick.outputs:tick", "ReadSimTime.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "CameraRenderProduct.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),

                        # Clock publishing
                        ("ReadSimTime.outputs:execOut", "PublishClock.inputs:execIn"),
                        ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),

                        # Camera publishing chain
                        ("CameraRenderProduct.outputs:execOut", "PublishRGB.inputs:execIn"),
                        ("CameraRenderProduct.outputs:execOut", "PublishDepth.inputs:execIn"),
                        ("CameraRenderProduct.outputs:execOut", "PublishCameraInfo.inputs:execIn"),
                        ("CameraRenderProduct.outputs:renderProductPath", "PublishRGB.inputs:renderProductPath"),
                        ("CameraRenderProduct.outputs:renderProductPath", "PublishDepth.inputs:renderProductPath"),
                        ("CameraRenderProduct.outputs:renderProductPath", "PublishCameraInfo.inputs:renderProductPath"),
                        ("ROS2Context.outputs:context", "PublishRGB.inputs:context"),
                        ("ROS2Context.outputs:context", "PublishDepth.inputs:context"),
                        ("ROS2Context.outputs:context", "PublishCameraInfo.inputs:context"),

                        # Odometry publishing
                        ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                        ("ComputeOdometry.outputs:execOut", "PublishOdomTF.inputs:execIn"),
                        ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                        ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                        ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                        ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                        ("ROS2Context.outputs:context", "PublishOdometry.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),

                        # Odometry TF (odom -> base_link)
                        ("ComputeOdometry.outputs:position", "PublishOdomTF.inputs:translation"),
                        ("ComputeOdometry.outputs:orientation", "PublishOdomTF.inputs:rotation"),
                        ("ROS2Context.outputs:context", "PublishOdomTF.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishOdomTF.inputs:timeStamp"),

                        # TF for robot links
                        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                        ("ROS2Context.outputs:context", "PublishTF.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [
                        # Camera render product
                        ("CameraRenderProduct.inputs:cameraPrim", self.camera_prim_path),
                        ("CameraRenderProduct.inputs:width", 640),
                        ("CameraRenderProduct.inputs:height", 480),

                        # RGB publisher
                        ("PublishRGB.inputs:type", "rgb"),
                        ("PublishRGB.inputs:topicName", self.RGB_TOPIC),
                        ("PublishRGB.inputs:frameId", self.CAMERA_FRAME),

                        # Depth publisher
                        ("PublishDepth.inputs:type", "depth"),
                        ("PublishDepth.inputs:topicName", self.DEPTH_TOPIC),
                        ("PublishDepth.inputs:frameId", self.CAMERA_FRAME),

                        # Camera info publisher
                        ("PublishCameraInfo.inputs:type", "camera_info"),
                        ("PublishCameraInfo.inputs:topicName", self.CAMERA_INFO_TOPIC),
                        ("PublishCameraInfo.inputs:frameId", self.CAMERA_FRAME),

                        # Odometry computation
                        ("ComputeOdometry.inputs:chassisPrim", self.chassis_prim_path),

                        # Odometry publisher
                        ("PublishOdometry.inputs:topicName", self.ODOM_TOPIC),
                        ("PublishOdometry.inputs:odomFrameId", self.ODOM_FRAME),
                        ("PublishOdometry.inputs:chassisFrameId", self.BASE_FRAME),

                        # Odom TF publisher
                        ("PublishOdomTF.inputs:parentFrameId", self.ODOM_FRAME),
                        ("PublishOdomTF.inputs:childFrameId", self.BASE_FRAME),

                        # Robot TF publisher
                        ("PublishTF.inputs:targetPrims", [self.robot_prim_path]),
                    ],
                }
            )

            self._graph_created = True
            print(f"[ROS2Bridge] Created OmniGraph at {self.graph_path}")
            print(f"[ROS2Bridge] Publishing topics:")
            print(f"  - {self.RGB_TOPIC}")
            print(f"  - {self.DEPTH_TOPIC}")
            print(f"  - {self.CAMERA_INFO_TOPIC}")
            print(f"  - {self.ODOM_TOPIC}")
            print(f"  - /tf")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_published_topics(self) -> dict:
        """Get dictionary of published topic names.

        Returns:
            Dict mapping topic type to topic name
        """
        return {
            "rgb": self.RGB_TOPIC,
            "depth": self.DEPTH_TOPIC,
            "camera_info": self.CAMERA_INFO_TOPIC,
            "odom": self.ODOM_TOPIC,
            "tf": "/tf",
        }
```

## Step 4: Integrate ROS2 Bridge into Main Controller

Modify `src/jetbot_keyboard_control.py` to add ROS2 support:

```python
# Add to imports (after SimulationApp creation)
ROS2Bridge = None

# Add to JetbotKeyboardController.__init__()
def __init__(self, ..., enable_ros2: bool = False):
    # ... existing initialization ...

    self.enable_ros2 = enable_ros2
    self.ros2_bridge = None

    # Initialize ROS2 bridge if enabled
    if self.enable_ros2:
        self._init_ros2_bridge()

def _init_ros2_bridge(self):
    """Initialize ROS2 bridge for publishing sensor data."""
    global ROS2Bridge

    try:
        from ros2_bridge import ROS2Bridge as _ROS2Bridge
        ROS2Bridge = _ROS2Bridge

        self.ros2_bridge = ROS2Bridge(
            robot_prim_path="/World/Jetbot"
        )

        if self.ros2_bridge.create_ros2_graph():
            self.tui.set_last_command("ROS2 bridge enabled")
            topics = self.ros2_bridge.get_published_topics()
            print(f"[ROS2] Publishing {len(topics)} topics")
        else:
            self.ros2_bridge = None
            self.enable_ros2 = False

    except ImportError as e:
        print(f"[Warning] ROS2 bridge not available: {e}")
        self.enable_ros2 = False

    except Exception as e:
        print(f"[Error] ROS2 bridge initialization failed: {e}")
        self.enable_ros2 = False

# Add command line argument in parse_args()
parser.add_argument(
    '--enable-ros2', action='store_true',
    help='Enable ROS2 bridge for sensor publishing'
)

# Update controller instantiation
controller = JetbotKeyboardController(
    ...,
    enable_ros2=args.enable_ros2,
)
```

## Step 5: Create Launch Script

Create `run_ros2.sh`:

```bash
#!/bin/bash
# Launch Isaac Sim Jetbot with ROS2 bridge enabled

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Set Fast DDS configuration
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml

# Isaac Sim Python path
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$HOME/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64}"

# Enable ROS2 bridge extension
EXTRA_ARGS="--enable isaacsim.ros2.bridge"

# Run with ROS2 enabled
"${ISAAC_SIM_PATH}/python.sh" $EXTRA_ARGS src/jetbot_keyboard_control.py --enable-ros2 "$@"
```

## Step 6: Verification

### Terminal 1: Launch Isaac Sim with ROS2

```bash
source ~/setup_ros2_isaacsim.sh
./run_ros2.sh
```

### Terminal 2: Verify ROS2 Topics

```bash
source /opt/ros/humble/setup.bash

# List all topics
ros2 topic list

# Expected output:
# /jetbot/camera/rgb/image_raw
# /jetbot/camera/depth/image_raw
# /jetbot/camera/camera_info
# /jetbot/odom
# /tf
# /tf_static
# /clock

# Check topic frequency
ros2 topic hz /jetbot/camera/rgb/image_raw

# Echo odometry
ros2 topic echo /jetbot/odom --once

# View TF tree
ros2 run tf2_tools view_frames
```

## Expected ROS2 Topics

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/jetbot/camera/rgb/image_raw` | `sensor_msgs/Image` | RGB camera images (640x480) |
| `/jetbot/camera/depth/image_raw` | `sensor_msgs/Image` | Depth images (32FC1 format) |
| `/jetbot/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsic parameters |
| `/jetbot/odom` | `nav_msgs/Odometry` | Robot odometry |
| `/tf` | `tf2_msgs/TFMessage` | Transform tree |
| `/clock` | `rosgraph_msgs/Clock` | Simulation clock |

## TF Tree Structure

```
odom
 └── base_link
      └── chassis
           ├── left_wheel
           ├── right_wheel
           └── rgb_camera
                └── camera_link
```

## Troubleshooting

### No Topics Published

1. Ensure ROS2 bridge extension is enabled:
   ```bash
   # Check in Isaac Sim: Window > Extensions > search "ros2"
   ```

2. Verify Fast DDS configuration:
   ```bash
   echo $FASTRTPS_DEFAULT_PROFILES_FILE
   cat ~/.ros/fastdds.xml
   ```

3. Check for conflicting LD_LIBRARY_PATH:
   ```bash
   unset LD_LIBRARY_PATH
   ```

### Topics Listed But No Data

1. Ensure simulation is playing (press Play in Isaac Sim)
2. Check topic echo with `--no-daemon` flag:
   ```bash
   ros2 topic echo /jetbot/camera/rgb/image_raw --no-daemon
   ```

### TF Tree Issues

1. Verify frames with:
   ```bash
   ros2 run tf2_ros tf2_echo odom base_link
   ```

2. Check for missing transforms:
   ```bash
   ros2 run tf2_tools view_frames
   evince frames.pdf
   ```

## Success Criteria

- [ ] `ros2 topic list` shows all expected topics
- [ ] `ros2 topic hz /jetbot/camera/rgb/image_raw` shows ~30 Hz
- [ ] `ros2 topic echo /jetbot/odom` shows position updates when robot moves
- [ ] TF tree shows odom -> base_link -> camera_link chain
- [ ] No errors in Isaac Sim console

## Next Phase

Once all topics are publishing correctly, proceed to [Phase 2: RViz Visualization](./phase_2_rviz.md).

## References

- [Isaac Sim ROS2 Bridge Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/py/source/extensions/isaacsim.ros2.bridge/docs/index.html)
- [Isaac Sim ROS2 Camera Tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera.html)
- [Isaac Sim TF and Odometry Tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_tf.html)
- [ROS2 Humble Installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
