"""ROS2 Bridge for Isaac Sim Jetbot.

This module sets up OmniGraph nodes to publish sensor data to ROS2 topics:
- /jetbot/camera/rgb/image_raw (sensor_msgs/Image)
- /jetbot/camera/depth/image_raw (sensor_msgs/Image)
- /jetbot/camera/camera_info (sensor_msgs/CameraInfo)
- /jetbot/odom (nav_msgs/Odometry)
- /tf (tf2_msgs/TFMessage)
- /clock (rosgraph_msgs/Clock)

The OmniGraph-based approach uses Isaac Sim's internal ROS2 libraries,
avoiding Python version conflicts with system ROS2 installations.
"""

import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension


def enable_ros2_bridge_extension() -> bool:
    """Enable the Isaac Sim ROS2 bridge extension.

    Returns:
        True if extension was enabled successfully
    """
    try:
        enable_extension("isaacsim.ros2.bridge")
        print("[ROS2Bridge] Enabled isaacsim.ros2.bridge extension")
        return True
    except Exception as e:
        print(f"[ROS2Bridge] Failed to enable extension: {e}")
        return False


class ROS2Bridge:
    """Manages ROS2 OmniGraph nodes for Isaac Sim Jetbot."""

    # Default topic names (namespaced under /jetbot)
    NAMESPACE = "jetbot"
    RGB_TOPIC = f"/{NAMESPACE}/camera/rgb/image_raw"
    DEPTH_TOPIC = f"/{NAMESPACE}/camera/depth/image_raw"
    CAMERA_INFO_TOPIC = f"/{NAMESPACE}/camera/camera_info"
    DEPTH_INFO_TOPIC = f"/{NAMESPACE}/camera/depth/camera_info"  # For depth_image_proc
    ODOM_TOPIC = f"/{NAMESPACE}/odom"
    TF_TOPIC = "/tf"
    CLOCK_TOPIC = "/clock"

    # Frame IDs (matching Isaac Sim's TF tree: world -> chassis)
    ODOM_FRAME = "world"
    BASE_FRAME = "chassis"
    CAMERA_FRAME = "camera_link"

    def __init__(self, robot_prim_path: str = "/World/Jetbot"):
        """Initialize ROS2 Bridge.

        Args:
            robot_prim_path: USD prim path of the Jetbot robot
        """
        self.robot_prim_path = robot_prim_path
        self.camera_prim_path = f"{robot_prim_path}/chassis/rgb_camera/jetbot_camera"
        self.chassis_prim_path = f"{robot_prim_path}/chassis"

        self.clock_graph_path = "/ROS2ClockGraph"
        self.camera_graph_path = "/ROS2CameraGraph"
        self.tf_odom_graph_path = "/ROS2TFOdomGraph"
        self._graph_created = False

    def create_ros2_graph(self) -> bool:
        """Create the OmniGraph for ROS2 publishing.

        Returns:
            True if graph created successfully
        """
        if self._graph_created:
            print("[ROS2Bridge] Graph already created")
            return True

        # Enable ROS2 bridge extension first
        if not enable_ros2_bridge_extension():
            print("[ROS2Bridge] Cannot create graph - extension not available")
            return False

        try:
            # Create clock publisher graph
            if not self._create_clock_graph():
                return False

            # Create camera publisher graph
            if not self._create_camera_graph():
                return False

            # Create TF and odometry publisher graph
            if not self._create_tf_odom_graph():
                return False

            self._graph_created = True
            print(f"[ROS2Bridge] Created OmniGraph nodes")
            print(f"[ROS2Bridge] Publishing topics:")
            print(f"  - {self.RGB_TOPIC}")
            print(f"  - {self.DEPTH_TOPIC}")
            print(f"  - {self.CAMERA_INFO_TOPIC}")
            print(f"  - {self.ODOM_TOPIC}")
            print(f"  - {self.TF_TOPIC}")
            print(f"  - {self.CLOCK_TOPIC}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_clock_graph(self) -> bool:
        """Create the clock publisher graph.

        Returns:
            True if successful
        """
        try:
            keys = og.Controller.Keys

            og.Controller.edit(
                {"graph_path": self.clock_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    keys.CONNECT: [
                        # Execute clock publishing on each tick
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        # Connect simulation time to clock publisher
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [
                        ("PublishClock.inputs:topicName", self.CLOCK_TOPIC),
                    ],
                }
            )
            print(f"[ROS2Bridge] Created clock graph at {self.clock_graph_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create clock graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_camera_graph(self) -> bool:
        """Create the camera publisher graph.

        Returns:
            True if successful
        """
        try:
            keys = og.Controller.Keys

            # Create execution graph for camera using IsaacCreateRenderProduct directly
            og.Controller.edit(
                {"graph_path": self.camera_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("createRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                        ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("cameraHelperDepthInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ],
                    keys.CONNECT: [
                        # Execution flow
                        ("OnPlaybackTick.outputs:tick", "createRenderProduct.inputs:execIn"),
                        ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                        ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                        ("createRenderProduct.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                        ("createRenderProduct.outputs:execOut", "cameraHelperDepthInfo.inputs:execIn"),
                        # Render product path to camera helpers
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperDepthInfo.inputs:renderProductPath"),
                    ],
                    keys.SET_VALUES: [
                        # Render product settings
                        ("createRenderProduct.inputs:cameraPrim", self.camera_prim_path),
                        ("createRenderProduct.inputs:width", 640),
                        ("createRenderProduct.inputs:height", 480),
                        # RGB publisher
                        ("cameraHelperRgb.inputs:frameId", self.CAMERA_FRAME),
                        ("cameraHelperRgb.inputs:topicName", self.RGB_TOPIC),
                        ("cameraHelperRgb.inputs:type", "rgb"),
                        # Camera info publisher (for RGB)
                        ("cameraHelperInfo.inputs:frameId", self.CAMERA_FRAME),
                        ("cameraHelperInfo.inputs:topicName", self.CAMERA_INFO_TOPIC),
                        # Depth publisher
                        ("cameraHelperDepth.inputs:frameId", self.CAMERA_FRAME),
                        ("cameraHelperDepth.inputs:topicName", self.DEPTH_TOPIC),
                        ("cameraHelperDepth.inputs:type", "depth"),
                        # Depth camera info publisher (for depth_image_proc)
                        ("cameraHelperDepthInfo.inputs:frameId", self.CAMERA_FRAME),
                        ("cameraHelperDepthInfo.inputs:topicName", self.DEPTH_INFO_TOPIC),
                    ],
                }
            )

            print(f"[ROS2Bridge] Created camera graph at {self.camera_graph_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create camera graph: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_tf_odom_graph(self) -> bool:
        """Create TF and odometry publisher graph.

        Returns:
            True if successful
        """
        try:
            import usdrt.Sdf
            keys = og.Controller.Keys

            og.Controller.edit(
                {"graph_path": self.tf_odom_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                        ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                        ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ],
                    keys.CONNECT: [
                        # TF publishing
                        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                        # Odometry computation and publishing
                        ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishOdometry.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                        ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                        ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                        ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                        ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                    ],
                    keys.SET_VALUES: [
                        # TF settings - publish full robot articulation tree
                        ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(self.robot_prim_path)]),
                        ("PublishTF.inputs:topicName", self.TF_TOPIC),
                        # Odometry settings
                        ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(self.chassis_prim_path)]),
                        ("PublishOdometry.inputs:topicName", self.ODOM_TOPIC),
                        ("PublishOdometry.inputs:odomFrameId", self.ODOM_FRAME),
                        ("PublishOdometry.inputs:chassisFrameId", self.BASE_FRAME),
                    ],
                }
            )

            print(f"[ROS2Bridge] Created TF/Odom graph at {self.tf_odom_graph_path}")
            return True

        except Exception as e:
            print(f"[ROS2Bridge] Failed to create TF/Odom graph: {e}")
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
            "tf": self.TF_TOPIC,
            "clock": self.CLOCK_TOPIC,
        }

    def is_enabled(self) -> bool:
        """Check if ROS2 bridge graph has been created.

        Returns:
            True if graph is active
        """
        return self._graph_created
