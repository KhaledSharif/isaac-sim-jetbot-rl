"""Tests for ROS2 Bridge and RViz configuration.

This module tests the ROS2 bridge functionality (Phase 1) and
RViz configuration files (Phase 2) for the Isaac Sim Jetbot.
"""

import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# =============================================================================
# MOCK SETUP - Must be done BEFORE importing ros2_bridge
# =============================================================================

# Mock omni.graph.core module - use simple MagicMock, don't override Controller
mock_og = MagicMock()
# Just set the Keys constants we need (accessed via auto-created attributes)
mock_og.Controller.Keys.CREATE_NODES = "CREATE_NODES"
mock_og.Controller.Keys.CONNECT = "CONNECT"
mock_og.Controller.Keys.SET_VALUES = "SET_VALUES"
sys.modules['omni'] = MagicMock()
sys.modules['omni.graph'] = MagicMock()
sys.modules['omni.graph.core'] = mock_og

# Mock isaacsim modules
mock_isaacsim = MagicMock()
mock_extensions = MagicMock()
sys.modules['isaacsim'] = mock_isaacsim
sys.modules['isaacsim.core'] = MagicMock()
sys.modules['isaacsim.core.utils'] = MagicMock()
sys.modules['isaacsim.core.utils.extensions'] = mock_extensions

# Mock usdrt for TF/odom graph
mock_usdrt = MagicMock()
mock_usdrt_sdf = MagicMock()
mock_usdrt_sdf.Path = MagicMock(side_effect=lambda x: f"SdfPath({x})")
sys.modules['usdrt'] = mock_usdrt
sys.modules['usdrt.Sdf'] = mock_usdrt_sdf

# =============================================================================
# IMPORT MODULE UNDER TEST
# =============================================================================

from ros2_bridge import ROS2Bridge, enable_ros2_bridge_extension

# Get a reference to the actual og module imported by ros2_bridge
# This ensures we're testing the same object that ros2_bridge uses
import ros2_bridge
og_controller_edit = ros2_bridge.og.Controller.edit

# =============================================================================
# FIXTURES
# =============================================================================

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def mock_enable_extension():
    """Fixture to mock enable_extension function."""
    with patch('ros2_bridge.enable_extension') as mock:
        yield mock


@pytest.fixture
def bridge():
    """Fixture to create a fresh ROS2Bridge instance."""
    return ROS2Bridge()


@pytest.fixture
def bridge_custom_path():
    """Fixture to create ROS2Bridge with custom robot path."""
    return ROS2Bridge(robot_prim_path="/CustomWorld/MyRobot")


# =============================================================================
# TEST: enable_ros2_bridge_extension function
# =============================================================================

class TestEnableRos2BridgeExtension:
    """Tests for the enable_ros2_bridge_extension function."""

    def test_enable_ros2_bridge_extension_success(self, mock_enable_extension):
        """Test that function returns True when extension enables successfully."""
        mock_enable_extension.return_value = None  # No exception

        result = enable_ros2_bridge_extension()

        assert result is True
        mock_enable_extension.assert_called_once_with("isaacsim.ros2.bridge")

    def test_enable_ros2_bridge_extension_failure(self, mock_enable_extension):
        """Test that function returns False when extension fails to enable."""
        mock_enable_extension.side_effect = RuntimeError("Extension not found")

        result = enable_ros2_bridge_extension()

        assert result is False


# =============================================================================
# TEST: ROS2Bridge Initialization
# =============================================================================

class TestROS2BridgeInit:
    """Tests for ROS2Bridge initialization."""

    def test_ros2_bridge_init_default_path(self, bridge):
        """Test default robot prim path is /World/Jetbot."""
        assert bridge.robot_prim_path == "/World/Jetbot"

    def test_ros2_bridge_init_custom_path(self, bridge_custom_path):
        """Test custom robot prim path is stored correctly."""
        assert bridge_custom_path.robot_prim_path == "/CustomWorld/MyRobot"

    def test_ros2_bridge_prim_paths(self, bridge):
        """Test camera and chassis prim paths are derived correctly."""
        assert bridge.camera_prim_path == "/World/Jetbot/chassis/rgb_camera/jetbot_camera"
        assert bridge.chassis_prim_path == "/World/Jetbot/chassis"

    def test_ros2_bridge_custom_prim_paths(self, bridge_custom_path):
        """Test prim paths with custom robot path."""
        assert bridge_custom_path.camera_prim_path == "/CustomWorld/MyRobot/chassis/rgb_camera/jetbot_camera"
        assert bridge_custom_path.chassis_prim_path == "/CustomWorld/MyRobot/chassis"

    def test_ros2_bridge_graph_paths(self, bridge):
        """Test graph paths are set correctly."""
        assert bridge.clock_graph_path == "/ROS2ClockGraph"
        assert bridge.camera_graph_path == "/ROS2CameraGraph"
        assert bridge.tf_odom_graph_path == "/ROS2TFOdomGraph"

    def test_ros2_bridge_initial_state(self, bridge):
        """Test _graph_created is False initially."""
        assert bridge._graph_created is False


# =============================================================================
# TEST: ROS2Bridge Topic Constants
# =============================================================================

class TestROS2BridgeTopicConstants:
    """Tests for ROS2Bridge topic constants."""

    def test_topic_namespace(self):
        """Test NAMESPACE constant."""
        assert ROS2Bridge.NAMESPACE == "jetbot"

    def test_rgb_topic(self):
        """Test RGB_TOPIC constant."""
        assert ROS2Bridge.RGB_TOPIC == "/jetbot/camera/rgb/image_raw"

    def test_depth_topic(self):
        """Test DEPTH_TOPIC constant."""
        assert ROS2Bridge.DEPTH_TOPIC == "/jetbot/camera/depth/image_raw"

    def test_camera_info_topic(self):
        """Test CAMERA_INFO_TOPIC constant."""
        assert ROS2Bridge.CAMERA_INFO_TOPIC == "/jetbot/camera/camera_info"

    def test_depth_info_topic(self):
        """Test DEPTH_INFO_TOPIC constant for depth_image_proc compatibility."""
        assert ROS2Bridge.DEPTH_INFO_TOPIC == "/jetbot/camera/depth/camera_info"

    def test_odom_topic(self):
        """Test ODOM_TOPIC constant."""
        assert ROS2Bridge.ODOM_TOPIC == "/jetbot/odom"

    def test_tf_topic(self):
        """Test TF_TOPIC constant."""
        assert ROS2Bridge.TF_TOPIC == "/tf"

    def test_clock_topic(self):
        """Test CLOCK_TOPIC constant."""
        assert ROS2Bridge.CLOCK_TOPIC == "/clock"


# =============================================================================
# TEST: ROS2Bridge Frame ID Constants
# =============================================================================

class TestROS2BridgeFrameConstants:
    """Tests for ROS2Bridge frame ID constants."""

    def test_odom_frame(self):
        """Test ODOM_FRAME matches Isaac Sim's world frame."""
        assert ROS2Bridge.ODOM_FRAME == "world"

    def test_base_frame(self):
        """Test BASE_FRAME matches Isaac Sim's chassis frame."""
        assert ROS2Bridge.BASE_FRAME == "chassis"

    def test_camera_frame(self):
        """Test CAMERA_FRAME constant."""
        assert ROS2Bridge.CAMERA_FRAME == "camera_link"


# =============================================================================
# TEST: ROS2Bridge Graph Creation
# =============================================================================

class TestROS2BridgeGraphCreation:
    """Tests for ROS2Bridge graph creation methods."""

    def test_create_ros2_graph_success(self, bridge, mock_enable_extension):
        """Test successful graph creation returns True."""
        mock_enable_extension.return_value = None
        # Use the actual mock that ros2_bridge uses
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        result = bridge.create_ros2_graph()

        assert result is True
        assert bridge._graph_created is True
        # Should call og.Controller.edit 3 times (clock, camera, tf_odom)
        assert og_controller_edit.call_count == 3

    def test_create_ros2_graph_already_created(self, bridge, mock_enable_extension):
        """Test that graph creation is skipped if already created."""
        bridge._graph_created = True

        result = bridge.create_ros2_graph()

        assert result is True
        mock_enable_extension.assert_not_called()

    def test_create_ros2_graph_extension_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if extension fails."""
        mock_enable_extension.side_effect = RuntimeError("Extension not found")

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False

    def test_create_ros2_graph_clock_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if clock graph creation fails."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.side_effect = RuntimeError("OmniGraph error")

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False
        # Reset side_effect for subsequent tests
        og_controller_edit.side_effect = None

    def test_create_ros2_graph_camera_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if camera graph creation fails."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        # First call (clock) succeeds, second call (camera) fails
        og_controller_edit.side_effect = [None, RuntimeError("Camera graph error")]

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False
        # Reset side_effect for subsequent tests
        og_controller_edit.side_effect = None

    def test_create_ros2_graph_tf_odom_fails(self, bridge, mock_enable_extension):
        """Test graph creation fails if TF/odom graph creation fails."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        # First two calls succeed, third call fails
        og_controller_edit.side_effect = [None, None, RuntimeError("TF/Odom error")]

        result = bridge.create_ros2_graph()

        assert result is False
        assert bridge._graph_created is False
        # Reset side_effect for subsequent tests
        og_controller_edit.side_effect = None

    def test_create_clock_graph_parameters(self, bridge, mock_enable_extension):
        """Test clock graph creation uses correct parameters."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        # Get the first call (clock graph)
        assert len(og_controller_edit.call_args_list) >= 1
        clock_call = og_controller_edit.call_args_list[0]
        graph_config = clock_call[0][0]

        assert graph_config["graph_path"] == "/ROS2ClockGraph"
        assert graph_config["evaluator_name"] == "execution"

    def test_create_camera_graph_parameters(self, bridge, mock_enable_extension):
        """Test camera graph creation uses correct parameters."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        # Get the second call (camera graph)
        assert len(og_controller_edit.call_args_list) >= 2
        camera_call = og_controller_edit.call_args_list[1]
        graph_config = camera_call[0][0]

        assert graph_config["graph_path"] == "/ROS2CameraGraph"
        assert graph_config["evaluator_name"] == "execution"

    def test_create_tf_odom_graph_parameters(self, bridge, mock_enable_extension):
        """Test TF/odom graph creation uses correct parameters."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        # Get the third call (TF/odom graph)
        assert len(og_controller_edit.call_args_list) >= 3
        tf_odom_call = og_controller_edit.call_args_list[2]
        graph_config = tf_odom_call[0][0]

        assert graph_config["graph_path"] == "/ROS2TFOdomGraph"
        assert graph_config["evaluator_name"] == "execution"


# =============================================================================
# TEST: ROS2Bridge Helper Methods
# =============================================================================

class TestROS2BridgeHelpers:
    """Tests for ROS2Bridge helper methods."""

    def test_get_published_topics(self, bridge):
        """Test get_published_topics returns correct dictionary."""
        topics = bridge.get_published_topics()

        assert topics["rgb"] == "/jetbot/camera/rgb/image_raw"
        assert topics["depth"] == "/jetbot/camera/depth/image_raw"
        assert topics["camera_info"] == "/jetbot/camera/camera_info"
        assert topics["odom"] == "/jetbot/odom"
        assert topics["tf"] == "/tf"
        assert topics["clock"] == "/clock"

    def test_get_published_topics_has_all_keys(self, bridge):
        """Test get_published_topics contains all expected keys."""
        topics = bridge.get_published_topics()
        expected_keys = {"rgb", "depth", "camera_info", "odom", "tf", "clock"}

        assert set(topics.keys()) == expected_keys

    def test_is_enabled_false_initially(self, bridge):
        """Test is_enabled returns False before graph creation."""
        assert bridge.is_enabled() is False

    def test_is_enabled_true_after_creation(self, bridge, mock_enable_extension):
        """Test is_enabled returns True after successful graph creation."""
        mock_enable_extension.return_value = None
        og_controller_edit.reset_mock()
        og_controller_edit.return_value = None
        og_controller_edit.side_effect = None

        bridge.create_ros2_graph()

        assert bridge.is_enabled() is True


# =============================================================================
# TEST: RViz Configuration File
# =============================================================================

class TestRVizConfigFile:
    """Tests for rviz/jetbot.rviz configuration file."""

    @pytest.fixture
    def rviz_config_path(self):
        """Get path to RViz config file."""
        return PROJECT_ROOT / "rviz" / "jetbot.rviz"

    @pytest.fixture
    def rviz_config(self, rviz_config_path):
        """Load and parse RViz config file."""
        import yaml
        with open(rviz_config_path, 'r') as f:
            return yaml.safe_load(f)

    def test_rviz_config_exists(self, rviz_config_path):
        """Test that rviz/jetbot.rviz file exists."""
        assert rviz_config_path.exists(), f"RViz config not found at {rviz_config_path}"

    def test_rviz_config_valid_yaml(self, rviz_config_path):
        """Test that config file is valid YAML."""
        import yaml
        try:
            with open(rviz_config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert config is not None
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML: {e}")

    def test_rviz_config_fixed_frame(self, rviz_config):
        """Test Fixed Frame is 'world' (matching Isaac Sim TF)."""
        fixed_frame = rviz_config["Visualization Manager"]["Global Options"]["Fixed Frame"]
        assert fixed_frame == "world"

    def test_rviz_config_has_displays(self, rviz_config):
        """Test config has displays defined."""
        displays = rviz_config["Visualization Manager"]["Displays"]
        assert isinstance(displays, list)
        assert len(displays) > 0

    def _find_display(self, rviz_config, name):
        """Helper to find display by name."""
        displays = rviz_config["Visualization Manager"]["Displays"]
        for display in displays:
            if display.get("Name") == name:
                return display
        return None

    def test_rviz_config_rgb_topic(self, rviz_config):
        """Test RGB camera topic is correct."""
        rgb_display = self._find_display(rviz_config, "Camera RGB")
        assert rgb_display is not None, "Camera RGB display not found"
        assert rgb_display["Topic"]["Value"] == "/jetbot/camera/rgb/image_raw"

    def test_rviz_config_depth_topic(self, rviz_config):
        """Test Depth camera topic is correct."""
        depth_display = self._find_display(rviz_config, "Camera Depth")
        assert depth_display is not None, "Camera Depth display not found"
        assert depth_display["Topic"]["Value"] == "/jetbot/camera/depth/image_raw"

    def test_rviz_config_odom_topic(self, rviz_config):
        """Test Odometry topic is correct."""
        odom_display = self._find_display(rviz_config, "Odometry")
        assert odom_display is not None, "Odometry display not found"
        assert odom_display["Topic"]["Value"] == "/jetbot/odom"

    def test_rviz_config_pointcloud_topic(self, rviz_config):
        """Test Point cloud topic is correct."""
        points_display = self._find_display(rviz_config, "Depth Points")
        assert points_display is not None, "Depth Points display not found"
        assert points_display["Topic"]["Value"] == "/jetbot/camera/depth/points"

    def test_rviz_config_rgb_qos_best_effort(self, rviz_config):
        """Test RGB camera uses Best Effort QoS (Isaac Sim default)."""
        rgb_display = self._find_display(rviz_config, "Camera RGB")
        assert rgb_display is not None
        assert rgb_display["Topic"]["Reliability Policy"] == "Best Effort"

    def test_rviz_config_depth_qos_best_effort(self, rviz_config):
        """Test Depth camera uses Best Effort QoS (Isaac Sim default)."""
        depth_display = self._find_display(rviz_config, "Camera Depth")
        assert depth_display is not None
        assert depth_display["Topic"]["Reliability Policy"] == "Best Effort"

    def test_rviz_config_has_tf_display(self, rviz_config):
        """Test config includes TF display."""
        tf_display = self._find_display(rviz_config, "TF")
        assert tf_display is not None, "TF display not found in config"
        assert tf_display["Enabled"] is True

    def test_rviz_config_has_grid_display(self, rviz_config):
        """Test config includes Grid display."""
        grid_display = self._find_display(rviz_config, "Grid")
        assert grid_display is not None, "Grid display not found in config"
        assert grid_display["Enabled"] is True

    def test_rviz_config_has_odometry_display(self, rviz_config):
        """Test config includes Odometry display."""
        odom_display = self._find_display(rviz_config, "Odometry")
        assert odom_display is not None, "Odometry display not found in config"
        assert odom_display["Enabled"] is True


# =============================================================================
# TEST: View Jetbot Launch Script
# =============================================================================

class TestViewJetbotScript:
    """Tests for rviz/view_jetbot.sh launch script."""

    @pytest.fixture
    def script_path(self):
        """Get path to launch script."""
        return PROJECT_ROOT / "rviz" / "view_jetbot.sh"

    @pytest.fixture
    def script_content(self, script_path):
        """Read script content."""
        with open(script_path, 'r') as f:
            return f.read()

    def test_view_jetbot_script_exists(self, script_path):
        """Test that rviz/view_jetbot.sh file exists."""
        assert script_path.exists(), f"Launch script not found at {script_path}"

    def test_view_jetbot_script_executable(self, script_path):
        """Test script has executable permission."""
        mode = script_path.stat().st_mode
        assert mode & stat.S_IXUSR, "Script is not executable by owner"

    def test_view_jetbot_sources_ros2(self, script_content):
        """Test script sources ROS2 Jazzy setup."""
        assert "source /opt/ros/jazzy/setup.bash" in script_content

    def test_view_jetbot_launches_depth_proc(self, script_content):
        """Test script runs depth_image_proc node."""
        assert "depth_image_proc" in script_content
        assert "point_cloud_xyz_node" in script_content

    def test_view_jetbot_correct_depth_topics(self, script_content):
        """Test script remaps to correct depth topics."""
        # Should remap to depth camera_info for depth_image_proc
        assert "/jetbot/camera/depth/camera_info" in script_content
        assert "/jetbot/camera/depth/image_raw" in script_content
        assert "/jetbot/camera/depth/points" in script_content

    def test_view_jetbot_launches_rviz(self, script_content):
        """Test script launches rviz2."""
        assert "rviz2" in script_content

    def test_view_jetbot_uses_rviz_config(self, script_content):
        """Test script uses the jetbot.rviz config file."""
        assert "jetbot.rviz" in script_content

    def test_view_jetbot_use_sim_time(self, script_content):
        """Test script sets use_sim_time parameter."""
        assert "use_sim_time:=true" in script_content


# =============================================================================
# TEST: Topic Consistency Between ROS2Bridge and RViz Config
# =============================================================================

class TestTopicConsistency:
    """Tests to verify topic names match between ROS2Bridge and RViz config."""

    @pytest.fixture
    def rviz_config(self):
        """Load RViz config."""
        import yaml
        config_path = PROJECT_ROOT / "rviz" / "jetbot.rviz"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _find_display(self, rviz_config, name):
        """Helper to find display by name."""
        displays = rviz_config["Visualization Manager"]["Displays"]
        for display in displays:
            if display.get("Name") == name:
                return display
        return None

    def test_rgb_topic_consistency(self, rviz_config):
        """Test RGB topic in RViz matches ROS2Bridge constant."""
        rgb_display = self._find_display(rviz_config, "Camera RGB")
        assert rgb_display["Topic"]["Value"] == ROS2Bridge.RGB_TOPIC

    def test_depth_topic_consistency(self, rviz_config):
        """Test Depth topic in RViz matches ROS2Bridge constant."""
        depth_display = self._find_display(rviz_config, "Camera Depth")
        assert depth_display["Topic"]["Value"] == ROS2Bridge.DEPTH_TOPIC

    def test_odom_topic_consistency(self, rviz_config):
        """Test Odometry topic in RViz matches ROS2Bridge constant."""
        odom_display = self._find_display(rviz_config, "Odometry")
        assert odom_display["Topic"]["Value"] == ROS2Bridge.ODOM_TOPIC

    def test_fixed_frame_matches_odom_frame(self, rviz_config):
        """Test RViz Fixed Frame matches ROS2Bridge ODOM_FRAME."""
        fixed_frame = rviz_config["Visualization Manager"]["Global Options"]["Fixed Frame"]
        assert fixed_frame == ROS2Bridge.ODOM_FRAME
