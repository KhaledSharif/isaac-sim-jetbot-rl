"""Camera streaming module for Isaac Sim Jetbot.

This module provides GStreamer H264 RTP UDP streaming for the Jetbot's
front-facing camera. Streaming is lazy - the pipeline only runs when
the viewer is open to save CPU resources.

Dependencies:
    System: gstreamer1.0-tools, gstreamer1.0-plugins-base,
            gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad,
            gstreamer1.0-libav, python3-gi, gir1.2-gstreamer-1.0
    Python: PyGObject
"""

import logging
import numpy as np
import subprocess
from datetime import datetime
from typing import Optional

# Set up logging to file
_log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"/tmp/jetbot_{_log_timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger('CameraStreamer')
logger.info(f"Logging to {_log_file}")

# GStreamer imports with graceful fallback
GSTREAMER_IMPORT_ERROR = None
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    from gi.repository import Gst, GstApp, GLib
    GSTREAMER_AVAILABLE = True
    logger.info("GStreamer imports successful")
except (ImportError, ValueError) as e:
    GSTREAMER_AVAILABLE = False
    GSTREAMER_IMPORT_ERROR = str(e)
    Gst = None
    GstApp = None
    GLib = None
    logger.error(f"GStreamer import failed: {e}")

# Isaac Sim Camera - imported after SimulationApp is created
Camera = None


class CameraStreamer:
    """Handles Isaac Sim camera capture and GStreamer H264 RTP streaming.

    The streamer is designed for lazy operation - the GStreamer pipeline
    only runs when the viewer is open to minimize CPU usage.

    Attributes:
        WIDTH: Default camera width (640)
        HEIGHT: Default camera height (480)
        FPS: Default framerate (30)
        DEFAULT_PORT: Default UDP port (5600)
        CAMERA_POSITION: Camera offset from robot chassis
        CAMERA_ORIENTATION: Camera rotation quaternion
    """

    # Configuration defaults
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    DEFAULT_PORT = 5600
    DEFAULT_HOST = '127.0.0.1'

    # Camera prim path relative to robot (Jetbot has built-in camera)
    CAMERA_PRIM_SUFFIX = "/chassis/rgb_camera/jetbot_camera"

    def __init__(self, world, robot_prim_path: str, port: int = None,
                 width: int = None, height: int = None, fps: int = None):
        """Initialize the CameraStreamer.

        Args:
            world: Isaac Sim World object
            robot_prim_path: Prim path of the robot (e.g., "/World/Jetbot")
            port: UDP port for streaming (default: 5600)
            width: Camera width (default: 640)
            height: Camera height (default: 480)
            fps: Camera framerate (default: 30)
        """
        self.world = world
        self.robot_prim_path = robot_prim_path
        self.port = port or self.DEFAULT_PORT
        self.host = self.DEFAULT_HOST
        self.width = width or self.WIDTH
        self.height = height or self.HEIGHT
        self.fps = fps or self.FPS

        # Isaac Sim camera
        self.camera = None

        # GStreamer state
        self.pipeline = None
        self.appsrc = None
        self.is_streaming = False
        self._pts = 0
        self._duration = int(1e9 / self.fps)  # nanoseconds per frame

        # Viewer subprocess
        self.viewer_process = None

        # Frame counter for logging
        self._frame_count = 0
        self._failed_frame_count = 0

        logger.info(f"CameraStreamer initialized: robot={robot_prim_path}, "
                    f"port={self.port}, resolution={self.width}x{self.height}@{self.fps}fps")
        logger.info(f"GStreamer available: {GSTREAMER_AVAILABLE}")
        if GSTREAMER_IMPORT_ERROR:
            logger.error(f"GStreamer import error: {GSTREAMER_IMPORT_ERROR}")

    def create_camera(self) -> bool:
        """Get reference to the existing Jetbot camera.

        The Jetbot USD model already includes a camera at
        /chassis/rgb_camera/jetbot_camera. This method creates a
        Camera wrapper for that existing prim.

        Returns:
            True if camera was found successfully, False otherwise
        """
        global Camera

        logger.info("Getting reference to existing Jetbot camera...")

        try:
            # Import Camera class if not already imported
            if Camera is None:
                logger.debug("Importing isaacsim.sensors.camera.Camera...")
                from isaacsim.sensors.camera import Camera as _Camera
                Camera = _Camera
                logger.info("Camera class imported successfully")

            # Use the existing Jetbot camera
            camera_prim_path = f"{self.robot_prim_path}{self.CAMERA_PRIM_SUFFIX}"
            logger.info(f"Using existing camera at: {camera_prim_path}")

            # Create Camera wrapper for existing prim
            self.camera = Camera(
                prim_path=camera_prim_path,
                name="jetbot_camera",
                resolution=(self.width, self.height),
            )
            logger.info("Camera wrapper created for existing prim")

            # Add camera to scene for easy access
            logger.debug("Adding camera to world scene...")
            self.world.scene.add(self.camera)
            logger.info("Camera added to scene successfully")
            return True

        except ImportError as e:
            logger.error(f"Camera import failed: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Camera creation failed: {e}", exc_info=True)
            return False

    def initialize(self) -> bool:
        """Initialize the camera sensor for rendering.

        This should be called after world.reset() to ensure the camera
        is ready for capturing frames.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing camera sensor...")

        if self.camera is None:
            logger.error("Cannot initialize: camera is None")
            return False

        try:
            self.camera.initialize()
            logger.info("Camera sensor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}", exc_info=True)
            return False

    def _create_gstreamer_pipeline(self) -> bool:
        """Create the GStreamer H264 RTP UDP streaming pipeline.

        Returns:
            True if pipeline created successfully, False otherwise
        """
        logger.info("Creating GStreamer pipeline...")

        if not GSTREAMER_AVAILABLE:
            logger.error("GStreamer not available - cannot create pipeline")
            return False

        try:
            # Initialize GStreamer
            logger.debug("Initializing GStreamer...")
            Gst.init(None)
            logger.debug("GStreamer initialized")

            # Build pipeline string
            # appsrc -> queue -> videoconvert -> x264enc -> rtph264pay -> udpsink
            pipeline_str = (
                f'appsrc name=source emit-signals=False is-live=True '
                f'caps=video/x-raw,format=RGB,width={self.width},height={self.height},'
                f'framerate={self.fps}/1 ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'videoconvert ! '
                f'x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast '
                f'key-int-max={self.fps} ! '
                f'rtph264pay config-interval=1 pt=96 ! '
                f'udpsink host={self.host} port={self.port} sync=false'
            )
            logger.info(f"Pipeline string: {pipeline_str}")

            logger.debug("Parsing pipeline...")
            self.pipeline = Gst.parse_launch(pipeline_str)
            logger.debug("Pipeline parsed successfully")

            logger.debug("Getting appsrc element...")
            self.appsrc = self.pipeline.get_by_name('source')
            if self.appsrc is None:
                logger.error("Failed to get appsrc element from pipeline")
                return False
            logger.debug("appsrc element obtained")

            self.appsrc.set_property('format', Gst.Format.TIME)
            logger.debug("appsrc format set to TIME")

            # Reset PTS counter
            self._pts = 0

            logger.info("GStreamer pipeline created successfully")
            return True

        except Exception as e:
            logger.error(f"Pipeline creation failed: {e}", exc_info=True)
            self.pipeline = None
            self.appsrc = None
            return False

    def start_streaming(self) -> bool:
        """Start the GStreamer streaming pipeline.

        Returns:
            True if streaming started successfully, False otherwise
        """
        logger.info("Starting streaming...")

        if self.is_streaming:
            logger.warning("Already streaming")
            return True

        if self.pipeline is None:
            logger.debug("Pipeline is None, creating new pipeline...")
            if not self._create_gstreamer_pipeline():
                logger.error("Failed to create pipeline")
                return False

        try:
            logger.debug("Setting pipeline state to PLAYING...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            logger.debug(f"set_state returned: {ret}")

            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to start pipeline - state change returned FAILURE")
                return False

            self.is_streaming = True
            self._frame_count = 0
            self._failed_frame_count = 0
            logger.info("Streaming started successfully")
            return True

        except Exception as e:
            logger.error(f"Start streaming failed: {e}", exc_info=True)
            return False

    def stop_streaming(self) -> bool:
        """Stop the GStreamer streaming pipeline.

        Returns:
            True if streaming stopped successfully, False otherwise
        """
        logger.info(f"Stopping streaming... (frames sent: {self._frame_count}, failed: {self._failed_frame_count})")

        if not self.is_streaming:
            logger.debug("Not currently streaming")
            return True

        try:
            if self.pipeline is not None:
                logger.debug("Setting pipeline state to NULL...")
                self.pipeline.set_state(Gst.State.NULL)
                self.pipeline = None
                self.appsrc = None
                logger.debug("Pipeline stopped and cleared")

            self.is_streaming = False
            self._pts = 0
            logger.info("Streaming stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Stop streaming failed: {e}", exc_info=True)
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture an RGB frame from the Isaac Sim camera.

        Returns:
            RGB frame as numpy array (H, W, 3) uint8, or None if capture failed
        """
        if self.camera is None:
            logger.warning("capture_frame: camera is None")
            return None

        try:
            # Get RGBA image from camera
            rgba = self.camera.get_rgba()
            if rgba is None:
                self._failed_frame_count += 1
                if self._failed_frame_count <= 5 or self._failed_frame_count % 100 == 0:
                    logger.warning(f"capture_frame: get_rgba() returned None (count: {self._failed_frame_count})")
                return None

            # Log frame info periodically
            if self._frame_count == 0:
                logger.info(f"First frame captured: shape={rgba.shape}, dtype={rgba.dtype}")

            # Convert RGBA to RGB (drop alpha channel)
            return rgba[:, :, :3].copy()

        except Exception as e:
            self._failed_frame_count += 1
            logger.error(f"Frame capture failed: {e}", exc_info=True)
            return None

    def push_frame(self, frame: np.ndarray) -> bool:
        """Push an RGB frame to the GStreamer pipeline.

        Args:
            frame: RGB frame as numpy array (H, W, 3) uint8

        Returns:
            True if frame was pushed successfully, False otherwise
        """
        if not self.is_streaming or self.appsrc is None:
            if not self.is_streaming:
                logger.warning("push_frame: not streaming")
            if self.appsrc is None:
                logger.warning("push_frame: appsrc is None")
            return False

        try:
            # Convert frame to bytes
            data = frame.tobytes()

            # Create GStreamer buffer
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            # Set timestamp
            buf.pts = self._pts
            buf.duration = self._duration
            self._pts += self._duration

            # Push buffer to appsrc
            ret = self.appsrc.emit('push-buffer', buf)
            success = ret == Gst.FlowReturn.OK

            if success:
                self._frame_count += 1
                if self._frame_count == 1:
                    logger.info("First frame pushed successfully")
                elif self._frame_count % 300 == 0:  # Log every 10 seconds at 30fps
                    logger.debug(f"Frames pushed: {self._frame_count}")
            else:
                self._failed_frame_count += 1
                logger.warning(f"push-buffer returned: {ret}")

            return success

        except Exception as e:
            self._failed_frame_count += 1
            logger.error(f"Push frame failed: {e}", exc_info=True)
            return False

    def capture_and_stream(self) -> bool:
        """Capture a frame and stream it (convenience method).

        This method combines capture_frame() and push_frame() for
        use in the main simulation loop.

        Returns:
            True if frame was captured and streamed, False otherwise
        """
        if not self.is_streaming:
            return False

        frame = self.capture_frame()
        if frame is None:
            return False

        return self.push_frame(frame)

    def open_viewer(self) -> bool:
        """Launch the GStreamer viewer subprocess.

        Returns:
            True if viewer was opened successfully, False otherwise
        """
        logger.info("Opening viewer...")

        # Check if viewer is already running
        if self.viewer_process is not None:
            if self.viewer_process.poll() is None:
                logger.warning("Viewer is already running")
                return True

        try:
            # Build viewer pipeline command
            viewer_cmd = (
                f'gst-launch-1.0 udpsrc port={self.port} '
                f'caps="application/x-rtp,media=video,clock-rate=90000,'
                f'encoding-name=H264,payload=96" ! '
                f'rtph264depay ! avdec_h264 ! videoconvert ! autovideosink'
            )
            logger.info(f"Viewer command: {viewer_cmd}")

            # Launch viewer as subprocess
            logger.debug("Launching viewer subprocess...")
            self.viewer_process = subprocess.Popen(
                viewer_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Viewer launched with PID: {self.viewer_process.pid}")

            return True

        except Exception as e:
            logger.error(f"Open viewer failed: {e}", exc_info=True)
            return False

    def close_viewer(self) -> bool:
        """Close the GStreamer viewer subprocess.

        Returns:
            True if viewer was closed successfully, False otherwise
        """
        logger.info("Closing viewer...")

        if self.viewer_process is None:
            logger.debug("No viewer process to close")
            return True

        try:
            # Check if process is still running
            if self.viewer_process.poll() is None:
                logger.debug("Terminating viewer process...")
                self.viewer_process.terminate()
                try:
                    self.viewer_process.wait(timeout=2.0)
                    logger.debug("Viewer terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Viewer didn't terminate, killing...")
                    self.viewer_process.kill()
                    self.viewer_process.wait()
                    logger.debug("Viewer killed")
            else:
                returncode = self.viewer_process.returncode
                logger.debug(f"Viewer already exited with code: {returncode}")
                # Try to get stderr output
                try:
                    _, stderr = self.viewer_process.communicate(timeout=0.1)
                    if stderr:
                        logger.warning(f"Viewer stderr: {stderr.decode('utf-8', errors='ignore')}")
                except Exception:
                    pass

            self.viewer_process = None
            logger.info("Viewer closed")
            return True

        except Exception as e:
            logger.error(f"Close viewer failed: {e}", exc_info=True)
            return False

    def is_viewer_open(self) -> bool:
        """Check if the viewer subprocess is running.

        Returns:
            True if viewer is running, False otherwise
        """
        if self.viewer_process is None:
            return False
        return self.viewer_process.poll() is None

    def toggle(self) -> bool:
        """Toggle streaming and viewer on/off.

        If streaming is active, stop streaming and close viewer.
        If streaming is inactive, start streaming and open viewer.

        Returns:
            True if streaming is now active, False if inactive
        """
        logger.info(f"Toggle called, current streaming state: {self.is_streaming}")

        if self.is_streaming:
            # Stop streaming and close viewer
            logger.info("Toggling OFF - stopping streaming and closing viewer")
            self.stop_streaming()
            self.close_viewer()
            return False
        else:
            # Start streaming and open viewer
            logger.info("Toggling ON - starting streaming and opening viewer")
            if self.start_streaming():
                self.open_viewer()
                return True
            logger.error("Failed to start streaming")
            return False

    def cleanup(self):
        """Clean up all resources (pipeline, viewer, camera)."""
        logger.info("Cleanup called")
        self.stop_streaming()
        self.close_viewer()
        logger.info(f"Cleanup complete. Total frames: {self._frame_count}, failed: {self._failed_frame_count}")
        # Note: Camera cleanup is handled by Isaac Sim world
