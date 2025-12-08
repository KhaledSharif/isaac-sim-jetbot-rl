#!/bin/bash
# Launch RViz2 for Jetbot visualization
# Run this in a separate terminal AFTER starting Isaac Sim with ./run_ros2.sh
#
# Prerequisites:
#   sudo apt install -y ros-jazzy-rviz2 ros-jazzy-depth-image-proc

source /opt/ros/jazzy/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="$SCRIPT_DIR/jetbot.rviz"

if [ ! -f "$RVIZ_CONFIG" ]; then
    echo "Error: RViz config not found at $RVIZ_CONFIG"
    exit 1
fi

echo "Starting RViz2 with Jetbot configuration..."
echo "Make sure Isaac Sim is running with ./run_ros2.sh"
echo ""
echo "QoS Note: Camera topics use Best Effort reliability (Isaac Sim default)"
echo ""

# Start depth_image_proc for point cloud conversion in background
echo "Starting depth-to-pointcloud converter..."
ros2 run depth_image_proc point_cloud_xyz_node \
    --ros-args \
    -r camera_info:=/jetbot/camera/depth/camera_info \
    -r image_rect:=/jetbot/camera/depth/image_raw \
    -r points:=/jetbot/camera/depth/points \
    -p use_sim_time:=true &

DEPTH_PID=$!

# Give it a moment to start
sleep 1

# Start RViz2
echo "Starting RViz2..."
rviz2 -d "$RVIZ_CONFIG" --ros-args -p use_sim_time:=true

# Cleanup depth node when RViz exits
echo "Cleaning up..."
kill $DEPTH_PID 2>/dev/null
