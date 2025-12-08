#!/bin/bash
# Launch Isaac Sim Jetbot with ROS2 bridge enabled
# Uses Isaac Sim's internal ROS2 libraries - DO NOT source system ROS2 here
#
# Usage:
#   ./run_ros2.sh                    # Run with ROS2 bridge enabled
#   ./run_ros2.sh --enable-recording # Run with ROS2 and recording enabled
#
# To verify topics, open a NEW terminal and run:
#   source /opt/ros/jazzy/setup.bash  # or your ROS2 distro
#   ros2 topic list

# Isaac Sim Python path
ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# IMPORTANT: Do NOT source system ROS2 (e.g., /opt/ros/jazzy/setup.bash)
# Isaac Sim has its own internal ROS2 implementation that handles DDS communication
# Mixing with system ROS2 causes Python version conflicts

# Set ROS_DOMAIN_ID to match your system ROS2 (default 0)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# Fast DDS configuration for cross-process communication (if file exists)
if [ -f ~/.ros/fastdds.xml ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml
fi

echo "Starting Isaac Sim with internal ROS2 bridge..."
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo ""
echo "To verify topics, open a NEW terminal and run:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  ros2 topic list"
echo ""

# Run with ROS2 enabled (extension is enabled programmatically in the code)
"$ISAAC_PYTHON" src/jetbot_keyboard_control.py --enable-ros2 "$@"
