# Phase 5: Complete cuVSLAM Integration & Testing Infrastructure

## Session Date: 2025-12-07

## Overview

Phase 5 transforms the current basic cuVSLAM integration into a complete SLAM simulation and testing platform. This includes ground truth comparison, performance metrics, automated testing scenarios, and parameter tuning infrastructure.

**Goal**: Make this repository a complete testbed for evaluating cuVSLAM performance under different conditions, hyperparameters, and scenarios.

---

## Current Integration Maturity: ~60%

| Component | Status | Completeness |
|-----------|--------|--------------|
| Core SLAM pipeline | ✅ Working | 100% |
| Stereo cameras (10cm baseline) | ✅ Working | 100% |
| TF tree (map→world→chassis→cameras) | ✅ Working | 100% |
| Ground truth odometry | ✅ Publishing | 100% |
| IMU sensor topic | ✅ Fixed (untested) | 90% |
| IMU fusion in cuVSLAM | ⚠️ Disabled | 0% |
| RViz SLAM visualization | ⚠️ Partial | 30% |
| Ground truth comparison | ❌ Missing | 0% |
| Error metrics (ATE/RPE) | ❌ Missing | 0% |
| Map save/load | ❌ Missing | 0% |
| Loop closure monitoring | ❌ Missing | 0% |
| Test scenarios | ❌ Missing | 0% |
| Parameter tuning | ❌ Missing | 0% |
| Performance profiling | ❌ Missing | 0% |

**Target**: 90%+ completeness by end of Phase 5

---

## Phase 5 Milestones

### Milestone 1: Ground Truth Comparison System (HIGH PRIORITY)

**Objective**: Create real-time SLAM accuracy metrics by comparing cuVSLAM estimates against Isaac Sim ground truth.

**New File**: `src/slam_evaluator.py`

**Implementation**:

```python
#!/usr/bin/env python3
"""SLAM accuracy evaluator node.

Subscribes to:
- /jetbot/odom (ground truth from Isaac Sim)
- /visual_slam/tracking/odometry (SLAM estimate from cuVSLAM)

Publishes:
- /slam/metrics (real-time error metrics)
- /slam/ground_truth_path (for RViz comparison)

Computes:
- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error)
- XYZ position drift over time
- Orientation error
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque

class SLAMEvaluator(Node):
    def __init__(self):
        super().__init__('slam_evaluator')

        # Subscribers
        self.sub_gt = self.create_subscription(
            Odometry, '/jetbot/odom', self.gt_callback, 10)
        self.sub_slam = self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry', self.slam_callback, 10)

        # Publishers
        self.pub_metrics = self.create_publisher(
            Float64MultiArray, '/slam/metrics', 10)
        self.pub_gt_path = self.create_publisher(
            Path, '/slam/ground_truth_path', 10)

        # Data storage
        self.gt_poses = deque(maxlen=1000)
        self.slam_poses = deque(maxlen=1000)

        # Timer for metric computation
        self.timer = self.create_timer(1.0, self.compute_metrics)

    def gt_callback(self, msg):
        """Store ground truth pose."""
        pose = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.gt_poses.append(pose)

        # Publish GT path for RViz
        path = Path()
        path.header = msg.header
        path.header.frame_id = 'map'
        for p in self.gt_poses:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = p[1]
            ps.pose.position.y = p[2]
            ps.pose.position.z = p[3]
            ps.pose.orientation.x = p[4]
            ps.pose.orientation.y = p[5]
            ps.pose.orientation.z = p[6]
            ps.pose.orientation.w = p[7]
            path.poses.append(ps)
        self.pub_gt_path.publish(path)

    def slam_callback(self, msg):
        """Store SLAM estimate pose."""
        pose = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.slam_poses.append(pose)

    def compute_metrics(self):
        """Compute and publish error metrics."""
        if len(self.gt_poses) < 2 or len(self.slam_poses) < 2:
            return

        # Align timestamps (find closest matches)
        aligned_gt, aligned_slam = self.align_trajectories()

        if len(aligned_gt) < 2:
            return

        # Compute ATE (Absolute Trajectory Error)
        ate = self.compute_ate(aligned_gt, aligned_slam)

        # Compute RPE (Relative Pose Error) over 1 meter
        rpe_trans, rpe_rot = self.compute_rpe(aligned_gt, aligned_slam, delta=1.0)

        # Compute current drift
        current_drift = np.linalg.norm(
            np.array(aligned_gt[-1][1:4]) - np.array(aligned_slam[-1][1:4])
        )

        # Publish metrics
        msg = Float64MultiArray()
        msg.data = [ate, rpe_trans, rpe_rot, current_drift,
                   float(len(aligned_gt)), float(len(self.gt_poses))]
        self.pub_metrics.publish(msg)

        # Log to console
        self.get_logger().info(
            f"ATE: {ate:.4f}m | RPE(trans): {rpe_trans:.4f}m | "
            f"RPE(rot): {rpe_rot:.4f}° | Drift: {current_drift:.4f}m"
        )

    def align_trajectories(self, max_dt=0.1):
        """Align GT and SLAM trajectories by timestamp."""
        aligned_gt = []
        aligned_slam = []

        for gt_pose in self.gt_poses:
            gt_t = gt_pose[0]
            # Find closest SLAM pose
            best_slam = min(self.slam_poses,
                          key=lambda s: abs(s[0] - gt_t),
                          default=None)
            if best_slam and abs(best_slam[0] - gt_t) < max_dt:
                aligned_gt.append(gt_pose)
                aligned_slam.append(best_slam)

        return aligned_gt, aligned_slam

    def compute_ate(self, gt_traj, slam_traj):
        """Compute Absolute Trajectory Error (RMSE)."""
        errors = []
        for gt, slam in zip(gt_traj, slam_traj):
            gt_pos = np.array(gt[1:4])
            slam_pos = np.array(slam[1:4])
            errors.append(np.linalg.norm(gt_pos - slam_pos))
        return np.sqrt(np.mean(np.square(errors))) if errors else 0.0

    def compute_rpe(self, gt_traj, slam_traj, delta=1.0):
        """Compute Relative Pose Error over distance delta."""
        trans_errors = []
        rot_errors = []

        for i in range(len(gt_traj) - 1):
            # Find j such that distance(i, j) ≈ delta
            for j in range(i + 1, len(gt_traj)):
                dist = np.linalg.norm(
                    np.array(gt_traj[j][1:4]) - np.array(gt_traj[i][1:4])
                )
                if dist >= delta:
                    # Compute relative motion (GT)
                    gt_delta = np.array(gt_traj[j][1:4]) - np.array(gt_traj[i][1:4])
                    slam_delta = np.array(slam_traj[j][1:4]) - np.array(slam_traj[i][1:4])

                    trans_errors.append(np.linalg.norm(gt_delta - slam_delta))

                    # Rotation error (simplified)
                    gt_q = np.array(gt_traj[j][4:8])
                    slam_q = np.array(slam_traj[j][4:8])
                    rot_errors.append(self.quaternion_angle_diff(gt_q, slam_q))
                    break

        rpe_trans = np.sqrt(np.mean(np.square(trans_errors))) if trans_errors else 0.0
        rpe_rot = np.mean(rot_errors) if rot_errors else 0.0

        return rpe_trans, rpe_rot

    @staticmethod
    def quaternion_angle_diff(q1, q2):
        """Compute angle difference between two quaternions in degrees."""
        dot = np.dot(q1, q2)
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.degrees(np.arccos(np.abs(dot)))

def main():
    rclpy.init()
    node = SLAMEvaluator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Integration**:
- Add to Docker container or run on host with ROS2 Jazzy
- Update `docker/cuvslam.launch.py` to include evaluator node

**Deliverables**:
- [x] `src/slam_evaluator.py` - Evaluator node implementation
- [ ] Real-time ATE/RPE metrics on `/slam/metrics` topic
- [ ] Ground truth path visualization in RViz

---

### Milestone 2: Enhanced RViz2 Visualization

**Objective**: Add cuVSLAM-specific displays to RViz for comprehensive SLAM visualization.

**File**: `rviz/jetbot.rviz`

**Additions**:

```yaml
# 1. SLAM estimated path (red)
- Class: nav_msgs/Path
  Name: SLAM Path
  Topic: /visual_slam/tracking/slam_path
  Color: 255, 0, 0
  Line Width: 0.03

# 2. Ground truth path (green)
- Class: nav_msgs/Path
  Name: Ground Truth Path
  Topic: /slam/ground_truth_path
  Color: 0, 255, 0
  Line Width: 0.03

# 3. Landmark point cloud (yellow)
- Class: PointCloud2
  Name: SLAM Landmarks
  Topic: /visual_slam/vis/landmarks_cloud
  Size: 0.02
  Color: 255, 255, 0

# 4. SLAM odometry with covariance
- Class: Odometry
  Name: SLAM Odometry
  Topic: /visual_slam/tracking/odometry
  Show Covariance: true
  Covariance Scale: 1.0
  Position Color: 255, 0, 0
  Orientation Color: 255, 127, 0

# 5. Observations (if enabled)
- Class: PointCloud2
  Name: SLAM Observations
  Topic: /visual_slam/vis/observations_cloud
  Size: 0.01
  Color: 0, 255, 255
```

**Deliverables**:
- [ ] Updated `rviz/jetbot.rviz` with SLAM displays
- [ ] Side-by-side GT vs SLAM path comparison
- [ ] Landmark cloud visualization

---

### Milestone 3: IMU Fusion Testing

**Objective**: Enable and validate IMU fusion in cuVSLAM.

**Changes**:

1. **Enable IMU in cuVSLAM** (`docker/cuvslam.launch.py:49`):
   ```python
   'enable_imu_fusion': True,  # Changed from False
   ```

2. **Test IMU Topic**:
   ```bash
   # Verify /jetbot/imu publishes
   ./run_slam.sh
   ros2 topic echo /jetbot/imu --once
   ```

3. **Comparative Evaluation**:
   - Run experiment WITH IMU: `enable_imu_fusion=true`
   - Run experiment WITHOUT IMU: `enable_imu_fusion=false`
   - Compare ATE/RPE metrics from `slam_evaluator.py`

**Test Scenarios**:
- Fast rotations (keyboard spin in place)
- Low-texture environment (add blank walls to Isaac Sim)
- Aggressive acceleration (quick start/stop)

**Deliverables**:
- [ ] IMU topic verified working
- [ ] IMU fusion enabled in cuVSLAM
- [ ] Quantitative comparison: IMU vs stereo-only
- [ ] Document findings in `plan/phase_5_results.md`

---

### Milestone 4: Map Persistence (Save/Load)

**Objective**: Add map save/load capability for localization-only mode.

**New Files**:
- `scripts/slam_save_map.sh` - Save current map
- `scripts/slam_load_map.sh` - Load existing map
- `scripts/slam_reset_map.sh` - Reset/clear map

**Implementation**:

```bash
#!/bin/bash
# scripts/slam_save_map.sh - Save SLAM map

MAP_DIR="${HOME}/.ros/maps"
mkdir -p "$MAP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAP_FILE="${MAP_DIR}/jetbot_map_${TIMESTAMP}.db"

echo "Saving SLAM map to: $MAP_FILE"

ros2 service call /visual_slam/save_map \
  isaac_ros_visual_slam_interfaces/srv/FilePath \
  "{file_path: '$MAP_FILE'}"

if [ $? -eq 0 ]; then
    echo "✅ Map saved successfully"
    echo "To load this map later, run:"
    echo "  ./scripts/slam_load_map.sh $MAP_FILE"
else
    echo "❌ Failed to save map"
fi
```

```bash
#!/bin/bash
# scripts/slam_load_map.sh - Load existing SLAM map

if [ -z "$1" ]; then
    echo "Usage: $0 <map_file.db>"
    echo "Available maps:"
    ls -lh ~/.ros/maps/*.db 2>/dev/null || echo "  (none)"
    exit 1
fi

MAP_FILE="$1"

if [ ! -f "$MAP_FILE" ]; then
    echo "❌ Map file not found: $MAP_FILE"
    exit 1
fi

echo "Loading SLAM map from: $MAP_FILE"

ros2 service call /visual_slam/load_map \
  isaac_ros_visual_slam_interfaces/srv/FilePath \
  "{file_path: '$MAP_FILE'}"

if [ $? -eq 0 ]; then
    echo "✅ Map loaded successfully"
    echo "cuVSLAM is now in localization-only mode"
else
    echo "❌ Failed to load map"
fi
```

**Localization-Only Mode**:

Update `docker/cuvslam.launch.py` to add argument:

```python
localization_only_arg = DeclareLaunchArgument(
    'localization_only',
    default_value='false',
    description='Localization-only mode (requires pre-loaded map)'
)

# In visual_slam_node parameters:
'enable_slam_visualization': LaunchConfiguration('localization_only'),  # Disable if localizing
```

**Deliverables**:
- [ ] Map save/load scripts in `scripts/`
- [ ] Test: Build map, save, restart, load, verify localization
- [ ] Document map persistence workflow

---

### Milestone 5: Automated Test Scenarios

**Objective**: Create repeatable test scenarios for SLAM evaluation.

**New Directory**: `scenarios/`

**Scenarios to Implement**:

1. **`scenarios/figure_eight.py`** - Figure-8 pattern for loop closure testing
2. **`scenarios/straight_line.py`** - Long straight path for drift measurement
3. **`scenarios/loop_closure.py`** - Drive loop, return to start
4. **`scenarios/rotation_stress.py`** - Aggressive rotations
5. **`scenarios/texture_variation.py`** - Rich vs poor texture areas

**Example**: `scenarios/figure_eight.py`

```python
#!/usr/bin/env python3
"""Figure-8 autonomous navigation scenario for SLAM testing.

Drives the Jetbot in a figure-8 pattern to test loop closure.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
import time

class FigureEightScenario(Node):
    def __init__(self):
        super().__init__('figure_eight_scenario')
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # Figure-8 parameters
        self.radius = 2.0  # meters
        self.speed = 0.3  # m/s
        self.omega = self.speed / self.radius  # rad/s

        self.get_logger().info("Starting figure-8 scenario...")
        self.run_scenario()

    def run_scenario(self):
        """Execute figure-8 pattern."""
        # First circle (clockwise)
        self.drive_circle(duration=2 * math.pi / self.omega, ccw=False)

        # Transition (straight)
        self.drive_straight(duration=1.0)

        # Second circle (counter-clockwise)
        self.drive_circle(duration=2 * math.pi / self.omega, ccw=True)

        # Stop
        self.stop()
        self.get_logger().info("Figure-8 complete!")

    def drive_circle(self, duration, ccw=True):
        """Drive in a circle for specified duration."""
        cmd = Twist()
        cmd.linear.x = self.speed
        cmd.angular.z = self.omega if ccw else -self.omega

        start = time.time()
        rate = self.create_rate(20)  # 20 Hz
        while (time.time() - start) < duration:
            self.pub_cmd.publish(cmd)
            rate.sleep()

    def drive_straight(self, duration):
        """Drive straight for specified duration."""
        cmd = Twist()
        cmd.linear.x = self.speed

        start = time.time()
        rate = self.create_rate(20)
        while (time.time() - start) < duration:
            self.pub_cmd.publish(cmd)
            rate.sleep()

    def stop(self):
        """Stop the robot."""
        self.pub_cmd.publish(Twist())

def main():
    rclpy.init()
    scenario = FigureEightScenario()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Integration Script**: `scripts/run_scenario.sh`

```bash
#!/bin/bash
# scripts/run_scenario.sh - Run a test scenario

if [ -z "$1" ]; then
    echo "Usage: $0 <scenario_name>"
    echo "Available scenarios:"
    ls -1 scenarios/*.py | xargs -n1 basename | sed 's/.py$/  /'
    exit 1
fi

SCENARIO="$1"

echo "Running scenario: $SCENARIO"
echo "Starting Isaac Sim + cuVSLAM in background..."

# Start Isaac Sim + ROS2 bridge
./run_slam.sh &
ISAAC_PID=$!

# Wait for startup
sleep 20

# Start cuVSLAM Docker
./docker/run_cuvslam.sh &
DOCKER_PID=$!

# Wait for cuVSLAM ready
sleep 10

# Run scenario
python3 scenarios/${SCENARIO}.py

# Keep running for 30s to finish SLAM processing
sleep 30

# Stop everything
echo "Stopping processes..."
kill $DOCKER_PID $ISAAC_PID
```

**Deliverables**:
- [ ] 5 test scenarios in `scenarios/`
- [ ] `scripts/run_scenario.sh` automation
- [ ] Record ATE/RPE for each scenario

---

### Milestone 6: Parameter Sweep Infrastructure

**Objective**: Systematically test cuVSLAM hyperparameters.

**New File**: `experiments/param_sweep.py`

**Parameters to Sweep**:

```python
PARAMS = {
    # Camera parameters
    'baseline': [0.05, 0.10, 0.15, 0.20],  # meters
    'img_width': [320, 640, 1280],
    'img_height': [240, 480, 960],

    # IMU parameters
    'enable_imu_fusion': [True, False],
    'gyro_noise_density': [0.000122, 0.000244, 0.000488],

    # SLAM parameters
    'enable_observations_view': [True, False],
    'image_jitter_threshold_ms': [17.0, 34.0, 50.0],
}
```

**Implementation**: Automated experiment runner with result logging.

**Deliverables**:
- [ ] `experiments/param_sweep.py` - Sweep implementation
- [ ] Results logged to `experiments/results/sweep_YYYYMMDD.csv`
- [ ] Analysis notebook `experiments/analyze_sweep.ipynb`

---

### Milestone 7: Loop Closure Monitoring

**Objective**: Detect and visualize loop closure events.

**Implementation**:
- Subscribe to cuVSLAM loop closure topic (if available)
- Log loop closure events with timestamps
- Visualize loop closure edges in RViz

**Deliverables**:
- [ ] Loop closure logging
- [ ] RViz markers for loop closures

---

### Milestone 8: Performance Profiling

**Objective**: Monitor cuVSLAM computational performance.

**New File**: `src/slam_profiler.py`

**Metrics**:
- Frame processing time (ms)
- CPU usage (%)
- GPU usage (%) - via nvidia-smi
- Memory usage (MB)
- Landmark count over time
- Topic publish rates (Hz)

**Deliverables**:
- [ ] `src/slam_profiler.py` - Performance monitor
- [ ] Real-time stats on `/slam/performance` topic
- [ ] Performance logs to `logs/perf_YYYYMMDD.csv`

---

## Success Criteria

Phase 5 is complete when:

1. ✅ SLAM evaluator runs and publishes ATE/RPE < 0.1m for 30s straight drive
2. ✅ RViz shows GT path (green) vs SLAM path (red) side-by-side
3. ✅ IMU fusion tested and quantitatively compared to stereo-only
4. ✅ Map save/load verified working (save, restart, load, localize)
5. ✅ 5 test scenarios implemented and runnable
6. ✅ Parameter sweep completes for at least 3 parameters
7. ✅ Performance profiling logs available
8. ✅ Documentation complete in `plan/phase_5_results.md`

---

## Phase 5 File Checklist

### New Files to Create

```
src/
├── slam_evaluator.py          # Ground truth comparison
├── slam_profiler.py            # Performance monitoring
└── test_slam_evaluator.py     # Unit tests

scenarios/
├── figure_eight.py             # Figure-8 loop closure test
├── straight_line.py            # Drift measurement
├── loop_closure.py             # Return to start
├── rotation_stress.py          # Aggressive rotations
└── texture_variation.py        # Texture robustness

scripts/
├── slam_save_map.sh            # Save SLAM map
├── slam_load_map.sh            # Load existing map
├── slam_reset_map.sh           # Clear map
└── run_scenario.sh             # Scenario automation

experiments/
├── param_sweep.py              # Hyperparameter sweep
├── analyze_sweep.ipynb         # Analysis notebook
└── results/
    └── sweep_YYYYMMDD.csv      # Sweep results

plan/
└── phase_5_results.md          # Findings & conclusions
```

### Files to Modify

```
docker/cuvslam.launch.py        # Enable IMU fusion, add localization mode
rviz/jetbot.rviz                # Add SLAM visualization displays
README.md                       # Document Phase 5 features
```

---

## Estimated Effort

| Milestone | Effort | Priority |
|-----------|--------|----------|
| M1: Ground truth comparison | 3 hours | HIGH |
| M2: RViz enhancement | 1 hour | HIGH |
| M3: IMU fusion testing | 2 hours | HIGH |
| M4: Map persistence | 2 hours | MEDIUM |
| M5: Test scenarios | 4 hours | MEDIUM |
| M6: Parameter sweep | 3 hours | LOW |
| M7: Loop closure monitoring | 2 hours | LOW |
| M8: Performance profiling | 3 hours | LOW |

**Total**: ~20 hours of implementation work

**Quick wins** (get to 80% in ~6 hours):
1. M1: Ground truth comparison (3h)
2. M2: RViz enhancement (1h)
3. M3: IMU fusion testing (2h)

---

## Next Steps

**Immediate** (today):
1. Implement `src/slam_evaluator.py`
2. Update `rviz/jetbot.rviz` with SLAM displays
3. Test IMU fusion (enable in launch file)

**Short-term** (this week):
4. Add map save/load scripts
5. Implement 2-3 test scenarios
6. Document findings

**Long-term** (as needed):
7. Parameter sweep infrastructure
8. Performance profiling
9. Loop closure monitoring

---

## References

- [cuVSLAM GitHub](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)
- [cuVSLAM Parameters](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html#parameters)
- [TUM RGB-D Benchmark (ATE/RPE)](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
- [EVO - Python package for trajectory evaluation](https://github.com/MichaelGrupp/evo)
