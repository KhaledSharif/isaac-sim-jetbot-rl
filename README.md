# Isaac Sim Jetbot Keyboard Control

Keyboard-controlled Jetbot mobile robot teleoperation with demonstration recording and reinforcement learning training pipeline for NVIDIA Isaac Sim 5.0.0.

## Features

- **Keyboard Teleoperation**: Control the Jetbot using WASD keys
- **Rich Terminal UI**: Real-time robot state display with visual feedback
- **Camera Streaming**: GStreamer H264 RTP UDP streaming from Jetbot camera
- **Random Obstacles**: Configurable static obstacles for navigation challenge
- **Demonstration Recording**: Record navigation trajectories for imitation learning
- **Automatic Demo Collection**: Autonomous data collection with noisy proportional controller
- **RL Training Pipeline**: Train PPO agents with optional behavioral cloning warmstart
- **Gymnasium Integration**: Standard RL environment compatible with Stable-Baselines3

## Requirements

- NVIDIA Isaac Sim 5.0.0 standalone
- NVIDIA RTX GPU
- Python 3.11 (bundled with Isaac Sim)

### Python Dependencies

```bash
~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh -m pip install numpy pynput rich stable-baselines3 tensorboard gymnasium
```

### Camera Streaming Dependencies (Optional)

For camera streaming functionality, install GStreamer and PyGObject:

```bash
# System packages (Ubuntu/Debian)
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
  libcairo2-dev libgirepository-2.0-dev pkg-config python3-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0

# Python package (in Isaac Sim environment)
~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh -m pip install PyGObject
```

## Quick Start

### Basic Teleoperation

```bash
./run.sh
```

### With Recording Enabled

```bash
./run.sh --enable-recording
```

### With Custom Obstacle Count

```bash
./run.sh --num-obstacles 10
```

### Train RL Agent

```bash
./run.sh train_rl.py --headless --timesteps 500000
```

## Controls

| Key | Action |
|-----|--------|
| W | Move forward |
| S | Move backward |
| A | Turn left |
| D | Turn right |
| Space | Stop (emergency brake) |
| R | Reset robot position |
| G | Spawn new goal |
| C | Toggle camera viewer |
| Esc | Exit |

### Recording Controls

| Key | Action |
|-----|--------|
| ` (backtick) | Toggle recording |
| [ | Mark episode success |
| ] | Mark episode failure |

## Project Structure

```
isaac-sim-jetbot-keyboard/
├── src/
│   ├── jetbot_keyboard_control.py    # Main teleoperation app
│   ├── camera_streamer.py            # Camera streaming module
│   ├── jetbot_rl_env.py              # Gymnasium RL environment
│   ├── train_rl.py                   # PPO training script
│   ├── eval_policy.py                # Policy evaluation
│   ├── train_bc.py                   # Behavioral cloning
│   ├── replay.py                     # Demo playback
│   ├── test_jetbot_keyboard_control.py
│   └── test_jetbot_rl_env.py
├── demos/                            # Recorded demonstrations
├── models/                           # Trained models
├── runs/                             # TensorBoard logs
├── run.sh                            # Isaac Sim Python launcher
├── run_tests.sh                      # Test runner
├── CLAUDE.md                         # AI assistant guidance
└── README.md
```

## Usage Examples

### Teleoperation

```bash
# Basic teleoperation (5 obstacles by default)
./run.sh

# With custom obstacle count
./run.sh --num-obstacles 10

# With recording enabled
./run.sh --enable-recording

# Disable camera streaming
./run.sh --no-camera

# Custom camera port
./run.sh --camera-port 5601

# Combine options
./run.sh --enable-recording --num-obstacles 8 --demo-path demos/my_demo.npz
```

### Automatic Demo Collection

```bash
# Collect 100 episodes automatically (default)
./run.sh --enable-recording --automatic

# Custom episode count
./run.sh --enable-recording --automatic --num-episodes 200

# Continuous mode (run until Esc)
./run.sh --enable-recording --automatic --continuous

# Headless TUI (console progress prints only)
./run.sh --enable-recording --automatic --num-episodes 200 --headless-tui
```

### Training

```bash
# Train PPO agent (headless for speed)
./run.sh train_rl.py --headless --timesteps 500000

# Train with behavioral cloning warmstart
./run.sh train_rl.py --bc-warmstart demos/recording.npz --timesteps 1000000

# Train BC model only
./run.sh train_bc.py demos/recording.npz --epochs 100
```

### Evaluation

```bash
# Evaluate trained policy
./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100

# Headless evaluation
./run.sh eval_policy.py models/ppo_jetbot.zip --headless --episodes 100
```

### Demo Inspection

```bash
# Show demo statistics (no simulation needed)
./run.sh replay.py demos/recording.npz --info

# Visual playback
./run.sh replay.py demos/recording.npz

# Replay specific episode
./run.sh replay.py demos/recording.npz --episode 0

# Replay successful episodes only
./run.sh replay.py demos/recording.npz --successful
```

### Testing

```bash
# Run all tests
./run_tests.sh

# Run with verbose output
./run_tests.sh -v

# Run specific test
./run_tests.sh -k test_action_mapper
```

## Observation Space (10D)

| Index | Description | Range |
|-------|-------------|-------|
| 0-1 | Robot position (x, y) | meters |
| 2 | Robot heading (theta) | radians |
| 3 | Linear velocity | m/s |
| 4 | Angular velocity | rad/s |
| 5-6 | Goal position (x, y) | meters |
| 7 | Distance to goal | meters |
| 8 | Angle to goal | radians |
| 9 | Goal reached flag | 0 or 1 |

## Action Space (2D)

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Linear velocity | [-1, 1] -> [-0.3, 0.3] m/s |
| 1 | Angular velocity | [-1, 1] -> [-1.0, 1.0] rad/s |

## Reward Function

### Dense Mode (default)
- Distance reward: Positive for getting closer to goal
- Heading bonus: Reward for facing the goal
- Goal reached: +10.0 bonus

### Sparse Mode
- Goal reached: +10.0
- Otherwise: 0.0

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir runs/
```

## Architecture

### Main Components

- **JetbotKeyboardController**: Main application with keyboard input and Rich TUI
- **JetbotNavigationEnv**: Gymnasium-compatible RL environment
- **DifferentialController**: Converts velocity commands to wheel speeds
- **SceneManager**: Manages goal markers, obstacles, and scene objects
- **DemoRecorder/DemoPlayer**: Recording and playback of demonstrations
- **CameraStreamer**: GStreamer H264 RTP UDP camera streaming
- **AutoPilot**: Noisy proportional controller for autonomous demo collection

### Obstacle System

The SceneManager randomly spawns static obstacles (FixedCuboid, FixedCylinder, FixedSphere, FixedCapsule) with:
- **Configurable count**: Set via `--num-obstacles` parameter (default: 5)
- **Random shapes**: Mix of boxes, cylinders, spheres, and capsules
- **Random sizes**: Varied dimensions for each obstacle type
- **Safe placement**: Maintains minimum distance from goal (0.5m) and robot start (1.0m)
- **Automatic respawn**: Obstacles regenerate when goal is reset (pressing 'G' or new episode)
- **Varied colors**: Gray, brown, and blue-gray tones to contrast with orange goal marker

### Isaac Sim Integration

The project uses Isaac Sim's:
- `WheeledRobot` class for the Jetbot
- `DifferentialController` for wheel velocity control
- `World` for simulation management

## License

MIT License

## Acknowledgments

- NVIDIA Isaac Sim team
- Based on the [isaac-sim-franka-keyboard](https://github.com/example/isaac-sim-franka-keyboard) project structure
