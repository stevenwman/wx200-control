# compact_gym

Clean, modular gym environment for WX200 robot arm control with teleoperation and data collection.

## Quick Start

### Running Data Collection

```bash
cd compact_gym
python collect_demo_gym.py
```

**Controls:**
- **SpaceMouse**: Move robot (translation/rotation in world frame)
- **Left/Right buttons**: Open/close gripper incrementally
- **GUI buttons**: Start Recording, Stop & Save, Stop & Discard, Home, Reset EE
- **Close GUI window**: Quit program

### Verifying Collected Data

```bash
# Validate most recent demo
python collection/validate_demo.py

# Validate specific file
python collection/validate_demo.py --file data/gym_demos/demo_0.npz

# Validate all demos
python collection/validate_demo.py --all
```

## Documentation

📚 **See [docs/INDEX.md](docs/INDEX.md)** for complete documentation index.

### Essential Reading

- **[docs/overview/ARCHITECTURE.md](docs/overview/ARCHITECTURE.md)** - System architecture and design
  - Clean layer separation (teleop → env → hardware)
  - Input-agnostic design (works with SpaceMouse, NN)
  - Dual-frequency control (120Hz motor commands, 10Hz data collection)
  - Data flow diagrams and verification

- **[docs/overview/ACTION_SPACE_NOTES.md](docs/overview/ACTION_SPACE_NOTES.md)** - Action space format
  - Normalized [-1, 1] action space
  - Denormalization semantics
  - Common pitfalls

### Troubleshooting

- **[docs/RUNTIME_FIXES.md](docs/RUNTIME_FIXES.md)** - Known issues and production fixes
- **[docs/TESTING.md](docs/TESTING.md)** - Testing procedures

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│ Teleop Layer (collect_demo_gym.py)         │
│  - SpaceMouse input                          │
│  - Action normalization [-1, 1]             │
│  - Data recording                            │
└─────────────────┬───────────────────────────┘
                  │ normalized action
                  ↓
┌─────────────────────────────────────────────┐
│ Environment Layer (gym_env.py)              │
│  - Input-agnostic step()                    │
│  - Action denormalization                    │
│  - Motor command execution (120Hz)          │
│  - Observation generation                    │
└─────────────────┬───────────────────────────┘
                  │ physical commands
                  ↓
┌─────────────────────────────────────────────┐
│ Hardware Layer (robot_hardware.py)          │
│  - IK solving                                │
│  - Motor commands                            │
│  - Encoder polling                           │
└─────────────────────────────────────────────┘
```

**Key Features:**
- ✅ **Input agnostic**: Swap SpaceMouse for NN policy without changing env
- ✅ **Clean separation**: No pollution between teleop, env, and hardware layers
- ✅ **Smooth control**: 120Hz motor commands, 10Hz data collection
- ✅ **Self-contained**: No dependencies on compact_code

## Project Structure

```
compact_gym/
├── collect_demo_gym.py          # Main data collection script
├── deployment/                  # 📦 Core gym environment (copy for NN deployment)
│   ├── gym_env.py              # Gymnasium environment
│   ├── robot_hardware.py       # Hardware interface
│   ├── robot_kinematics.py     # IK/FK solver
│   ├── robot_driver.py         # Motor commands
│   ├── robot_config.py         # Configuration
│   ├── camera.py               # Camera + ArUco
│   └── wx200/                  # Robot model
├── collection/                  # Data collection infrastructure
│   ├── validate_demo.py        # Demo validation
│   ├── spacemouse/             # SpaceMouse driver
│   └── utils/                  # Collection utilities
├── scripts/                     # Development/testing scripts
├── data/                        # Collected demonstrations
└── docs/                        # Documentation
```

## Data Format

Collected demos are saved as NPZ files in `data/gym_demos/` with the following structure:

```python
{
    'timestamp': float[T],           # Timestamps (seconds)
    'state': float[T, 6],            # Joint angles from encoders
    'encoder_values': int[T, 7],     # Raw encoder values
    'ee_pose_encoder': float[T, 7],  # EE pose from FK (pos + quat)
    'action': float[T, 7],           # Velocity commands (unnormalized)
    'action_normalized': float[T, 7], # Normalized action in [-1, 1]
    'augmented_actions': float[T, 10], # With axis-angle integration
    'ee_pose_target': float[T, 7],   # IK target pose
    'object_pose': float[T, 7],      # Object pose in world (if visible)
    'object_visible': float[T, 1],   # Object visibility flag
    'aruco_ee_in_world': float[T, 7],
    'aruco_object_in_world': float[T, 7],
    'aruco_ee_in_object': float[T, 7],
    'aruco_object_in_ee': float[T, 7],
    'aruco_visibility': float[T, 3],
    'camera_frame': uint8[T, 270, 480, 3]  # RGB frames (downscaled)
    'smoothed_aruco_*': float[T, 7], # Added in-place after save (if enabled)
    'metadata': dict                # created_at, file_name, config_snapshot
}
```

See [collection/validate_demo.py](collection/validate_demo.py) for validation checks.

### Trajectory Viewer

```bash
python trajectory_viewer_gui.py
```

Loads demos from `data/gym_demos/` and can render a video from `camera_frame`.

### Dataset Compilation (for training)

```bash
python merge_smoothed_trajectories.py data/gym_demos -o gym/hardware/merged_data_aruco_pos_ac_targets.npz
```

This produces a merged dataset with `observations`, `smoothed_observations`,
`actions_flat`, `next_observations`, `rewards`, `terminals`, and `masks`.

## Requirements

- Python 3.8+
- Robot hardware: Interbotix WX200 robot arm
- Input device: 3Dconnexion SpaceMouse
- Camera: USB camera for ArUco tracking (optional)

**Dependencies:**
```bash
pip install numpy gymnasium mujoco opencv-python pyrealsense2 scipy
pip install loop-rate-limiters  # For rate limiting
```

## Testing

```bash
# Syntax check (no hardware)
python scripts/verify_encoder_syntax.py

# Full hardware test
python scripts/test_encoder_polling.py

# Teleop verification
python scripts/verify_teleop_gym.py
```

See [docs/TESTING.md](docs/TESTING.md) for complete testing guide.

## Deployment for Neural Network Training

### Copy Deployment Package

For NN training/inference, copy only the `deployment/` folder to your project:

```bash
cp -r compact_gym/deployment /path/to/your/nn/project/
```

### Usage Example

```python
from deployment.gym_env import WX200GymEnv
from deployment.robot_config import robot_config

# Initialize environment
env = WX200GymEnv(
    max_episode_length=1000,
    show_video=False,
    enable_aruco=True,
    control_frequency=robot_config.control_frequency
)

# Run policy
obs, _ = env.reset()
while True:
    action = policy.predict(obs)  # Your NN replaces SpaceMouse
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

**What to copy:**
- ✅ `deployment/` folder (entire directory) - Core gym environment
- ❌ `collection/` folder - Data collection infrastructure (not needed)
- ❌ `collect_demo_gym.py` - Collection script (not needed)
- ❌ `scripts/` folder - Development/testing scripts (not needed)

## Development

### Changing Action Space

The environment is designed to easily support different action spaces (velocity, delta pose, absolute pose). See [docs/overview/ARCHITECTURE.md](docs/overview/ARCHITECTURE.md#input-source-independence) for examples.

### Adding New Input Sources

The environment is completely input-agnostic. See architecture docs for details on swapping SpaceMouse for other input sources.

---

**Status**: ✅ Production Ready

**Last Updated**: 2026-01-16
