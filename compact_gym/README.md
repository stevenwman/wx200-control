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
  - Input-agnostic design (works with SpaceMouse, NN, keyboard)
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
    'augmented_actions': float[T, 10], # With axis-angle integration
    'ee_pose_target': float[T, 7],   # IK target pose
    'aruco_*': float[T, 7],          # ArUco marker observations
    'camera_frame': uint8[T, 270, 480, 3]  # RGB frames (downscaled)
}
```

See [collection/validate_demo.py](collection/validate_demo.py) for validation checks.

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
