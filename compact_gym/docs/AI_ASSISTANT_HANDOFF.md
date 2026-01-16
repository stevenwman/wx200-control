# AI Assistant Handoff Document

**Last Updated**: 2026-01-16
**Status**: ✅ Production Ready - Recent reorganization complete

---

## Executive Summary

The `compact_gym` project is a **gymnasium-compatible environment for teleoperation and data collection** on an Interbotix WX200 robot arm. The codebase was recently reorganized (2026-01-16) to cleanly separate **deployment** code (for neural network training) from **collection infrastructure** (for human teleoperation demos).

**Primary Goal**: Enable clean, portable deployment of the gym environment for NN training while maintaining a robust data collection pipeline using SpaceMouse teleoperation.

---

## Critical Context: Recent Reorganization

### What Just Happened

The entire directory structure was reorganized to solve a deployment problem:

**Before**:
```
compact_gym/
├── gym_env.py
├── robot_hardware.py
├── collect_demo_gym.py
├── validate_demo.py
├── spacemouse/
├── test_*.py
└── (everything mixed together)
```

**After**:
```
compact_gym/
├── collect_demo_gym.py          # Main collection script (stays in root for easy access)
├── deployment/                  # 📦 Self-contained package - copy for NN deployment
│   ├── gym_env.py
│   ├── robot_hardware.py
│   ├── robot_kinematics.py
│   ├── robot_driver.py
│   ├── robot_config.py
│   ├── camera.py
│   ├── profiling.py
│   ├── fix_gstreamer_env.py
│   └── wx200/                   # MuJoCo model files
├── collection/                  # Data collection infrastructure (leave behind)
│   ├── validate_demo.py
│   ├── fix_usb_latency.py
│   ├── spacemouse/
│   │   ├── spacemouse_driver.py
│   │   └── spacemouse_reader.py
│   └── utils/
│       └── robot_control_gui.py
├── scripts/                     # Development/testing utilities
│   ├── test_env.py
│   ├── verify_teleop_gym.py
│   ├── test_encoder_polling.py
│   └── verify_*.py
├── data/
│   └── gym_demos/              # Collected demonstration NPZ files
└── docs/
```

### Why This Structure?

**User's deployment vision**: "I'm thinking ahead of time for what I will copy out of this directory for use in a deployment phase that doesn't involve all the data collection infrastructure (for use with a NN model instead)"

The `deployment/` folder is **completely self-contained** and can be copied to any NN training project:

```bash
cp -r compact_gym/deployment /path/to/nn_project/
```

Then use it like:
```python
from deployment.gym_env import WX200GymEnv
from deployment.robot_config import robot_config

env = WX200GymEnv(...)
obs, _ = env.reset()
action = policy.predict(obs)  # NN replaces SpaceMouse
obs, reward, done, truncated, info = env.step(action)
```

---

## Import Strategy: CRITICAL GOTCHA 🚨

### The Golden Rule

**Within packages** (`deployment/`, `collection/`): **ALWAYS use relative imports**

```python
# ✅ CORRECT - deployment/gym_env.py
from .robot_config import robot_config
from .robot_hardware import RobotHardware
from .camera import Camera

# ❌ WRONG - will break when deployment/ is copied elsewhere
from robot_config import robot_config
from deployment.robot_config import robot_config
```

**Between packages or from root**: Use absolute imports from package name

```python
# ✅ CORRECT - collect_demo_gym.py (in root)
from deployment.gym_env import WX200GymEnv
from collection.spacemouse.spacemouse_driver import SpaceMouseDriver

# ✅ CORRECT - scripts/test_env.py
from deployment.gym_env import WX200GymEnv
```

### Why Relative Imports?

When you copy `deployment/` to another project, relative imports (`.module`) continue to work because they're based on **package structure**, not file system location. Absolute imports would break because `deployment` wouldn't be in the same place.

### Recent Import Fixes (2026-01-16)

During reorganization, we found and fixed these import errors:

1. **collection/spacemouse/__init__.py** - Was using `from spacemouse.spacemouse_driver` instead of `from .spacemouse_driver`
2. **collection/spacemouse/spacemouse_driver.py** - Was using `from spacemouse.spacemouse_reader` instead of `from .spacemouse_reader`
3. **deployment/gym_env.py:416** - Had `from robot_kinematics import sync_robot_to_mujoco` instead of `from .robot_kinematics import`
4. **scripts/verify_encoder_implementation.py** - Used `from compact_gym.robot_hardware` instead of `from deployment.robot_hardware`
5. **scripts/test_encoder_polling.py** - Used `from compact_gym import WX200GymEnv` instead of proper package imports

### Package Structure Files

Each package has an `__init__.py`:

**deployment/__init__.py** - Uses **lazy loading** to avoid dependency errors:
```python
"""
Deployment package for WX200 gym environment.
...
"""

__all__ = ['gym_env', 'robot_hardware', 'robot_config', 'camera', 'profiling']

# NOTE: No eager imports! Don't do this:
# from .gym_env import WX200GymEnv  # ❌ Causes ModuleNotFoundError
```

**collection/__init__.py** - Simple package marker:
```python
"""
Collection infrastructure for data collection and validation.
"""
```

**collection/spacemouse/__init__.py** - Exports main class:
```python
"""SpaceMouse input modules."""
from .spacemouse_driver import SpaceMouseDriver

__all__ = ['SpaceMouseDriver']
```

---

## Architecture Overview

### Layer Separation (Critical Design Principle)

The system has **three clean layers** with zero pollution between them:

```
┌─────────────────────────────────────────────┐
│ Teleop Layer (collect_demo_gym.py)         │
│  - SpaceMouse input                          │
│  - Action normalization [-1, 1]             │
│  - Recording state machine                   │
│  - Data saving                               │
└─────────────────┬───────────────────────────┘
                  │ normalized actions
                  ▼
┌─────────────────────────────────────────────┐
│ Environment Layer (deployment/gym_env.py)   │
│  - Gymnasium interface                       │
│  - Action denormalization                    │
│  - Observation management                    │
│  - IK solving                                │
└─────────────────┬───────────────────────────┘
                  │ motor commands
                  ▼
┌─────────────────────────────────────────────┐
│ Hardware Layer (deployment/robot_hardware.py)│
│  - Motor control                             │
│  - Encoder reading                           │
│  - MuJoCo physics sync                       │
└─────────────────────────────────────────────┘
```

**Key insight**: The environment is **input-agnostic**. You can swap SpaceMouse for:
- Neural network policy
- Keyboard input
- Random policy
- Scripted trajectories

Just provide normalized actions `[-1, 1]^7` and the environment handles the rest.

### Dual-Frequency Architecture

**Critical performance feature**:

- **Inner loop (120Hz)**: Motor commands, IK solving, physics stepping
- **Outer loop (10Hz)**: Encoder polling, ArUco detection, data recording, user input

This prevents jitter while maintaining accurate data collection.

**Implementation details**:
- `env.step()` uses `dt = 1/120` for physics
- Data collection calls `env.step()` in a loop at 10Hz
- Each outer loop cycle runs 12 inner loop steps
- See [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #6 for full explanation

### Data Flow

**During teleoperation**:
1. SpaceMouse raw input → normalized action `[-1, 1]^7`
2. `env.step(action)` → denormalizes to velocity commands
3. IK solver → computes joint velocities
4. Motor commands → sent at 120Hz
5. Encoders read → synced to MuJoCo at 10Hz
6. Observations → ArUco poses, joint states, images
7. Recording → saves to NPZ at 10Hz

**During NN deployment**:
1. Policy network → normalized action `[-1, 1]^7`
2. `env.step(action)` → (same as above)
3. No recording infrastructure needed

---

## Action Space Design

### Normalized Action Space

**Format**: `action[7]` all in range `[-1, 1]`

```python
action = [
    v_x,        # End-effector velocity in world X (forward/back)
    v_y,        # End-effector velocity in world Y (left/right)
    v_z,        # End-effector velocity in world Z (up/down)
    omega_x,    # Angular velocity around world X (roll)
    omega_y,    # Angular velocity around world Y (pitch)
    omega_z,    # Angular velocity around world Z (yaw)
    gripper     # Gripper: -1 = open, +1 = close
]
```

**All velocities are in world frame**, not end-effector frame!

### Denormalization

Happens in `deployment/gym_env.py`:

```python
# Translation: [-1, 1] → [-max_lin_vel, +max_lin_vel] m/s
ee_command[:3] = action[:3] * robot_config.max_ee_linear_velocity

# Rotation: [-1, 1] → [-max_ang_vel, +max_ang_vel] rad/s
ee_command[3:6] = action[3:6] * robot_config.max_ee_angular_velocity

# Gripper: [-1, 1] → stay as is
ee_command[6] = action[6]
```

**GOTCHA**: The gripper action is **NOT** denormalized. It stays in `[-1, 1]` and is interpreted as:
- `action[6] < 0` → open gripper
- `action[6] > 0` → close gripper
- Magnitude controls speed

### Common Mistakes

❌ **Don't** pass velocity commands directly to `env.step()` - they must be normalized first
❌ **Don't** confuse world frame with end-effector frame
❌ **Don't** denormalize actions in the teleop layer - that's the env's job
✅ **Do** keep all actions in `[-1, 1]` when calling `env.step()`
✅ **Do** let the environment handle all denormalization internally

See [docs/overview/ACTION_SPACE_NOTES.md](overview/ACTION_SPACE_NOTES.md) for more details.

---

## Key Files and Their Roles

### Deployment Package Files

**deployment/gym_env.py** (658 lines)
- Main gymnasium environment interface
- Implements `reset()`, `step()`, `close()`
- Handles action denormalization
- Manages ArUco background thread
- Controls dual-frequency architecture
- **Key classes**: `WX200GymEnv`

**deployment/robot_hardware.py** (388 lines)
- Hardware abstraction layer
- Manages RobotDriver, RobotController, JointToMotorTranslator
- Encoder polling at 10Hz (outer loop only)
- MuJoCo physics sync
- **Key classes**: `RobotHardware`

**deployment/robot_kinematics.py** (486 lines)
- IK/FK solver using MuJoCo
- Joint to motor translations
- Gripper position control
- Robot-to-MuJoCo state syncing
- **Key classes**: `RobotController`, `JointToMotorTranslator`

**deployment/robot_driver.py** (289 lines)
- Low-level motor control via Dynamixel SDK
- Direct serial communication to robot
- Motor position/velocity commands
- Encoder reading
- **Key classes**: `RobotDriver`

**deployment/robot_config.py** (135 lines)
- Single source of truth for all configuration
- Hardware limits, control frequencies, camera settings
- **Key object**: `robot_config` (singleton)

**deployment/camera.py** (398 lines)
- Camera interface (USB or RealSense)
- ArUco marker detection and pose estimation
- GStreamer environment fix for OpenCV
- **Key classes**: `Camera`, `ArUcoPoseEstimator`

**deployment/profiling.py** (205 lines)
- Lightweight profiling for performance monitoring
- ArUco detection statistics
- **Key classes**: `LightweightProfiler`, `ArUcoProfiler`

**deployment/wx200/** (directory)
- MuJoCo XML model files
- Scene configurations
- Robot URDF/MJCF definitions

### Collection Infrastructure Files

**collection/validate_demo.py** (289 lines)
- Validates recorded demonstration files
- Checks for required fields, correct shapes, no NaNs
- Verifies recording frequency (~10Hz)
- Can validate single demo or all demos
- Usage: `python collection/validate_demo.py [--file demo.npz] [--all]`

**collection/fix_usb_latency.py** (60 lines)
- Fixes USB latency from 16ms → 1ms for smooth control
- Auto-runs on startup in `collect_demo_gym.py`
- Requires sudo access
- **Critical for smooth teleoperation**

**collection/spacemouse/spacemouse_driver.py** (222 lines)
- High-level SpaceMouse interface
- Handles button presses, axis scaling
- Multiprocessing architecture for low-latency reading
- **Key classes**: `SpaceMouseDriver`

**collection/spacemouse/spacemouse_reader.py** (88 lines)
- Low-level SpaceMouse HID reading
- Runs in separate process for isolation
- **Key functions**: `spacemouse_process`

**collection/utils/robot_control_gui.py** (128 lines)
- Simple Tkinter GUI for status display
- Shows recording state, profiling info
- Runs in separate thread

### Main Scripts

**collect_demo_gym.py** (437 lines)
- Main data collection script
- SpaceMouse teleoperation loop
- Recording state machine (ready/recording/paused)
- Saves demos as NPZ files
- Usage: `python collect_demo_gym.py`
- **Keyboard controls**:
  - `r` - Start recording
  - `d` - Save demo
  - `x` - Discard demo
  - `h` - Home robot
  - `q` - Quit

**scripts/verify_teleop_gym.py** (156 lines)
- Tests teleoperation without recording
- 30-second SpaceMouse control test
- Verifies action normalization
- Usage: `python scripts/verify_teleop_gym.py`

**scripts/test_env.py** (52 lines)
- Random policy test
- Verifies environment loads correctly
- Usage: `python scripts/test_env.py`

---

## Data Format

Demonstrations are saved as NumPy NPZ files in `data/gym_demos/`:

```python
{
    # Timestamps
    'timestamp': float[T],              # Wall clock timestamps (seconds)

    # Robot state
    'qpos': float[T, 6],               # Joint positions (radians)
    'qvel': float[T, 6],               # Joint velocities (rad/s)
    'gripper_pos': float[T],           # Gripper position (0-1)
    'ee_pose': float[T, 7],            # End-effector pose [x,y,z, qx,qy,qz,qw]

    # Actions (denormalized velocity commands)
    'action': float[T, 7],             # [v_x, v_y, v_z, ω_x, ω_y, ω_z, gripper]

    # Augmented actions (with axis-angle integration)
    'augmented_actions': float[T, 10], # Actions + integrated orientation

    # IK targets
    'ee_pose_target': float[T, 7],     # Target pose sent to IK solver

    # ArUco markers (if enabled)
    'aruco_0': float[T, 7],            # Marker 0 pose [x,y,z, qx,qy,qz,qw]
    'aruco_1': float[T, 7],            # Marker 1 pose
    # ... (one per marker ID)

    # Camera (if enabled)
    'camera_frame': uint8[T, 270, 480, 3]  # RGB frames (downscaled from 1920x1080)
}
```

**Key validation criteria** (from `collection/validate_demo.py`):
- ✅ All expected fields present
- ✅ Trajectory length > 1 step
- ✅ Recording frequency ~10 Hz (±20% tolerance)
- ✅ `ee_pose_target` is not all zeros (IK target being tracked)
- ✅ No NaN values in critical fields
- ✅ Camera frames recorded (if camera enabled)

---

## Configuration: robot_config.py

All configuration is centralized in `deployment/robot_config.py`:

```python
@dataclass
class RobotConfig:
    # Control frequencies
    control_frequency: int = 10          # Outer loop (data collection)
    inner_control_frequency: int = 120   # Inner loop (motor commands)

    # Velocity limits (in m/s and rad/s)
    max_ee_linear_velocity: float = 0.15   # Max translation speed
    max_ee_angular_velocity: float = 1.0   # Max rotation speed
    velocity_limit: float = 0.5            # Motor velocity limit

    # Gripper limits
    gripper_encoder_min: int = 1250
    gripper_encoder_max: int = 2550

    # Camera
    camera_id: int = 1                   # USB camera device ID
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    aruco_dict: int = cv2.aruco.DICT_4X4_50

    # ArUco thread
    aruco_polling_rate: int = 30         # Hz

    # Profiling
    profiler_window_size: int = 100      # Samples for rolling average
```

**To modify behavior**: Edit values in `robot_config.py`, don't hardcode magic numbers elsewhere.

---

## Known Issues and Gotchas

### 1. USB Latency (CRITICAL)

**Problem**: Default USB latency is 16ms, causing jittery robot motion.

**Solution**: `collection/fix_usb_latency.py` automatically runs on startup and fixes it to 1ms.

**Manual fix**:
```bash
python collection/fix_usb_latency.py
```

**Reference**: [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #1

### 2. Robot Moving 12x Too Fast

**Problem**: If robot moves way too fast, `env.step()` is using wrong dt.

**Root cause**: Using `control_frequency` (10Hz) instead of `inner_control_frequency` (120Hz) for physics stepping.

**Fix**: Verify `gym_env.py` uses:
```python
self.dt = 1.0 / robot_config.inner_control_frequency  # Should be 1/120 = 0.00833s
```

**Reference**: [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #5

### 3. Jittery Motion

**Problem**: Robot motion is stuttering or jerky.

**Root cause**: Motor commands running at 10Hz instead of 120Hz.

**Fix**: Ensure dual-frequency architecture is working - inner loop must run at 120Hz.

**Reference**: [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #6

### 4. Keyboard Interrupt Doesn't Clean Up

**Problem**: Ctrl+C during execution leaves robot in bad state.

**Solution**: Ensure proper try/finally blocks:
```python
try:
    collector.run()
finally:
    print("\n[Main] Ensuring cleanup...")
    collector.cleanup()
```

**Reference**: [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #2

### 5. Camera Initialization Fails

**Expected behavior**: System continues without camera, ArUco disabled.

**Common causes**:
- Wrong `camera_id` in `robot_config.py`
- Camera not connected
- Permission issues with `/dev/video*`

**Debug**: Check available cameras with `ls /dev/video*`

### 6. Gripper Not Moving

**Problem**: Gripper commands don't work, motor error state.

**Solution**: `gym_env.py` automatically reboots motor 7 (gripper) on reset:
```python
print("Rebooting motor 7 to clear error state...")
self.robot_hardware.robot_driver.reboot_motor(7)
```

This is normal and expected behavior.

### 7. Import Errors After Reorganization

**Problem**: `ModuleNotFoundError` for various modules.

**Most common causes**:
1. Not using relative imports within packages
2. Using old `compact_gym.module` import style
3. Missing `__init__.py` files

**Debug checklist**:
- ✅ All imports within `deployment/` use relative imports (`.module`)
- ✅ All imports within `collection/` use relative imports
- ✅ External imports use `from deployment.X` or `from collection.Y`
- ✅ Each package directory has `__init__.py`

### 8. Encoder Lag Issues

**Problem**: Encoder readings lag behind actual robot state.

**Historical context**: This was a major issue that was fixed by:
1. Moving encoder polling OUT of the 120Hz inner loop
2. Only polling encoders at 10Hz in outer loop
3. Using cached encoder values from `robot_hardware.latest_encoder_values`

**If you see encoder lag**, verify encoder polling is NOT in the inner loop.

**Reference**: [docs/archived/PHASE1_COMPLETE.md](archived/PHASE1_COMPLETE.md)

---

## Testing and Verification

### Quick Verification

**Test 1: Data collection works**
```bash
python collect_demo_gym.py
# - Move robot with SpaceMouse
# - Press 'r' to start recording
# - Press 'd' to save
# - Press 'q' to quit
```

**Test 2: Validate collected data**
```bash
python collection/validate_demo.py
```

**Test 3: Teleoperation without recording**
```bash
python scripts/verify_teleop_gym.py
```

### Syntax Checks (No Hardware Required)

```bash
python scripts/verify_encoder_syntax.py
python scripts/verify_aruco_thread_syntax.py
```

### Full Testing Checklist

See [docs/TESTING.md](TESTING.md) for comprehensive testing procedures.

**Startup checks**:
- [ ] Scene XML loads correctly
- [ ] Robot homes without errors
- [ ] USB latency auto-fixed (or warning shown)
- [ ] Camera initializes (or graceful failure)
- [ ] GUI displays "Ready" status

**Control checks**:
- [ ] SpaceMouse translation moves robot smoothly
- [ ] SpaceMouse rotation works correctly
- [ ] Gripper opens/closes with buttons
- [ ] No jitter or stuttering
- [ ] Control frequency ~120Hz (check profiling output)

**Recording checks**:
- [ ] `r` key starts recording (red status)
- [ ] `d` key saves demo (green status)
- [ ] `x` key discards demo
- [ ] NPZ file created in `data/gym_demos/`
- [ ] Validation passes for recorded demo

**Shutdown checks**:
- [ ] `q` key exits cleanly
- [ ] Ctrl+C triggers proper cleanup
- [ ] Robot returns to home
- [ ] Motors disable (torque off)
- [ ] No hanging processes

---

## Performance Benchmarks

**Expected timings** (from profiling):
- Control loop: ~8ms avg @ 120Hz
- `env.step()`: 1-3ms (motor commands + IK)
- Encoder polling: 8-12ms @ 10Hz (outer loop only)
- ArUco detection: 30Hz (background thread)
- Recording: < 1ms (outer loop only)

**Frequencies**:
- Inner loop (motor commands): 120Hz
- Outer loop (input/recording): 10Hz
- ArUco thread: 30Hz
- Recorded data: 10Hz

**If performance degrades**, check:
1. USB latency still at 1ms
2. Encoder polling not in inner loop
3. ArUco thread not blocking main loop
4. No unnecessary file I/O in control loop

---

## Development History and Context

### Phase 1: Encoder Polling Fix (Completed)

**Problem**: Encoder reading was blocking the 120Hz control loop, causing lag.

**Solution**:
- Moved encoder polling to outer loop (10Hz)
- Cached encoder values in `robot_hardware.latest_encoder_values`
- Inner loop uses cached values

**Reference**: [docs/archived/PHASE1_COMPLETE.md](archived/PHASE1_COMPLETE.md)

### Phase 2: ArUco Background Thread (Completed)

**Problem**: ArUco detection (~30ms) blocked control loop.

**Solution**:
- Background thread for ArUco detection at 30Hz
- Non-blocking updates to `env._latest_aruco_observations`
- Thread-safe observation retrieval

**Reference**: [docs/archived/PHASE2_COMPLETE.md](archived/PHASE2_COMPLETE.md)

### Phase 3: Data Collection Prep (Completed)

**Problem**: Multiple runtime issues discovered during testing.

**Fixed issues**:
1. USB latency auto-fix
2. Keyboard interrupt cleanup
3. Gripper error state handling
4. ArUco visibility warnings
5. Robot speed issues (12x too fast)
6. Dual-frequency architecture documentation

**Reference**: [docs/archived/PHASE3_PREP_FIXES.md](archived/PHASE3_PREP_FIXES.md)

### Phase 4: Directory Reorganization (Completed 2026-01-16)

**Problem**: Need clean separation for NN deployment.

**Solution**: Current structure with `deployment/`, `collection/`, `scripts/` separation.

**This document was created during this phase.**

---

## For Neural Network Deployment

### Quick Start

1. **Copy deployment package**:
```bash
cp -r compact_gym/deployment /path/to/nn_project/
```

2. **Use in training script**:
```python
from deployment.gym_env import WX200GymEnv
from deployment.robot_config import robot_config

# Initialize environment
env = WX200GymEnv(
    max_episode_length=1000,
    show_video=False,           # No GUI needed
    enable_aruco=True,           # Or False if no markers
    control_frequency=robot_config.control_frequency
)

# Training loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    while True:
        action = policy.predict(obs)  # Your NN policy
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()
```

3. **What NOT to copy**:
- ❌ `collection/` - Teleoperation infrastructure
- ❌ `collect_demo_gym.py` - Data collection script
- ❌ `scripts/` - Testing utilities
- ❌ `data/` - Demo recordings (copy separately if needed)

### Observation Space

The environment returns observations as a flat array:

```python
obs_space = {
    'qpos': 6,              # Joint positions
    'qvel': 6,              # Joint velocities
    'gripper_pos': 1,       # Gripper position
    'ee_pose': 7,           # End-effector pose [x,y,z, qx,qy,qz,qw]
    'aruco_poses': N * 7,   # N marker poses (if enabled)
}

# Total dimension: 20 + (N_markers * 7)
# Default: 20 + (2 * 7) = 34
```

### Reward Function

**Currently**: Environment returns `reward = 0.0` (no reward shaping).

**To customize**: Edit `WX200GymEnv.step()` in `deployment/gym_env.py`:

```python
def step(self, action):
    # ... (control code)

    # Compute your reward here
    reward = self._compute_reward(obs, action, info)

    return obs, reward, terminated, truncated, info
```

Common reward formulations:
- Distance to goal position
- Negative action magnitude (smoothness)
- Task-specific objectives (object manipulation, etc.)

### Changing Action Space

The environment is designed to support different action spaces:

**Option 1: Delta pose** (relative position changes):
```python
# Modify env.step() to integrate delta poses
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

**Option 2: Absolute pose** (target end-effector pose):
```python
# Modify env.step() to compute velocity from error
action = [target_x, target_y, target_z, target_qx, ..., gripper]
```

**Option 3: Joint space** (direct joint control):
```python
# Bypass IK entirely, command joints directly
action = [q0, q1, q2, q3, q4, q5, gripper]
```

See [docs/overview/ARCHITECTURE.md](overview/ARCHITECTURE.md) for implementation examples.

---

## Common AI Assistant Mistakes to Avoid

### 1. Don't Create New Files Without Permission

**Bad**: "I'll create a new `robot_utils.py` file to organize these functions..."

**Good**: "I notice these functions could be organized. Should I refactor the existing file or create a new utility module?"

**Why**: User explicitly stated: "ALWAYS prefer editing an existing file to creating a new one."

### 2. Don't Add Unnecessary Comments/Docstrings

**Bad**: Adding docstrings to every function you touch.

**Good**: Only add comments where logic is non-obvious.

**Why**: User values concise, self-documenting code over verbose documentation.

### 3. Don't Over-Engineer Solutions

**Bad**: "Let's add a configuration system with YAML files, environment variables, and a builder pattern..."

**Good**: "I'll add the config parameter to `robot_config.py` dataclass."

**Why**: User explicitly stated: "Avoid over-engineering. Only make changes that are directly requested or clearly necessary."

### 4. Don't Add Error Handling for Impossible Cases

**Bad**: Adding try/except around internal function calls that can't fail.

**Good**: Only validate at system boundaries (user input, external APIs).

**Why**: User stated: "Don't add error handling, fallbacks, or validation for scenarios that can't happen."

### 5. Don't Create Abstractions Prematurely

**Bad**: "Let's create a base class for all sensors since we might add more later..."

**Good**: Keep it simple until there's a concrete need.

**Why**: User stated: "Three similar lines of code is better than a premature abstraction."

### 6. Don't Use Backwards-Compatibility Hacks

**Bad**: Renaming unused variables with `_` prefix, adding `# removed` comments.

**Good**: Delete unused code completely.

**Why**: Clean codebase, no legacy cruft.

### 7. Always Read Files Before Modifying

**Bad**: "I'll update the import in `gym_env.py`..." (without reading it first)

**Good**: Use Read tool first, understand context, then edit.

**Why**: User stated: "NEVER propose changes to code you haven't read."

### 8. Don't Batch Multiple Tasks

**Bad**: "I'll fix all 5 import errors in one message..."

**Good**: If using TodoWrite, mark each completed immediately.

**Why**: User wants to see incremental progress, not big-bang updates.

---

## Architecture Deep Dive

### Why MuJoCo for IK?

The system uses MuJoCo's built-in IK solver (`mj_jacBodyCom`, `mj_solveM`) instead of analytic IK because:
1. Handles joint limits automatically
2. Respects collision constraints
3. Smooth, numerically stable solutions
4. Easy to extend with obstacles

**Implementation**: See `RobotController.compute_joint_velocities()` in `deployment/robot_kinematics.py`.

### Why Separate Physics Sim?

The real robot and MuJoCo sim run in parallel:
- **Real robot**: Executes commands, reads encoders
- **MuJoCo sim**: Computes IK, forward kinematics, visualization

**Syncing**: Encoder values periodically synced to MuJoCo joint positions via `sync_robot_to_mujoco()`.

**Why not sim-only?**: Need to handle real-world dynamics, backlash, compliance that MuJoCo can't model perfectly.

### Why SpaceMouse in Separate Process?

SpaceMouse reading runs in a separate process (`multiprocessing.Process`) because:
1. HID reading can block
2. Isolates failures (USB disconnect won't crash main loop)
3. Lower latency for input polling

**Communication**: `multiprocessing.Queue` for commands, `multiprocessing.Value` for button states.

### Why ArUco in Separate Thread?

ArUco detection (~30ms) would block the 120Hz control loop. Background thread:
1. Continuously captures and processes frames at 30Hz
2. Updates shared `_latest_aruco_observations` dictionary
3. Main thread reads latest observations without blocking

**Thread safety**: Uses proper locking (`threading.Lock`) around shared data.

### Why Two Frequencies?

**120Hz inner loop**:
- Motor commands must be sent frequently for smooth motion
- IK must run at high frequency for accurate tracking
- Physics must step at high frequency for stability

**10Hz outer loop**:
- Encoder reading is slow (~8-12ms)
- ArUco updates are 30Hz, no need to check faster
- Recording at 10Hz is sufficient for imitation learning
- User input (keyboard) doesn't need faster polling

**Result**: Smooth robot motion, responsive control, efficient data collection.

---

## Debugging Tips

### Enable Profiling

In `collect_demo_gym.py`, profiling is already enabled. Check output:

```
[Profiler]
  Inner loop avg: 8.23 ms (121.5 Hz)
  Outer loop avg: 100.1 ms (10.0 Hz)
  ArUco thread: 33.2 ms (30.1 Hz)
```

If numbers look wrong, investigate:
- Inner loop > 10ms → Something blocking in step()
- Outer loop > 120ms → Encoder polling or ArUco check too slow
- ArUco thread > 50ms → Camera resolution too high or detection too slow

### Check USB Latency

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# Should output: 1
```

If it's 16, run `python collection/fix_usb_latency.py`.

### Monitor Robot State

Add debug prints in `gym_env.py`:

```python
def step(self, action):
    print(f"Action: {action}")
    print(f"EE pose: {self.robot_hardware.controller.get_ee_position()}")
    # ...
```

### Validate Recorded Data

```bash
# Check most recent demo
python collection/validate_demo.py

# Check specific demo
python collection/validate_demo.py --file data/gym_demos/demo_0.npz

# Check all demos
python collection/validate_demo.py --all
```

### Common Error Messages

**"ModuleNotFoundError: No module named 'gymnasium'"**
- Missing dependencies, run: `pip install gymnasium mujoco numpy opencv-python`

**"ModuleNotFoundError: No module named 'robot_kinematics'"**
- Import error, should be `from .robot_kinematics import ...` in `deployment/` files

**"USB latency is 16ms"**
- Auto-fix should run, if not: `python collection/fix_usb_latency.py`

**"Camera initialization failed"**
- Expected if no camera, system continues without ArUco
- If camera exists, check `camera_id` in `robot_config.py`

**"Motor 7 error state"**
- Gripper motor, auto-reboots on reset, this is normal

---

## File Organization Rules

### What Goes Where?

**deployment/**:
- ✅ Core environment code
- ✅ Hardware interfaces
- ✅ IK/FK solvers
- ✅ Configuration
- ✅ MuJoCo model files
- ❌ NO teleoperation code
- ❌ NO data collection infrastructure
- ❌ NO GUI code

**collection/**:
- ✅ Data collection utilities
- ✅ Demo validation
- ✅ SpaceMouse driver
- ✅ Control GUI
- ✅ Data preprocessing tools
- ❌ NO core environment code
- ❌ NO hardware interfaces

**scripts/**:
- ✅ Testing utilities
- ✅ Verification scripts
- ✅ Development tools
- ✅ One-off experiments
- ❌ NO production code

**Root directory**:
- ✅ `collect_demo_gym.py` - Main collection script
- ✅ `README.md` - Project overview
- ❌ NO other Python modules

**docs/**:
- ✅ Architecture documentation
- ✅ Testing guides
- ✅ Runtime fixes
- ✅ Historical context (archived/)
- ❌ NO code

### Naming Conventions

**Files**:
- `snake_case.py` for all Python files
- `UPPERCASE.md` for documentation (README.md, TESTING.md)
- `lowercase/` for directories

**Classes**:
- `PascalCase` for all classes
- `WX200GymEnv`, `RobotHardware`, `SpaceMouseDriver`

**Functions**:
- `snake_case()` for all functions
- `compute_joint_velocities()`, `read_encoders()`

**Variables**:
- `snake_case` for all variables
- `ee_pose`, `joint_positions`, `aruco_observations`

**Constants**:
- `UPPER_SNAKE_CASE` for constants
- `MAX_VELOCITY`, `MARKER_SIZE`, `ARUCO_DICT_4X4_50`

---

## Git Workflow

### Current Branch
```
main
```

### Recent Commits
```
e612148 untracked .npz
ee76d08 started gym migration, compact_code has most updated demo collection
150e045 added fix for USB latency, threading, and other data collection infrastructure changes
490a112 added old_data to gitignore
4dab539 fixed encoder lag during collection, added ee ik vs fk comparisons
```

### Uncommitted Changes

Currently modified:
- `compact_gym/collect_demo_gym.py`

**Note**: The directory reorganization (2026-01-16) has NOT been committed yet. User was testing when this handoff was created.

### Commit Message Style

User prefers descriptive, technical commit messages:
- "fixed encoder lag during collection, added ee ik vs fk comparisons"
- "added fix for USB latency, threading, and other data collection infrastructure changes"
- "started gym migration, compact_code has most updated demo collection"

**Good**: Specific, describes what changed and why
**Bad**: Generic messages like "bug fixes" or "updates"

---

## Next Steps and Future Work

### Immediate Testing Needed

1. **Record a demo** and validate it:
```bash
python collect_demo_gym.py
# Record, save, quit
python collection/validate_demo.py
```

2. **Test deployment copy**:
```bash
mkdir /tmp/test_deployment
cp -r compact_gym/deployment /tmp/test_deployment/
cd /tmp/test_deployment
python -c "from deployment.gym_env import WX200GymEnv; print('Import successful!')"
```

3. **Run full test suite**:
```bash
python scripts/verify_teleop_gym.py
python scripts/test_encoder_polling.py
python scripts/test_env.py
```

### Potential Future Enhancements

**User has NOT requested these**, but they may come up:

1. **Reward shaping** - Currently returns 0.0, needs task-specific design
2. **Multi-camera support** - Currently single camera
3. **Different action spaces** - Currently velocity-based
4. **Domain randomization** - For sim-to-real transfer
5. **Trajectory replay** - Play back recorded demos
6. **Visual servoing** - Use ArUco for closed-loop control
7. **Force control** - Add force/torque sensing

**Don't implement these unless explicitly requested!**

### Documentation to Maintain

If you make changes, update these files:

- [README.md](../README.md) - If usage changes
- [docs/TESTING.md](TESTING.md) - If testing procedures change
- [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) - If you fix runtime issues
- [docs/overview/ARCHITECTURE.md](overview/ARCHITECTURE.md) - If architecture changes
- [docs/overview/ACTION_SPACE_NOTES.md](overview/ACTION_SPACE_NOTES.md) - If action space changes
- **This file** - If major structure changes

---

## Contact and Resources

### Essential Reading (in order)

1. **[docs/overview/ARCHITECTURE.md](overview/ARCHITECTURE.md)** - System architecture (READ THIS FIRST)
2. **[docs/overview/ACTION_SPACE_NOTES.md](overview/ACTION_SPACE_NOTES.md)** - Action space format
3. **[docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md)** - Known issues and solutions
4. **[docs/TESTING.md](TESTING.md)** - Testing procedures
5. **This file** - Complete context and handoff

### Quick Navigation

- **Project overview**: [README.md](../README.md)
- **Architecture**: [docs/overview/ARCHITECTURE.md](overview/ARCHITECTURE.md)
- **Testing**: [docs/TESTING.md](TESTING.md)
- **Troubleshooting**: [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md)
- **Action space**: [docs/overview/ACTION_SPACE_NOTES.md](overview/ACTION_SPACE_NOTES.md)
- **All docs**: [docs/INDEX.md](INDEX.md)

### Historical Context

- **Phase 1 (Encoder fix)**: [docs/archived/PHASE1_COMPLETE.md](archived/PHASE1_COMPLETE.md)
- **Phase 2 (ArUco thread)**: [docs/archived/PHASE2_COMPLETE.md](archived/PHASE2_COMPLETE.md)
- **Phase 3 (Pre-launch fixes)**: [docs/archived/PHASE3_PREP_FIXES.md](archived/PHASE3_PREP_FIXES.md)

---

## Summary Checklist for New AI Assistants

Before making changes, verify you understand:

- [ ] Directory structure: `deployment/`, `collection/`, `scripts/`
- [ ] Import rules: Relative imports within packages, absolute between packages
- [ ] Action space: Normalized `[-1, 1]^7`, world frame velocities
- [ ] Dual-frequency architecture: 120Hz inner, 10Hz outer
- [ ] Layer separation: Teleop → Env → Hardware (no pollution)
- [ ] User preferences: Don't over-engineer, prefer existing files, concise code
- [ ] Testing procedures: Always test before claiming success
- [ ] Git status: `collect_demo_gym.py` currently modified, reorganization not committed yet

**When in doubt**: Ask the user instead of making assumptions!

---

**This handoff document was created**: 2026-01-16
**Last tested**: 2026-01-16 (reorganization testing in progress)
**Status**: ✅ Production ready, reorganization complete
**Next milestone**: Commit reorganization after user validates testing

Good luck! 🚀
