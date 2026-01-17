# AI Assistant Handoff Document

**Last Updated**: 2026-01-16
**Status**: ✅ Production Ready (reorg complete; see Known Issues)

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
3. **deployment/gym_env.py** - Had `from robot_kinematics import ...` instead of `from .robot_kinematics import ...`
4. **scripts/verify_encoder_implementation.py** - Used `from compact_gym.robot_hardware` instead of `from deployment.robot_hardware`
5. **scripts/test_encoder_polling.py** - Used `from compact_gym import WX200GymEnv` instead of proper package imports

### Package Structure Files

Each package has an `__init__.py`:

**deployment/__init__.py** - Package marker; keep it import-light (no eager imports):
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

**Critical performance feature** (implemented in `collect_demo_gym.py`):

- **Inner loop (~120Hz)**: `env.step()` with cached action (IK + motor commands)
- **Outer loop (~10Hz)**: SpaceMouse input, GUI commands, encoder polling, recording
- **ArUco thread (~camera_fps)**: Camera + marker detection in background

This prevents jitter while keeping data collection lightweight.

**Implementation notes**:
- `WX200GymEnv` is single-step; it does **not** run the inner loop for you.
- Teleop uses `loop_rate_limiters.RateLimiter` to hold ~120Hz.
- Encoder polling happens via `env.poll_encoders()` in the outer loop.
- `env.step()` uses `dt = 1 / control_frequency` from its constructor (pass `inner_control_frequency` for 120Hz stepping).

### Data Flow

**During teleoperation**:
1. SpaceMouse raw input → normalized action `[-1, 1]^7`
2. `env.step(action)` → denormalizes to linear/angular velocities + gripper position
3. IK solver → joint commands → motor positions at inner-loop rate
4. `env.poll_encoders()` (outer loop) → refresh encoder cache for observations
5. ArUco thread → updates `aruco_*` + `last_frame` at `camera_fps`
6. Recording (outer loop) → saves NPZ at `control_frequency` (then smoothing adds `smoothed_*` keys in-place)

**During NN deployment**:
1. Policy network → normalized action `[-1, 1]^7`
2. `env.step(action)` → same denormalization + control path
3. You must call `env.step()` at your desired control rate (set `control_frequency` accordingly)
4. No recording infrastructure needed

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
    gripper     # Gripper position (normalized): -1=open, +1=closed
]
```

**All velocities are in world frame**, not end-effector frame!

### Denormalization

Happens in `deployment/gym_env.py`:

```python
# Translation: [-1, 1] → [-velocity_scale, +velocity_scale] m/s
vel = action[:3] * robot_config.velocity_scale

# Rotation: [-1, 1] → [-angular_velocity_scale, +angular_velocity_scale] rad/s
ang = action[3:6] * robot_config.angular_velocity_scale

# Gripper: [-1, 1] → [gripper_open_pos, gripper_closed_pos] (meters)
grip = (action[6] + 1.0) / 2.0 * (robot_config.gripper_closed_pos - robot_config.gripper_open_pos) + robot_config.gripper_open_pos
```

**GOTCHA**: The gripper action is interpreted as a **position command** (normalized).  
`-1` → open, `+1` → closed; intermediate values are valid positions.

### Common Mistakes

❌ **Don't** pass velocity commands directly to `env.step()` - they must be normalized first
❌ **Don't** confuse world frame with end-effector frame
❌ **Don't** denormalize actions in the teleop layer - that's the env's job
❌ **Don't** pass gripper position in meters to `env.step()` (it expects normalized)
✅ **Do** keep all actions in `[-1, 1]` when calling `env.step()`
✅ **Do** let the environment handle all denormalization internally

See [docs/overview/ACTION_SPACE_NOTES.md](overview/ACTION_SPACE_NOTES.md) for more details.

---

## Key Files and Their Roles

### Deployment Package Files (self-contained for NN use)

**deployment/gym_env.py**
- Gymnasium environment (`WX200GymEnv`)
- Action denormalization + observation assembly
- ArUco background thread + frame storage
- Exposes `poll_encoders()`, `home()`, `reset_gripper()`, `emergency_stop()`

**deployment/robot_hardware.py**
- Hardware abstraction + MuJoCo state sync
- Encoder polling and cached encoder/EE pose state
- Startup/home/reset/shutdown sequences

**deployment/robot_kinematics.py**
- IK/FK + joint↔motor translation
- Gripper position control

**deployment/robot_driver.py**
- Dynamixel SDK driver (serial comms, motor commands)

**deployment/robot_config.py**
- Single source of truth for configuration (`robot_config`)

**deployment/camera.py**
- GStreamer-only camera capture
- ArUco pose estimation (with optional pose hold)

**deployment/fix_gstreamer_env.py**
- Ensures GStreamer environment is configured before `cv2` import

**deployment/profiling.py**
- Lightweight control loop + ArUco profiling

**deployment/wx200/** (directory)
- MuJoCo XML + mesh assets

### Collection Infrastructure Files

**collect_demo_gym.py** (root)
- Teleop + data recording loop (inner/outer rate)
- GUI + SpaceMouse integration
- Saves NPZ + optional in-place smoothing
- GUI commands: Home, Reset EE, Start Recording, Stop & Save, Stop & Discard

**collection/validate_demo.py**
- Validates demo NPZ schema + basic quality checks
- `action_normalized` and `metadata` are optional

**collection/fix_usb_latency.py**
- Sets USB latency timer to 1ms (auto-run on startup)

**collection/spacemouse/**
- `spacemouse_driver.py`: high-level SpaceMouse interface (stale-state reset)
- `spacemouse_reader.py`: multiprocessing reader (pyspacemouse)

**collection/utils/robot_control_gui.py**
- Tkinter GUI with buttons + status

### Scripts (dev/test utilities)

**scripts/test_env.py**
- Random-policy smoke test

**scripts/verify_teleop_gym.py**
- Manual teleop (runs until Ctrl+C)

**scripts/test_encoder_polling.py**
- Hardware encoder polling performance test

**scripts/verify_encoder_syntax.py**
- AST-only structure/syntax check (no hardware)

**scripts/verify_encoder_implementation.py**
- Import-level check for encoder polling API

**scripts/verify_aruco_thread_syntax.py**
- AST check for ArUco thread wiring

**scripts/smooth_aruco_trajectory.py**
- Smooth ArUco trajectories (atomic save; supports in-place)

**scripts/verify_smoothed_trajectory.py**
- Verify smoothed file matches original (except `smoothed_*` keys)

### Dataset Utilities

**merge_smoothed_trajectories.py** (root)
- Merge demo/trajectory NPZ files into a training dataset
- Uses `smoothed_aruco_*` keys when available

---

## Data Format

Demonstrations are saved as NumPy NPZ files in `data/gym_demos/`:

```python
{
    # Timestamps
    'timestamp': float[T],              # Wall clock timestamps (time.time(), seconds)

    # Robot state
    'state': float[T, 6],              # Joint angles from encoders
    'encoder_values': int[T, 7],       # Raw encoder values
    'ee_pose_encoder': float[T, 7],    # EE pose from FK [x,y,z, qw,qx,qy,qz]

    # Actions (denormalized velocity + gripper position)
    'action': float[T, 7],             # [v_x, v_y, v_z, ω_x, ω_y, ω_z, gripper_pos_m]
    'action_normalized': float[T, 7],  # Normalized action in [-1, 1]

    # Augmented actions (with axis-angle integration)
    'augmented_actions': float[T, 10], # vel(3) + ang(3) + axis_angle(3) + gripper_pos(1)

    # IK targets
    'ee_pose_target': float[T, 7],     # Target pose sent to IK solver

    # ArUco markers (if enabled)
    'object_pose': float[T, 7],        # Object pose in world (alias of aruco_object_in_world)
    'object_visible': float[T, 1],     # Object visibility flag
    'aruco_ee_in_world': float[T, 7],
    'aruco_object_in_world': float[T, 7],
    'aruco_ee_in_object': float[T, 7],
    'aruco_object_in_ee': float[T, 7],
    'aruco_visibility': float[T, 3],   # Order: [world, object, ee]

    # Camera (if enabled)
    'camera_frame': uint8[T, H, W, 3]  # RGB frames (downscaled by frame_downscale_factor)

    # Smoothing (added in-place by collector; script can also write _smoothed files)
    'smoothed_aruco_*': float[T, 7],

    # Metadata
    'metadata': dict,                  # created_at, file_name, config_snapshot (+ smoothing_note)
}
```

Note: `camera_frame` is always saved; if no frame is available it is a zero array.  
`metadata` is stored as a pickled dict — load with `np.load(..., allow_pickle=True)`.

**Key validation criteria** (from `collection/validate_demo.py`):
- ✅ All expected fields present
- ⚠ Optional fields (warn-only): `action_normalized`, `metadata`
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
    control_frequency: float = 10.0        # Outer loop (teleop/policy/recording)
    inner_control_frequency: float = 120.0 # Inner loop (motor commands)

    # Action scaling (m/s and rad/s)
    velocity_scale: float = 0.25
    angular_velocity_scale: float = 1.0
    velocity_limit: int = 40               # Motor velocity limit (Dynamixel)

    # Gripper limits
    gripper_open_pos: float = -0.026
    gripper_closed_pos: float = 0.0
    gripper_encoder_min: int = 1559
    gripper_encoder_max: int = 2776

    # Camera (GStreamer only)
    camera_id: int = 1                     # Maps to /dev/video{camera_id}
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    frame_downscale_factor: int = 4

    # ArUco
    aruco_marker_size_m: float = 0.030
    aruco_world_id: int = 0
    aruco_object_id: int = 2
    aruco_ee_id: int = 3
    aruco_max_preserve_frames: int = 0

    # Profiling + input
    control_perf_window_sec: float = 3.0
    control_perf_stats_interval: int = 500
    profiler_window_size: int = 200
    spacemouse_stale_timeout: float = 0.2
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

### 2. Control-Frequency Mismatch (Robot too fast/slow)

**Problem**: Robot moves much too fast or sluggish.

**Root cause**: `WX200GymEnv` uses `dt = 1 / control_frequency` from its constructor.  
If you call `env.step()` at 10Hz but pass `control_frequency=120`, the motion will be ~12× too fast (and vice versa).

**Fix**: Set `control_frequency` to the actual rate you call `env.step()`.  
In teleop, `collect_demo_gym.py` calls `env.step()` at ~120Hz and passes `inner_control_frequency`.

**Reference**: [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #5

### 3. Jittery Motion

**Problem**: Robot motion is stuttering or jerky.

**Root cause**: Inner loop not hitting ~120Hz, or encoder polling creeping into the inner loop.

**Fix**: Keep encoder polling in the outer loop (`env.poll_encoders()`), and run `env.step()` at inner-loop rate.

**Reference**: [docs/RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #6

### 4. Double Cleanup / Tkinter Crash on Exit

**Problem**: `Tcl_AsyncDelete` or Tkinter errors on shutdown.

**Root cause**: `collect_demo_gym.py` already cleans up in `DemoCollector.run()`, then `__main__` repeats cleanup.

**Fix**: Add a cleanup guard or remove the redundant cleanup block if this shows up.

### 5. Camera Initialization Fails

**Expected behavior**: System continues without camera; ArUco disabled.

**Common causes**:
- Missing GStreamer / pygobject packages
- Wrong `camera_id` in `robot_config.py`
- Camera not connected or busy
- Permission issues with `/dev/video*`

**Debug**: Check available cameras with `ls /dev/video*`

### 6. Gripper Not Moving

**Problem**: Gripper commands don't work, motor error state.

**Solution**: Call `env.reset()` or `env.reset_gripper()` to reboot/open the gripper motor (motor 7).

### 7. ArUco Dictionary Mismatch

**Problem**: Markers never detected despite camera working.

**Root cause**: `gym_env.py` uses `cv2.aruco.DICT_5X5_50` (hard-coded, not in config).

**Fix**: Update the dictionary in `deployment/gym_env.py` if your tags differ.

### 8. Import Errors After Reorganization

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

### 9. Encoder Lag Issues

**Problem**: Encoder readings lag behind actual robot state.

**Historical context**: This was a major issue that was fixed by:
1. Moving encoder polling OUT of the 120Hz inner loop
2. Only polling encoders at 10Hz in outer loop
3. Using cached encoder values from `robot_hardware.latest_encoder_values` (read by `env.step()`)

**If you see encoder lag**, verify encoder polling is NOT in the inner loop.

**Reference**: [docs/archived/PHASE1_COMPLETE.md](archived/PHASE1_COMPLETE.md)

### 10. README Trajectory Viewer Reference

**Problem**: `README.md` mentions `trajectory_viewer_gui.py`, but that file is not in `compact_gym/`.

**Fix**: Copy it in if needed (or update the README to avoid confusion).

---

## Testing and Verification

### Quick Verification

**Test 1: Data collection works**
```bash
python collect_demo_gym.py
# - Move robot with SpaceMouse
# - Use GUI buttons to start/save/discard
# - Close GUI to quit
```

**Test 2: Validate collected data**
```bash
python collection/validate_demo.py
```

**Test 3: Teleoperation without recording**
```bash
python scripts/verify_teleop_gym.py
```

**Test 4: Random policy smoke test**
```bash
python scripts/test_env.py
```

**Test 5: Encoder polling performance (hardware)**
```bash
python scripts/test_encoder_polling.py
```

### Syntax Checks (No Hardware Required)

```bash
python scripts/verify_encoder_syntax.py
python scripts/verify_encoder_implementation.py
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
- [ ] Start Recording button starts recording (red status)
- [ ] Stop & Save button saves demo (green status)
- [ ] Stop & Discard button discards demo
- [ ] NPZ file created in `data/gym_demos/`
- [ ] Validation passes for recorded demo
- [ ] `smoothed_aruco_*` keys present (in-place)
- [ ] Optional: verify `_smoothed.npz` files with `scripts/verify_smoothed_trajectory.py`

**Shutdown checks**:
- [ ] Close GUI window exits cleanly
- [ ] Ctrl+C triggers proper cleanup
- [ ] Robot returns to home
- [ ] Motors disable (torque off)
- [ ] No hanging processes

---

## Performance Benchmarks

**Targets (from config)**:
- Inner loop (motor commands): `inner_control_frequency` (default 120Hz)
- Outer loop (input/recording): `control_frequency` (default 10Hz)
- ArUco thread: `camera_fps` (default 30Hz)
- Recorded data: `control_frequency` (default 10Hz)

**If performance degrades**, check:
1. USB latency still at 1ms
2. Encoder polling remains in the outer loop only
3. ArUco thread not blocking the main loop
4. No unnecessary file I/O in the inner loop

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
- Non-blocking updates to `env.latest_aruco_obs`
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
    control_frequency=robot_config.control_frequency  # Match your env.step() rate
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

The environment returns a flat `np.ndarray` with 20 floats:

```python
obs = np.concatenate([
    aruco_object_in_world,  # 7D (zeros if ArUco disabled)
    robot_state,            # 6D joint angles
    ee_pose_debug           # 7D [x, y, z, qw, qx, qy, qz]
])
# shape: (20,)
```

`info` includes extra fields when hardware is initialized:
`encoder_values`, `qpos`, `ee_pose_fk`, and `raw_aruco`.

`WX200GymEnv.observation_space` is a `Box` with shape `(20,)`.

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
2. Updates shared `latest_aruco_obs` dictionary
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
- User input (GUI) doesn't need faster polling

**Result**: Smooth robot motion, responsive control, efficient data collection.

---

## Debugging Tips

### Enable Profiling

In `collect_demo_gym.py`, profiling is already enabled. Check output:

```
[CONTROL LOOP] Avg=8.3ms, Max=11.2ms, Iterations=500
  [BREAKDOWN AVG] Input=0.2ms, GUI=0.1ms, Command=0.3ms, Step=2.1ms, Record=0.4ms, Viz=3.2ms, VizFPS=29.8
  [INNER LOOP] Running at 120Hz (motor commands)
    [ENV.STEP] Avg~3.0s Denorm=0.1ms, Exec=1.8ms, Encoder=0.0ms, Obs=0.4ms, Info=0.1ms
[ARUCO STATS] Freq=29.9Hz, Total=33.1ms (Detect=8.4ms, Max=45.0ms)
```

If numbers look wrong, investigate:
- Inner loop > 10ms → Something blocking in step()
- Outer loop > 120ms → Encoder polling or ArUco check too slow
- ArUco thread > 50ms → Camera resolution too high or detection too slow

Tip: Set `robot_config.verbose_profiling = True` for more frequent `[ENV.STEP]` summaries.

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
    print(f"Target pos: {self.robot_hardware.robot_controller.get_target_position()}")
    print(f"Target quat: {self.robot_hardware.robot_controller.get_target_orientation_quat_wxyz()}")
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

**"ModuleNotFoundError: No module named 'loop_rate_limiters'"**
- Required by `collect_demo_gym.py`, install: `pip install loop-rate-limiters`

**"ModuleNotFoundError: No module named 'mink'"**
- Required by IK solver, install the `mink` package (see `robot_kinematics.py` imports)

**"ModuleNotFoundError: No module named 'robot_kinematics'"**
- Import error, should be `from .robot_kinematics import ...` in `deployment/` files

**"USB latency is 16ms"**
- Auto-fix should run, if not: `python collection/fix_usb_latency.py`

**"Camera initialization failed"**
- Expected if no camera, system continues without ArUco
- If camera exists, check `camera_id` in `robot_config.py`

**"GStreamer not available"**
- `deployment/camera.py` requires GStreamer + pygobject; install or disable camera/ArUco

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
- `MARKER_SIZE`, `AXIS_LENGTH`

---

## Git Workflow

### Branch / Status
Always check `git status` before making assumptions about local changes.  
Use `git branch` if you need to confirm the current branch.

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
- [ ] Action space: Normalized `[-1, 1]^7`, world frame velocities + gripper position
- [ ] Dual-frequency architecture: 120Hz inner, 10Hz outer
- [ ] Layer separation: Teleop → Env → Hardware (no pollution)
- [ ] User preferences: Don't over-engineer, prefer existing files, concise code
- [ ] Testing procedures: Always test before claiming success
- [ ] Git status: check `git status` before assuming local modifications

**When in doubt**: Re-read the code, document assumptions, and ask only if blocked.

---

**This handoff document was created**: 2026-01-16  
**Last tested**: 2026-01-16 (see docs/TESTING.md for current checklist)  
**Status**: ✅ Production Ready (reorg complete; validate after edits)

Good luck! 🚀
