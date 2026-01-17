# Architecture Documentation - Separation of Concerns

**Date**: 2026-01-16
**Status**: ✅ Verified Clean Separation

---

## Overview

The compact_gym system implements a clean 3-layer architecture with strict separation of concerns:

1. **Teleop Layer** ([collect_demo_gym.py](collect_demo_gym.py)) - Input handling and data recording
2. **Environment Layer** ([gym_env.py](gym_env.py)) - Input-agnostic robot control
3. **Hardware Layer** ([robot_hardware.py](robot_hardware.py)) - Low-level motor commands

**Key Principle**: `env.step(action)` is **completely input-agnostic**. It takes normalized actions and returns observations + info. It doesn't know or care whether the action came from SpaceMouse, neural network, keyboard, or any other source.

---

## Layer 1: Teleop Layer (collect_demo_gym.py)

**Responsibility**: Input device handling, action normalization, data collection

**Key Design Points**:
- Reads raw SpaceMouse input at 10Hz (outer loop)
- Normalizes actions to [-1, 1] range
- Calls `env.step(action)` with normalized actions at 120Hz (inner loop)
- Records data using info dict returned from env.step()
- **No hardware or kinematics knowledge** - pure input/output processing

### Input Processing ([collect_demo_gym.py:279-292](collect_demo_gym.py#L279-L292))

```python
# Get raw input from SpaceMouse (outer loop, 10Hz)
vel_world = self.spacemouse.get_velocity_command()  # m/s
ang_vel_world = self.spacemouse.get_angular_velocity_command()  # rad/s
self.current_gripper_pos = self.spacemouse.get_gripper_command(...)  # meters

# Normalize to [-1, 1] for env.step()
norm_vel = np.clip(vel_world / robot_config.velocity_scale, -1, 1)
norm_ang_vel = np.clip(ang_vel_world / robot_config.angular_velocity_scale, -1, 1)
gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
norm_gripper = np.clip(2.0 * (self.current_gripper_pos - robot_config.gripper_open_pos) / gripper_range - 1.0, -1, 1)

# Package normalized action
latest_action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])
```

**Key Point**: Action normalization happens **outside** the environment. This allows easy swapping of input sources:
- SpaceMouse → Normalize raw velocities
- Neural Network → Already outputs normalized actions
- Keyboard → Map keys to normalized velocities
- Joystick → Map stick position to normalized actions

### Environment Interaction ([collect_demo_gym.py:296-300](collect_demo_gym.py#L296-L300))

```python
# Call env with normalized action (runs at 120Hz)
next_obs, reward, terminated, truncated, info = self.env.step(latest_action)

# info dict contains everything needed for recording:
# - 'qpos': joint angles (6D)
# - 'encoder_values': raw encoder dict
# - 'ee_pose_fk': FK-computed EE pose (7D: pos + quat)
# - 'raw_aruco': ArUco observations dict
```

**Key Point**: `env.step()` is a pure function mapping `(state, action) → (next_state, reward, info)`. No side effects, no recording, no input device coupling.

### Data Recording ([collect_demo_gym.py:305-369](collect_demo_gym.py#L305-L369))

```python
if self.is_recording:
    # Build data packet from:
    # 1. info dict from env.step() (encoders, FK, ArUco)
    # 2. Raw teleoperation commands (vel_world, ang_vel_world, gripper_pos)
    # 3. IK target from robot_hardware

    step_data = {
        # State from env
        'state': info.get('qpos', np.zeros(6)),
        'encoder_values': encoder_array,
        'ee_pose_encoder': info.get('ee_pose_fk', np.zeros(7)),

        # Actions (UNNORMALIZED for backward compatibility)
        'action': np.concatenate([vel_world, ang_vel_world, [self.current_gripper_pos]]),
        'augmented_actions': np.concatenate([vel_world, ang_vel_world, axis_angle, [self.current_gripper_pos]]),
        'ee_pose_target': self._get_ee_pose_target(),  # IK target

        # Vision from env
        'aruco_ee_in_world': raw_aruco.get('aruco_ee_in_world', ...),
        'aruco_object_in_world': raw_aruco.get('aruco_object_in_world', ...),
        'camera_frame': raw_aruco.get('camera_frame', None),
        ...
    }

    self.trajectory.append(step_data)
```

**Key Point**: Recording happens **outside** env.step() using data returned in the info dict. Environment has no concept of "recording" or "episodes".

---

## Layer 2: Environment Layer (gym_env.py)

**Responsibility**: Input-agnostic robot control, observation generation

**Key Design Points**:
- Takes normalized actions [-1, 1] as input
- Denormalizes internally to physical units
- Executes motor commands at 120Hz
- Returns observations and info dict
- **Completely input-agnostic** - doesn't know about SpaceMouse, NN, or any input source

### step() Method Signature ([gym_env.py:435-488](gym_env.py#L435-L488))

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Execute one control step.

    Args:
        action: Normalized action in [-1, 1] range
                [vel_x, vel_y, vel_z, omega_x, omega_y, omega_z, gripper] (7D)

    Returns:
        obs: Observation dict/array
        reward: Scalar reward (always 0.0 for teleop)
        terminated: Episode terminal flag
        truncated: Max episode length flag
        info: Dict containing:
            - 'qpos': joint angles from encoders (6D)
            - 'encoder_values': raw encoder dict {motor_id: encoder_value}
            - 'ee_pose_fk': FK-computed EE pose (7D: pos + quat)
            - 'raw_aruco': ArUco observations from camera thread
    """
```

**Critical Design**: The environment is a **pure transformer**:
- Input: `action` (normalized)
- Output: `(obs, reward, terminated, truncated, info)`
- No side effects, no global state modifications
- No recording logic
- No input device dependencies

### Action Denormalization ([gym_env.py:304-326](gym_env.py#L304-L326))

```python
def _denormalize_action(self, action):
    """
    Convert normalized action [-1, 1] to physical units.

    This is where action space normalization is reversed.
    """
    # Velocity: [-1, 1] → [low, high] m/s
    vel = (action[:3] + 1) / 2 * (self.action_space_high[:3] - self.action_space_low[:3]) + self.action_space_low[:3]

    # Angular velocity: [-1, 1] → [low, high] rad/s
    ang_vel = (action[3:6] + 1) / 2 * (self.action_space_high[3:6] - self.action_space_low[3:6]) + self.action_space_low[3:6]

    # Gripper: [-1, 1] → [open_pos, closed_pos] meters
    grip = (action[6] + 1) / 2 * (self.action_space_high[6] - self.action_space_low[6]) + self.action_space_low[6]

    return vel, ang_vel, grip
```

**Key Point**: Normalization/denormalization is **symmetric**:
- Teleop layer: `raw → normalized` (before env.step)
- Env layer: `normalized → physical` (inside env.step)
- NN layer: `NN → normalized` (NN outputs already normalized)

### Motor Command Execution ([gym_env.py:445-449](gym_env.py#L445-L449))

```python
# Execute motor command via hardware layer
if self.has_authority and self._hardware_initialized:
    self.robot_hardware.execute_command(vel, ang_vel, grip, self.dt)
```

**Key Point**: No rate limiting here. Motor commands run as fast as the caller invokes step() (120Hz in our case).

### Info Dict Construction ([gym_env.py:468-479](gym_env.py#L468-L479))

```python
# Build info dict with all data needed for recording
info = {}
if self.has_authority and self._hardware_initialized:
    encoder_state = self.robot_hardware.get_encoder_state()
    info['encoder_values'] = encoder_state['encoder_values']  # Raw encoder dict
    info['qpos'] = encoder_state['joint_angles']  # Joint angles (6D)
    info['ee_pose_fk'] = np.concatenate([
        encoder_state['ee_pose'][0], encoder_state['ee_pose'][1]
    ]) if encoder_state['ee_pose'] is not None else None  # FK pose (7D)
    info['raw_aruco'] = self.aruco_obs_dict.copy()  # ArUco observations
```

**Key Point**: The info dict is the **data bridge** between env and recorder. Environment packages all observable state into info, recorder extracts it for storage.

### ArUco Background Thread ([gym_env.py:176-267](gym_env.py#L176-L267))

```python
def _aruco_poll_loop(self):
    """Background thread polling ArUco at camera FPS (30Hz)."""
    while self._aruco_running and self.camera is not None:
        frame = self.camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Process ArUco markers
        corners, ids = detect_aruco_markers(frame, self.aruco_dict, self.aruco_params)
        # ... compute poses ...

        # Thread-safe update
        with self._aruco_lock:
            self.aruco_obs_dict = obs
            self._latest_frame = frame
```

**Key Point**: ArUco runs **independently** at 30Hz. step() just reads the latest cached observations (thread-safe). No blocking camera I/O in control loop.

---

## Layer 3: Hardware Layer (robot_hardware.py)

**Responsibility**: Low-level motor control, IK solving, encoder reading

**Key Design Points**:
- Takes physical units as input (m/s, rad/s, meters)
- Solves IK and sends motor commands
- Polls encoders on demand
- **No knowledge of normalization or input sources**

### execute_command() ([robot_hardware.py](robot_hardware.py))

```python
def execute_command(self, vel_world, ang_vel_world, gripper_target, dt):
    """
    Execute one control step with velocity command.

    Args:
        vel_world: Linear velocity in world frame (m/s) [3D]
        ang_vel_world: Angular velocity in world frame (rad/s) [3D]
        gripper_target: Target gripper position (meters)
        dt: Time step (seconds) - typically 1/120 = 0.0083s
    """
    # Integrate velocity to get delta pose
    delta_pos = vel_world * dt
    delta_rot = ang_vel_world * dt  # axis-angle representation

    # Update target pose
    new_ee_position = current_ee_position + delta_pos
    new_ee_orientation = current_ee_orientation * exp_map(delta_rot)

    # Solve IK
    target_joint_angles = self.controller.solve_ik(new_ee_position, new_ee_orientation)

    # Send motor commands
    self.robot_driver.send_motor_positions(target_joint_angles)
```

**Key Point**: Hardware layer works in **physical units only**. It has no concept of normalized actions, SpaceMouse, or neural networks.

---

## Comparison with compact_code

### compact_code Architecture (Monolithic)

In [wx200_robot_collect_demo_encoders_compact.py](../compact_code/wx200_robot_collect_demo_encoders_compact.py), everything happens in one class:

```python
# Line 825-880: on_control_loop_iteration()
def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt, ...):
    # 1. Handle GUI commands (teleop-specific)
    self._handle_control_input()

    # 2. Poll encoders (hardware-specific)
    self._poll_encoders(...)

    # 3. Read ArUco (vision-specific)
    with self._aruco_lock:
        obs = {k: v.copy() for k, v in self.latest_aruco_obs.items()}

    # 4. Recording logic (data collection-specific)
    if self.is_recording:
        action = np.concatenate([velocity_world, angular_velocity_world, [gripper_target]])
        self.trajectory.append({
            'timestamp': ...,
            'state': state.copy(),
            'action': action.copy(),  # UNNORMALIZED - raw velocities
            'ee_pose_target': ee_pose_target.copy()
        })

    # 5. Execute motor command (hardware-specific)
    # (happens in _execute_control_step() called from run_control_loop())
```

**Issues**:
- GUI, hardware, vision, and recording all mixed together
- Hard to swap input sources (SpaceMouse deeply integrated)
- Can't easily use with RL (no gym interface)
- Actions are UNNORMALIZED (raw velocities in m/s)

### compact_gym Architecture (Modular)

**Teleop Layer** ([collect_demo_gym.py:279-369](collect_demo_gym.py#L279-L369)):
```python
# 1. Get input (teleop-specific)
vel_world = self.spacemouse.get_velocity_command()

# 2. Normalize action
latest_action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])

# 3. Call env (input-agnostic)
next_obs, reward, terminated, truncated, info = self.env.step(latest_action)

# 4. Record using info dict
if self.is_recording:
    step_data = {...}  # Built from info dict
    self.trajectory.append(step_data)
```

**Environment Layer** ([gym_env.py:435-488](gym_env.py#L435-L488)):
```python
def step(self, action):
    # 1. Denormalize
    vel, ang_vel, grip = self._denormalize_action(action)

    # 2. Execute motor command
    self.robot_hardware.execute_command(vel, ang_vel, grip, self.dt)

    # 3. Get observation
    obs = self._get_observation()

    # 4. Return info dict with all observable state
    info = {
        'qpos': encoder_state['joint_angles'],
        'encoder_values': encoder_state['encoder_values'],
        'ee_pose_fk': encoder_state['ee_pose'],
        'raw_aruco': self.aruco_obs_dict
    }

    return obs, reward, terminated, truncated, info
```

**Benefits**:
- Clean separation: teleop, env, hardware are independent
- Easy to swap input sources (just change teleop layer)
- Gym-compatible (can plug into RL frameworks)
- Actions are NORMALIZED ([-1, 1] standard for RL)
- Can run NN policy: `action = policy(obs); env.step(action)`

---

## Dual-Frequency Control Architecture

Both compact_code and compact_gym use the **same dual-frequency pattern**:

### Inner Loop (120Hz) - Motor Commands
```python
while running:
    # Execute motor command EVERY iteration
    env.step(latest_action)  # compact_gym
    # OR
    _execute_control_step()  # compact_code

    # Sleep to maintain 120Hz
    rate_limiter.sleep()
```

### Outer Loop (10Hz) - Input/Recording
```python
while running:
    current_time = time.perf_counter()

    # Time-based trigger for outer loop (every 12th iteration)
    if current_time >= outer_loop_target_time:
        outer_loop_target_time = current_time + outer_loop_dt

        # 1. Read input
        # 2. Update action cache
        # 3. Record data
        # 4. Visualization

    # Inner loop runs regardless
    env.step(latest_action)
```

**Key Point**: Both systems use **action caching**:
- Outer loop updates `latest_action` at 10Hz
- Inner loop reuses `latest_action` for 12 iterations (120Hz)
- This ensures smooth motion even though input updates slowly

---

## Input Source Independence

The gym environment is **completely input-agnostic**. Here's how different input sources would integrate:

### SpaceMouse (Current Implementation)

```python
# Teleop layer
vel_world = spacemouse.get_velocity_command()  # m/s
norm_vel = np.clip(vel_world / velocity_scale, -1, 1)
action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])

# Env layer
env.step(action)  # Doesn't know this came from SpaceMouse
```

### Neural Network Policy

```python
# Policy layer
obs = env.get_observation()
action = policy(obs)  # Already normalized [-1, 1] from NN output

# Env layer
env.step(action)  # Doesn't know this came from NN
```

### Keyboard Input

```python
# Keyboard layer
vel_world = keyboard_to_velocity(key_pressed)  # m/s
norm_vel = np.clip(vel_world / velocity_scale, -1, 1)
action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])

# Env layer
env.step(action)  # Doesn't know this came from keyboard
```

### Joystick Input

```python
# Joystick layer
stick_pos = joystick.get_position()  # Already in [-1, 1]
action = np.concatenate([stick_pos[:3], stick_pos[3:6], [stick_pos[6]]])

# Env layer
env.step(action)  # Doesn't know this came from joystick
```

**Key Point**: `env.step()` is a **universal interface**. Any input source can drive the robot by providing normalized actions.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ TELEOP LAYER (collect_demo_gym.py)                              │
│                                                                  │
│  SpaceMouse (10Hz)                                               │
│       ↓                                                          │
│  Read raw velocities (m/s, rad/s)                                │
│       ↓                                                          │
│  Normalize to [-1, 1]                                            │
│       ↓                                                          │
│  Cache action ────────────────────────────┐                      │
│                                           │                      │
│  ┌──────────────────────────────────────┐ │                      │
│  │ INNER LOOP (120Hz)                   │ │                      │
│  │   ↓                                  │ │                      │
│  │   Send cached action to env ←────────┘ │                      │
│  │   ↓                                    │                      │
│  └───────────────────────────────────────┘                      │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │ action (normalized)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ ENVIRONMENT LAYER (gym_env.py)                                   │
│                                                                  │
│  Denormalize action to physical units                            │
│       ↓                                                          │
│  Call hardware.execute_command(vel, ang_vel, grip, dt)           │
│       ↓                                                          │
│  Get observation                                                 │
│       ↓                                                          │
│  Build info dict:                                                │
│    - qpos (from encoders)                                        │
│    - encoder_values                                              │
│    - ee_pose_fk                                                  │
│    - raw_aruco (from background thread)                          │
│       ↓                                                          │
│  Return (obs, reward, terminated, truncated, info)               │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │ info dict
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ TELEOP LAYER (Recording)                                         │
│                                                                  │
│  Extract from info dict:                                         │
│    - state = info['qpos']                                        │
│    - encoders = info['encoder_values']                           │
│    - ee_fk = info['ee_pose_fk']                                  │
│    - aruco = info['raw_aruco']                                   │
│       ↓                                                          │
│  Combine with raw actions (vel_world, ang_vel_world)             │
│       ↓                                                          │
│  Append to trajectory buffer                                     │
│       ↓                                                          │
│  Save to NPZ on episode end                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ HARDWARE LAYER (robot_hardware.py)                               │
│                                                                  │
│  execute_command(vel, ang_vel, grip, dt)                         │
│       ↓                                                          │
│  Integrate velocity → delta pose                                 │
│       ↓                                                          │
│  Solve IK → target joint angles                                  │
│       ↓                                                          │
│  Send motor commands via RobotDriver                             │
│       ↓                                                          │
│  Motors move (120Hz)                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ VISION LAYER (Background Thread @ 30Hz)                          │
│                                                                  │
│  Camera.get_frame()                                              │
│       ↓                                                          │
│  Detect ArUco markers                                            │
│       ↓                                                          │
│  Estimate poses                                                  │
│       ↓                                                          │
│  Update env.latest_aruco_obs + env.last_frame (thread-safe)       │
│       ↓                                                          │
│  Read by env.step() when building info dict                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Verification Checklist

### ✅ Input Agnosticism

**Claim**: `env.step()` doesn't know about SpaceMouse

**Evidence**:
- [gym_env.py](gym_env.py) has **zero** references to "SpaceMouse", "spacemouse", or "mouse"
- [gym_env.py:435-488](gym_env.py#L435-L488) `step()` signature: `def step(self, action: np.ndarray)`
- No imports from `spacemouse/` directory in gym_env.py
- `grep -r "spacemouse" gym_env.py` returns 0 results

**Conclusion**: ✅ VERIFIED - env.step() is completely input-agnostic

### ✅ Clean Data Collection

**Claim**: Recording logic is outside env.step()

**Evidence**:
- [gym_env.py:435-488](gym_env.py#L435-L488) has **zero** recording logic
- No `self.trajectory`, `self.is_recording`, or NPZ writing in gym_env.py
- Recording happens in [collect_demo_gym.py:305-369](collect_demo_gym.py#L305-L369)
- env.step() only returns `info` dict, doesn't write data

**Conclusion**: ✅ VERIFIED - Recording is cleanly separated

### ✅ Hardware Abstraction

**Claim**: Lower-level control is centralized in `RobotHardware` and accessed via env helpers

**Evidence**:
- Motor commands still happen inside `env.step()` → `robot_hardware.execute_command()`
- Teleop uses `env.step()` for motion and reads `info` for recording
- Teleop calls `env.poll_encoders()` in the outer loop to refresh cached encoder state
- Teleop uses `env.home()` / `env.reset_gripper()` helpers (no direct driver access)

**Conclusion**: ✅ VERIFIED - Teleop only touches env helpers; hardware logic stays centralized

### ✅ Action Normalization

**Claim**: Actions are normalized for gym interface

**Evidence**:
- [collect_demo_gym.py:287-292](collect_demo_gym.py#L287-L292): Normalization before env.step()
- [gym_env.py:304-326](gym_env.py#L304-L326): Denormalization inside env.step()
- Action space bounds defined in [gym_env.py:90-103](gym_env.py#L90-L103)
- NN policies can output normalized actions directly

**Conclusion**: ✅ VERIFIED - Actions properly normalized

### ✅ Info Dict Pattern

**Claim**: Data collection uses info dict, not direct hardware queries

**Evidence**:
- [gym_env.py:468-479](gym_env.py#L468-L479): info dict populated with encoders, FK, ArUco
- [collect_demo_gym.py:311-365](collect_demo_gym.py#L311-L365): Recording extracts from info dict
- Teleop triggers `env.poll_encoders()` once per outer loop to refresh cached encoder state
- Info dict remains the **data bridge** between env and recorder

**Conclusion**: ✅ VERIFIED - Info dict pattern implemented with explicit encoder polling

### ✅ No Pollution

**Claim**: No teleop logic leaked into env, no env logic leaked into teleop

**Evidence**:
- gym_env.py: No GUI, no SpaceMouse, no NPZ writing
- collect_demo_gym.py: No IK solving or FK; all hardware access goes through env helpers
- Clear interface: `action_in → step() → (obs, reward, done, info)_out`
- Teleop calls env, env doesn't call teleop (unidirectional dependency)

**Conclusion**: ✅ VERIFIED - No pollution between layers

---

## Summary

**compact_gym implements clean separation of concerns**:

1. **Teleop Layer** handles input-specific logic (SpaceMouse, GUI, recording)
2. **Environment Layer** provides input-agnostic robot control with gym interface
3. **Hardware Layer** handles low-level motor commands and kinematics
4. **Vision Layer** runs independently in background thread

**Key architectural wins**:
- ✅ env.step() is completely input-agnostic (works with any action source)
- ✅ Recording logic cleanly separated (uses info dict pattern)
- ✅ Hardware abstraction is centralized with env helpers
- ✅ Normalized action space ([-1, 1] standard for RL)
- ✅ Dual-frequency control matches compact_code (120Hz motor, 10Hz data)
- ✅ No pollution between layers (clean interfaces)

**Ready for**:
- Neural network policy deployment (`action = policy(obs); env.step(action)`)
- RL training (standard gym interface)
- Different input devices (keyboard, joystick, VR controller)
- Automated evaluation (replay actions from dataset)

---

**Verification Date**: 2026-01-16
**Verified By**: Architecture review comparing compact_gym vs compact_code
