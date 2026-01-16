# Action Space Notes

## Normalized Action Space

The gym environment uses a **normalized action space** in the range `[-1, 1]` for all action dimensions.

### Action Vector Format

```python
action = [vx, vy, vz, wx, wy, wz, gripper]  # All in range [-1, 1]
```

### Denormalization Mapping

The `_denormalize_action()` method maps normalized actions to physical units:

```python
denormalized = (action + 1.0) / 2.0 * (action_high - action_low) + action_low
```

This means:
- `action = -1.0` → `action_low` (minimum value)
- `action =  0.0` → midpoint between low and high
- `action = +1.0` → `action_high` (maximum value)

### Gripper Action Space

**Important**: The gripper action is **NOT** a delta or velocity - it's a **target position**.

| Normalized Action | Physical Target | Gripper State |
|-------------------|-----------------|---------------|
| `-1.0` | `gripper_open_pos = -0.026m` | Fully Open |
| `0.0` | `-0.013m` (halfway) | Half Closed |
| `+1.0` | `gripper_closed_pos = 0.0m` | Fully Closed |

**Common Mistake**:
```python
# WRONG - gripper will close to halfway!
action = np.zeros(7)
env.step(action)

# CORRECT - gripper stays open
action = np.zeros(7)
action[6] = -1.0  # Explicitly set to open position
env.step(action)
```

### No-Op Action

To send a "no operation" action (no movement), use:

```python
noop_action = np.zeros(7)  # Zero velocity/angular velocity
noop_action[6] = -1.0      # Maintain open gripper position
```

Or to maintain current gripper position, you need to track and set it:

```python
# Track current gripper state
current_gripper_normalized = ...  # From previous action or observation

noop_action = np.zeros(7)
noop_action[6] = current_gripper_normalized
```

### Velocity Actions (First 6 Dimensions)

For the velocity and angular velocity components:

| Component | Range | Physical Unit |
|-----------|-------|---------------|
| `vx, vy, vz` | `[-1, 1]` | Maps to `[-0.25, 0.25]` m/s |
| `wx, wy, wz` | `[-1, 1]` | Maps to `[-1.0, 1.0]` rad/s |

These are **velocity commands**, so:
- `action = 0.0` → zero velocity (no movement) ✓
- `action = -1.0` → maximum negative velocity
- `action = +1.0` → maximum positive velocity

### Test Script Fix

The test script was updated to use the correct no-op action:

```python
# Before (WRONG - caused gripper to close halfway)
zero_action = np.zeros(7)
env.step(zero_action)

# After (CORRECT - gripper stays open)
noop_action = np.zeros(7)
noop_action[6] = -1.0  # Keep gripper open
env.step(noop_action)
```

### Why This Design?

This normalized action space is standard for RL:

1. **Consistent range**: All actions in `[-1, 1]` simplifies neural network outputs
2. **Easy clipping**: Networks can use `tanh` activation
3. **Policy learning**: Symmetric range around zero helps gradient flow

The tradeoff is that gripper control is **positional** not **incremental**, which means:
- RL policy must output absolute gripper positions
- Can't easily do "close a little bit more" without tracking state
- For teleoperation, need to maintain gripper state externally

### Alternative for Data Collection

For teleoperation (data collection), you might want to track gripper position:

```python
class TeleopWrapper:
    def __init__(self):
        self.current_gripper_pos = -1.0  # Start open

    def step(self, spacemouse_input):
        # Update gripper position based on button presses
        if spacemouse.left_button:
            self.current_gripper_pos = max(-1.0, self.current_gripper_pos - 0.1)
        if spacemouse.right_button:
            self.current_gripper_pos = min(1.0, self.current_gripper_pos + 0.1)

        # Build action
        action = np.concatenate([
            spacemouse.velocity,        # [vx, vy, vz]
            spacemouse.angular_velocity, # [wx, wy, wz]
            [self.current_gripper_pos]   # Tracked position
        ])

        return env.step(action)
```

This is exactly what `compact_code/wx200_robot_teleop_control.py` does with `gripper_current_position`.

## Configuration

Denormalization parameters in `robot_config.py`:

```python
# Velocity scaling
velocity_scale: float = 0.25  # m/s
angular_velocity_scale: float = 1.0  # rad/s

# Gripper positions
gripper_open_pos: float = -0.026  # meters
gripper_closed_pos: float = 0.0  # meters
```

## See Also

- `gym_env.py:353-367` - `_denormalize_action()` implementation
- `test_encoder_polling.py:66-68` - Correct no-op action example
- `compact_gym/collect_demo_gym.py` - Full teleoperation example with gripper tracking
