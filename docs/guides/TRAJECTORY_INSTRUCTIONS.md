# Trajectory Recording and Replay Instructions

## Overview

This guide explains how to record demonstration trajectories and replay them on the real robot for BC (Behavioral Cloning) policy training.

## Recording a Trajectory

### Step 1: Start Recording

Run the recording script with the `--record` flag:

```bash
python wx200_real_robot_spacemouse_control_record.py --record
```

This will:
- Connect to the robot
- Move robot to home position
- Start recording trajectory data
- Save to `trajectory_YYYYMMDD_HHMMSS.npz` by default

### Step 2: Control the Robot

Use the SpaceMouse to control the robot:
- **Translation**: Push/pull the SpaceMouse to move end-effector in world frame
- **Rotation**: Twist the SpaceMouse to rotate end-effector in world frame
- **Gripper**: 
  - Hold left button = open incrementally
  - Hold right button = close incrementally

### Step 3: Stop Recording

Press `Ctrl+C` to stop recording. The script will:
- Save the trajectory to the output file
- Execute shutdown sequence
- Disconnect from robot

### Custom Output Filename

To specify a custom output filename:

```bash
python wx200_real_robot_spacemouse_control_record.py --record --output my_demo.npz
```

## Replaying a Trajectory

### Basic Replay

To replay an entire trajectory:

```bash
python wx200_real_robot_replay_trajectory.py trajectory_20240101_120000.npz
```

This will:
- Connect to the robot
- Move robot to home position
- Replay the recorded actions at the same frequency
- Execute shutdown sequence on completion

### Partial Replay

To replay a specific segment of the trajectory:

```bash
# Replay from index 100 to 500
python wx200_real_robot_replay_trajectory.py trajectory.npz --start-index 100 --end-index 500
```

## Recorded Data Format

The trajectory file (`.npz`) contains:

- **`timestamps`**: Array of relative timestamps (starting at t=0)
- **`states`**: Array of shape `(N, 6)` - joint positions
  - First 5 values: Arm joint angles (radians)
  - Last value: Gripper position (meters)
- **`actions`**: Array of shape `(N, 6)` - velocity commands
  - First 3 values: Linear velocity `[vx, vy, vz]` (m/s)
  - Last 3 values: Angular velocity `[wx, wy, wz]` (rad/s)
- **`metadata`**: Dictionary with recording info (frequency, duration, etc.)

## Tips

1. **Recording Quality**: 
   - Record smooth, consistent motions
   - Avoid sudden jerky movements
   - Practice the task a few times before recording

2. **Replay Verification**:
   - Start with short segments (`--start-index` and `--end-index`) to verify replay works
   - Check that the robot follows the recorded trajectory accurately

3. **Multiple Demonstrations**:
   - Record multiple trajectories for the same task
   - Use descriptive filenames: `pick_demo_1.npz`, `pick_demo_2.npz`, etc.

4. **Safety**:
   - Always ensure the workspace is clear before replaying
   - Be ready to press Ctrl+C if the robot behaves unexpectedly
   - The shutdown sequence will execute automatically on exit
