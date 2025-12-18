# Dependencies for BC Algorithm Integration

This document lists all files and folders required to use `wx200_gym_env_utils.py` with your BC algorithm.

## Required Python Packages (install via pip/conda)

These are external packages that need to be installed:
- `gymnasium` (or `gym`)
- `numpy`
- `scipy`
- `opencv-python` (cv2)
- `mujoco` (MuJoCo Python bindings)
- `mink` (mink-ik package)
- `dynamixel-sdk` (for robot hardware control)
- `loop_rate_limiters` (optional, has fallback)

## Required Folders and Files

### 1. Core Environment File
```
wx200_gym_env_utils.py
```

### 2. Robot Control Module (`robot_control/` folder)
**All files in this folder are required:**
```
robot_control/
├── __init__.py
├── robot_config.py          # Configuration (EE bounds, motor IDs, etc.)
├── robot_control_base.py    # Main control base class
├── robot_controller.py       # IK controller
├── robot_driver.py          # Dynamixel motor driver
├── robot_joint_to_motor.py  # Joint/motor translation
├── ee_pose_controller.py    # End-effector pose controller
├── robot_startup.py         # Startup sequence
└── robot_shutdown.py        # Shutdown sequence
```

### 3. Camera Module (`camera/` folder)
**All files in this folder are required:**
```
camera/
├── __init__.py
├── gstreamer_camera.py      # GStreamer camera (high performance)
└── opencv_camera.py         # OpenCV fallback camera
```

### 4. ArUco Module (now in camera/)
```
camera/aruco_pose_estimator.py      # ArUco marker pose estimation
```

### 5. MuJoCo Model Files (`wx200/` folder)
**Required for robot model:**
```
wx200/
├── robot.xml                # Main robot model (REQUIRED)
├── scene.xml                 # Scene file (if used)
└── assets/                  # Model assets (meshes, textures, etc.)
    ├── base.part
    ├── link1.part
    ├── link2.part
    ├── link3.part
    ├── link4.part
    ├── link5.part
    ├── gripper.part
    └── ... (all mesh files)
```

**Note:** The `wx200/` folder path is hardcoded in some places. You may need to update paths or ensure it's accessible.

## Import Structure

The code expects this import structure:
```python
from robot_control.robot_config import robot_config
from robot_control.robot_control_base import RobotControlBase
from robot_control.robot_joint_to_motor import sync_robot_to_mujoco
from camera import Camera, is_gstreamer_available, ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix
```

## Minimal Setup for BC Algorithm

If you only need the environment wrapper (no hardware), you might be able to skip:
- `robot_driver.py` (only needed for real hardware)
- `robot_shutdown.py` (only needed for real hardware)
- `camera/` folder (if `enable_aruco=False`, but still needed for Camera class)

However, the imports are still required, so you'd need to create stub files or modify the code.

## Recommended Approach

1. **Copy the entire `robot_control/` folder** - all files are interconnected
2. **Copy the `camera/` folder** - needed for ArUco tracking
3. **Copy `wx200/` folder** - needed for MuJoCo model
5. **Copy `wx200_gym_env_utils.py`** - the main environment file

## Path Dependencies

**IMPORTANT:** The folder structure must be maintained:

```
your_project/
├── wx200_gym_env_utils.py
├── aruco_pose_estimator.py
├── robot_control/
│   ├── __init__.py
│   ├── robot_config.py
│   ├── robot_control_base.py  # Line 22: expects wx200/ at parent level
│   └── ...
├── camera/
│   ├── __init__.py
│   └── ...
└── wx200/                      # Must be at same level as robot_control/
    ├── robot.xml
    ├── scene.xml
    └── assets/
```

The path `wx200/scene.xml` is hardcoded in `robot_control_base.py` line 22:
```python
_XML = Path(__file__).parent.parent / "wx200" / "scene.xml"
```

This means `wx200/` must be a sibling directory of `robot_control/`.

Other hardcoded paths to check:
- `robot_control/robot_config.py` - may have device paths like `/dev/ttyUSB0` (for real hardware)

## Testing

After copying, test with:
```python
from wx200_gym_env_utils import make_wx200_env

env = make_wx200_env(
    max_episode_length=1000,
    enable_aruco=False,  # If you don't need camera
    show_video=False,
    seed=0
)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
```

