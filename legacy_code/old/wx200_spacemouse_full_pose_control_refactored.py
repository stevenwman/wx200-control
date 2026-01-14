"""
WX200 robot full pose control using SpaceMouse (refactored version).

This version uses a modular EndEffectorPoseController class that takes 6-vector
velocity commands and produces mink-compatible targets. This makes it easy to
replace SpaceMouse input with neural network predictions.
"""
from pathlib import Path
import multiprocessing as mp
import queue
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import time
import mink
from spacemouse_reader import spacemouse_process
from ee_pose_controller import EndEffectorPoseController

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20

# SpaceMouse scaling for velocity control
VELOCITY_SCALE = 0.5  # m/s per unit SpaceMouse input
ANGULAR_VELOCITY_SCALE = 0.5  # rad/s per unit SpaceMouse rotation input


def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """
    Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False.
    """
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        # Only checking the first FrameTask here (end_effector_task).
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False


def add_world_frame_axes(scene, origin, length=0.15):
    """
    Add world frame coordinate axes (X, Y, Z) as colored arrows.
    MuJoCo convention: Red=X, Green=Y, Blue=Z
    """
    # X-axis (Red)
    x_end = origin + np.array([length, 0, 0])
    add_visual_arrow(scene, origin, x_end, radius=0.003, rgba=(1.0, 0.0, 0.0, 0.8))
    
    # Y-axis (Green)
    y_end = origin + np.array([0, length, 0])
    add_visual_arrow(scene, origin, y_end, radius=0.003, rgba=(0.0, 1.0, 0.0, 0.8))
    
    # Z-axis (Blue)
    z_end = origin + np.array([0, 0, length])
    add_visual_arrow(scene, origin, z_end, radius=0.003, rgba=(0.0, 0.0, 1.0, 0.8))


def add_visual_arrow(scene, from_point, to_point, radius=0.001, rgba=(0, 0, 1, 1)):
    """Adds a single visual arrow to the mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
                        pos=np.zeros(3), mat=np.eye(3).flatten(),
                        rgba=np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 
                         radius, from_point, to_point)
    scene.ngeom += 1


# Global trajectory history for velocity arrows (max 100 entries)
_velocity_trajectory = []
_MAX_TRAJECTORY_ARROWS = 100


def update_velocity_visualization(scene, velocity_world, angular_velocity_world, ee_pos):
    """
    Visualize the world-frame velocity and angular velocity commands with trajectory history.
    Maintains a history of the last MAX_TRAJECTORY_ARROWS velocity commands.
    
    Args:
        scene: mjvScene to add arrows to
        velocity_world: [vx, vy, vz] velocity in world frame (m/s)
        angular_velocity_world: [wx, wy, wz] angular velocity in world frame (rad/s)
        ee_pos: End-effector position in world frame
    """
    global _velocity_trajectory
    
    # Linear velocity arrow: shows commanded linear velocity in world frame
    vel_magnitude = np.linalg.norm(velocity_world)
    if vel_magnitude > 0.001:
        # Velocity is in world frame - no transformation needed
        vel_world = velocity_world
        
        # Scale arrow length based on magnitude
        arrow_scale = 0.3  # Visual scaling factor (meters per m/s)
        arrow_length = vel_magnitude * arrow_scale
        
        # Arrow direction in world frame (normalized)
        arrow_dir = vel_world / vel_magnitude
        
        # Arrow start and end points
        arrow_start = ee_pos.copy()
        arrow_end = arrow_start + arrow_dir * arrow_length
        
        # Arrow radius scales with magnitude
        arrow_radius = 0.003 + vel_magnitude * 0.01
        
        # Add current arrow to trajectory history
        _velocity_trajectory.append({
            'start': arrow_start.copy(),
            'end': arrow_end.copy(),
            'radius': arrow_radius,
            'magnitude': vel_magnitude
        })
        
        # Limit trajectory history to MAX_TRAJECTORY_ARROWS
        if len(_velocity_trajectory) > _MAX_TRAJECTORY_ARROWS:
            _velocity_trajectory.pop(0)  # Remove oldest arrow
    
    # Draw all linear velocity arrows in trajectory history
    # Fade older arrows (reduce alpha based on age)
    for i, arrow_data in enumerate(_velocity_trajectory):
        # Calculate fade: newer arrows are brighter, older ones fade
        age_ratio = i / max(len(_velocity_trajectory), 1)
        alpha = 0.3 + 0.5 * (1.0 - age_ratio)  # Fade from 0.8 to 0.3
        
        add_visual_arrow(
            scene,
            arrow_data['start'],
            arrow_data['end'],
            radius=arrow_data['radius'],
            rgba=(0.0, 1.0, 1.0, alpha)  # Cyan with fading alpha
        )
    
    # Angular velocity arrow: shows commanded angular velocity in world frame
    # This arrow should ALWAYS point in the world frame direction of rotation
    # e.g., if rotating about +z world, arrow points UP regardless of EE pose
    omega_magnitude = np.linalg.norm(angular_velocity_world)
    if omega_magnitude > 0.01:
        # Angular velocity is in world frame - no transformation needed
        omega_world = angular_velocity_world
        
        # Scale arrow length based on magnitude
        arrow_scale = 0.15  # Visual scaling factor (meters per rad/s)
        arrow_length = omega_magnitude * arrow_scale
        
        # Arrow direction in world frame (normalized)
        # This is the axis of rotation in world coordinates
        arrow_dir = omega_world / omega_magnitude
        
        # Arrow start and end points (centered at end-effector)
        arrow_start = ee_pos.copy()
        arrow_end = arrow_start + arrow_dir * arrow_length
        
        # Arrow radius scales with magnitude
        arrow_radius = 0.004 + omega_magnitude * 0.02
        
        # Add angular velocity arrow (magenta/purple color)
        # Arrow points in the direction of the rotation axis in world frame
        add_visual_arrow(
            scene,
            arrow_start,
            arrow_end,
            radius=arrow_radius,
            rgba=(1.0, 0.0, 1.0, 0.8)  # Magenta for angular velocity
        )


def main():
    # Load model & data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Create a Mink configuration
    configuration = mink.Configuration(model)

    # Define tasks - position and orientation control
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.1,  # Enable orientation control
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    # Create queue for spacemouse data
    data_queue = mp.Queue(maxsize=5)
    
    # Start spacemouse reader process (translation + rotation for end-effector)
    spacemouse_proc = mp.Process(
        target=spacemouse_process,
        args=(data_queue, VELOCITY_SCALE, ANGULAR_VELOCITY_SCALE)  # Translation and rotation scales
    )
    spacemouse_proc.start()
    print("SpaceMouse process started (Position + Orientation Control)")
    print("Translation: Pushing forward = world +X, right = world +Y, up = world +Z")
    print("Rotation: Roll/Pitch/Yaw = world frame angular velocity [wx, wy, wz]")
    print("  - Z-twist (yaw) should make purple arrow point UP (+Z world)")
    print("  - Purple arrow shows angular velocity direction in world frame")

    # Initialize viewer in passive mode
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset simulation data to the 'home' keyframe
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Get site ID
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        
        # Initialize target position and orientation from current site
        current_site_pos = data.site(site_id).xpos.copy()
        current_site_xmat = data.site(site_id).xmat.reshape(3, 3)
        current_site_rot = R.from_matrix(current_site_xmat)
        current_site_quat = current_site_rot.as_quat()  # [x, y, z, w]
        current_site_quat_wxyz = np.array([
            current_site_quat[3], 
            current_site_quat[0], 
            current_site_quat[1], 
            current_site_quat[2]
        ])  # [w, x, y, z]
        
        # Initialize the end-effector pose controller
        pose_controller = EndEffectorPoseController(
            initial_position=current_site_pos,
            initial_orientation_quat_wxyz=current_site_quat_wxyz
        )
        
        print(f"Initial site position (world frame): {current_site_pos}")
        print(f"World frame: X=Red, Y=Green, Z=Blue (see axes at origin)")

        # World frame reference position
        world_frame_pos = np.array([0.0, 0.0, 0.1])
        world_frame_length = 1.0

        rate = RateLimiter(frequency=200.0, warn=False)
        
        # Initialize velocity commands (will be updated from SpaceMouse)
        current_velocity_world = np.zeros(3)
        current_angular_velocity_world = np.zeros(3)  # [wx, wy, wz] in world frame
        
        # Gripper control
        # gripper_l has a position actuator, gripper_r is coupled via equality constraint
        gripper_l_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_l")
        
        gripper_open_pos = -0.026
        gripper_closed_pos = 0.0
        gripper_target = gripper_open_pos  # Start open
        gripper_prev_button = False  # Track previous button state for edge detection
        
        print(f"Gripper actuator: gripper_l (actuator_id={gripper_l_actuator_id})")
        print(f"Gripper control: Press SpaceMouse button to toggle open/close")

        while viewer.is_running():
            dt = rate.dt
            
            # Read spacemouse commands from queue
            # Reset velocities to zero if no new command (prevents drift)
            current_velocity_world = np.zeros(3)
            current_angular_velocity_world = np.zeros(3)
            
            if not data_queue.empty():
                try:
                    twist_command = data_queue.get_nowait()
                    # Extract translation as velocity in world frame
                    vel_raw = twist_command['translation']
                    
                    # Apply deadzone to prevent drift from noise
                    vel_magnitude = np.linalg.norm(vel_raw)
                    if vel_magnitude > 0.001:  # Deadzone threshold
                        current_velocity_world = vel_raw
                    
                    # Extract rotation as angular velocity in world frame
                    # Based on spacemouse_reader.py: rotation_twist = [roll, yaw, pitch]
                    # SpaceMouse: roll = x-axis rotation, pitch = y-axis rotation, yaw = z-axis rotation
                    # Map to world frame angular velocity: [wx, wy, wz] = [pitch, -roll, yaw] (x and y flipped)
                    rotation_twist = twist_command.get('rotation', np.zeros(3))
                    # rotation_twist is [roll, yaw, pitch] from spacemouse_reader
                    # Map to world frame angular velocity: [wx, wy, wz] = [pitch, -roll, yaw] (x and y flipped)
                    omega_raw = np.array([rotation_twist[2], -rotation_twist[0], rotation_twist[1]])  # [pitch, -roll, yaw] = [wx, wy, wz]
                    
                    # Negate to match CAD convention (opposite of SolidWorks behavior)
                    omega_raw = -omega_raw
                    
                    # Apply deadzone for angular velocity
                    omega_magnitude = np.linalg.norm(omega_raw)
                    if omega_magnitude > 0.01:  # Deadzone threshold
                        current_angular_velocity_world = omega_raw
                        
                        # Debug: print angular velocity when significant
                        # When you twist +z on SpaceMouse, wz should be positive, arrow should point UP
                        if abs(omega_raw[2]) > 0.05:  # z-component (wz)
                            print(f"DEBUG: Angular velocity (world frame): {current_angular_velocity_world}")
                            print(f"  wz={omega_raw[2]:.3f} (positive = rotate about +z world = arrow points UP)")
                            print(f"  rotation_twist was: {rotation_twist}")
                    
                    # Handle gripper button toggle
                    button_state = twist_command.get('button', [])
                    # button_state is a list where 1 = pressed, 0 = released
                    # Check if any button is pressed (typically button[0] is the main button)
                    # For SpaceMouse, side buttons are usually at index 0 or 1
                    button_pressed = len(button_state) > 0 and (button_state[0] == 1 if len(button_state) > 0 else False)
                    
                    # Also check other button indices in case the side button is at a different index
                    if not button_pressed and len(button_state) > 1:
                        button_pressed = button_state[1] == 1
                    
                    # Edge detection: toggle gripper on button press (not while held)
                    if button_pressed and not gripper_prev_button:
                        # Toggle gripper state
                        if gripper_target == gripper_open_pos:
                            gripper_target = gripper_closed_pos
                            print(f"Gripper: CLOSING (button_state={button_state})")
                        else:
                            gripper_target = gripper_open_pos
                            print(f"Gripper: OPENING (button_state={button_state})")
                    
                    gripper_prev_button = button_pressed
                    
                    # Debug: print button state occasionally
                    if button_pressed:
                        print(f"DEBUG: Button pressed! button_state={button_state}")
                    
                    # DEBUG: Print to verify world frame
                    if vel_magnitude > 0.01:
                        print(f"Velocity command (world frame): {current_velocity_world}")
                except queue.Empty:
                    pass

            # Update pose controller with velocity commands
            # This is the key interface: 6-vector input -> mink-compatible target
            pose_controller.update_from_velocity_command(
                velocity_world=current_velocity_world,
                angular_velocity_world=current_angular_velocity_world,
                dt=dt
            )
            
            # Get target pose from controller and set for IK
            target_pose = pose_controller.get_target_pose_se3()
            end_effector_task.set_target(target_pose)

            # Attempt to converge IK for position control
            converge_ik(
                configuration,
                tasks,
                dt,
                SOLVER,
                POS_THRESHOLD,
                ORI_THRESHOLD,
                MAX_ITERS,
            )

            # Set robot controls from IK (all 5 joints)
            data.ctrl[:5] = configuration.q[:5]
            
            # Set gripper control via data.ctrl
            # gripper_l: -0.026 (open) to 0 (closed)
            # gripper_r is automatically coupled via equality constraint (gripper_r = -gripper_l)
            if gripper_l_actuator_id >= 0 and gripper_l_actuator_id < model.nu:
                data.ctrl[gripper_l_actuator_id] = gripper_target

            # Step simulation
            mujoco.mj_step(model, data)

            # Get current site position for visualization
            current_site_pos = data.site(site_id).xpos
            
            # Update visualization
            scene = viewer.user_scn
            # Clear previous frame's geoms (start fresh each frame)
            scene.ngeom = 0
            
            # Add world frame reference axes (always visible at origin)
            add_world_frame_axes(scene, world_frame_pos, world_frame_length)
            # Add velocity and angular velocity command visualization (with trajectory history)
            update_velocity_visualization(
                scene,
                current_velocity_world,
                current_angular_velocity_world,
                current_site_pos
            )

            # Visualize at fixed FPS
            viewer.sync()
            rate.sleep()
    
    # Cleanup
    spacemouse_proc.terminate()
    spacemouse_proc.join()
    print("SpaceMouse process stopped")


if __name__ == "__main__":
    main()
