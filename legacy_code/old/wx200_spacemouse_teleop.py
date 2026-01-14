"""
WX200 robot teleoperation using SpaceMouse.
Integrates spacemouse twist commands with MuJoCo simulation.
"""
from pathlib import Path
import multiprocessing as mp
import queue
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter

import mink
from spacemouse_reader import spacemouse_process

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20

# SpaceMouse scaling
TRANSLATION_SCALE = 0.05  # m/s per unit input
ROTATION_SCALE = 0.5      # rad/s per unit input

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
    
    Args:
        scene: mjvScene to add axes to
        origin: np.ndarray [x, y, z] position of frame origin
        length: float length of each axis arrow
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
    """
    Adds a single visual arrow to the mjvScene.
    This is a visual-only object and does not affect the physics.

    Args:
        scene (mjvScene): The scene to add the arrow to.
        from_point (np.ndarray): The starting point of the arrow.
        to_point (np.ndarray): The ending point of the arrow.
        radius (float): The radius of the arrow's shaft.
        rgba (tuple): The color and alpha of the arrow.
    """
    if scene.ngeom >= scene.maxgeom:
        print("Warning: Maximum number of geoms reached. Cannot add arrow.")
        return
    
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
                        pos=np.zeros(3), mat=np.eye(3).flatten(), # Will be updated by mjv_connector
                        rgba=np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 
                         radius, from_point, to_point)
    scene.ngeom += 1

def update_arrow_visualization(scene, translation_twist, rotation_twist, ee_pos, ee_quat):
    """
    Add arrow visualizations for twist commands using mjvScene.
    Arrows are attached to the end-effector.
    Arrow size (length and thickness) indicates magnitude of the twist command.
    
    Args:
        scene: mjvScene to add arrows to (viewer.user_scn)
        translation_twist: [vx, vy, vz] in world frame (m/s)
        rotation_twist: [wx, wy, wz] in world frame (rad/s)
        ee_pos: End-effector position in world frame
        ee_quat: End-effector quaternion in world frame [w, x, y, z]
    """
    # Clear previous arrows
    scene.ngeom = 0
    
    # Translation arrow: show in world frame, starting from EE
    trans_magnitude = np.linalg.norm(translation_twist)
    if trans_magnitude > 0.001:
        # Translation is already in world frame, no transformation needed
        trans_world = translation_twist
        
        # Scale arrow length based on magnitude (show as velocity vector)
        arrow_scale = 0.3  # Base visual scaling factor (meters per m/s)
        arrow_length = trans_magnitude * arrow_scale
        
        # Arrow direction in world frame (normalized)
        arrow_dir = trans_world / trans_magnitude
        
        # Arrow start and end points
        arrow_start = ee_pos.copy()
        arrow_end = arrow_start + arrow_dir * arrow_length
        
        # Arrow radius scales with magnitude (thicker for larger commands)
        arrow_radius = 0.002 + trans_magnitude * 0.005  # Min 0.002, scales up
        
        # Add translation arrow (blue/cyan for delta xyz)
        add_visual_arrow(
            scene,
            arrow_start,
            arrow_end,
            radius=arrow_radius,
            rgba=(0.0, 0.5, 1.0, 0.9)  # Bright blue/cyan for translation
        )
    
    # Rotation arrow: show angular velocity as axis (delta omega)
    rot_magnitude = np.linalg.norm(rotation_twist)
    if rot_magnitude > 0.001:
        # Rotation is already in world frame, no transformation needed
        rot_world = rotation_twist
        
        # Scale arrow length based on magnitude
        arrow_scale = 0.2  # Base visual scaling factor
        arrow_length = rot_magnitude * arrow_scale
        
        # Arrow direction (rotation axis, normalized)
        arrow_dir = rot_world / rot_magnitude
        
        # Position arrow at end-effector (offset slightly to avoid overlap)
        arrow_start = ee_pos.copy() + np.array([0, 0, 0.05])
        arrow_end = arrow_start + arrow_dir * arrow_length
        
        # Arrow radius scales with magnitude
        arrow_radius = 0.0015 + rot_magnitude * 0.004  # Min 0.0015, scales up
        
        # Add rotation arrow (red/orange for delta omega)
        add_visual_arrow(
            scene,
            arrow_start,
            arrow_end,
            radius=arrow_radius,
            rgba=(1.0, 0.3, 0.0, 0.9)  # Bright red/orange for rotation
        )

def integrate_twist(current_pose, translation_twist, rotation_twist, dt):
    """
    Integrate twist command to get new target pose.
    Twist commands are interpreted in world frame.
    
    Args:
        current_pose: mink.SE3 current pose
        translation_twist: [vx, vy, vz] in world frame (m/s)
        rotation_twist: [wx, wy, wz] in world frame (rad/s)
        dt: time step
    
    Returns:
        new_pose: mink.SE3 new target pose
    """
    # Get current rotation matrix and translation
    R_current = current_pose.rotation().as_matrix()
    t_current = current_pose.translation()
    
    # Twists are already in world frame, no transformation needed
    v_world = translation_twist
    w_world = rotation_twist
    
    # Integrate translation
    new_translation = t_current + v_world * dt
    
    # Integrate rotation using Rodrigues' formula
    # w_world is [wx, wy, wz] - angular velocity vector in world frame
    # This represents rotation around the axis w_world/||w_world|| with magnitude ||w_world||
    theta = np.linalg.norm(w_world) * dt
    if theta > 1e-6:
        axis = w_world / np.linalg.norm(w_world)
        # Skew-symmetric matrix for axis-angle rotation
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        # Rodrigues' rotation formula
        R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        R_delta = np.eye(3)
    
    # Apply rotation delta to current rotation
    R_new = R_current @ R_delta
    
    # Convert rotation matrix to quaternion (wxyz format for mink)
    rot = R.from_matrix(R_new)
    quat_wxyz = rot.as_quat()  # Returns [x, y, z, w]
    quat_wxyz = np.array([quat_wxyz[3], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2]])  # Convert to [w, x, y, z]
    
    # Create new SE3 pose using array format [w, x, y, z, tx, ty, tz]
    new_pose_array = np.concatenate([quat_wxyz, new_translation])
    new_pose = mink.SE3(new_pose_array)
    
    return new_pose

def main():
    # Load model & data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Create a Mink configuration
    configuration = mink.Configuration(model)

    # Define tasks
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,  # Enable orientation control
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    # Create queue for spacemouse data
    data_queue = mp.Queue(maxsize=5)
    
    # Start spacemouse reader process
    spacemouse_proc = mp.Process(
        target=spacemouse_process,
        args=(data_queue, TRANSLATION_SCALE, ROTATION_SCALE)
    )
    spacemouse_proc.start()
    print("SpaceMouse process started")

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

        # Get initial target pose directly from end-effector (no mocap needed)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        ee_pos = data.site(site_id).xpos
        ee_xmat = data.site(site_id).xmat.reshape(3, 3)
        ee_rot = R.from_matrix(ee_xmat)
        ee_quat = ee_rot.as_quat()  # [x, y, z, w]
        ee_quat_wxyz = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])  # [w, x, y, z]
        T_wt = mink.SE3(np.concatenate([ee_quat_wxyz, ee_pos]))
        end_effector_task.set_target(T_wt)
        
        # Get site ID for accessing current EE pose
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # World frame reference position (origin or slightly offset)
        world_frame_pos = np.array([0.0, 0.0, 0.1])  # Slightly above ground
        world_frame_length = 1  # Length of axis arrows

        rate = RateLimiter(frequency=200.0, warn=False)
        
        # Initialize twist commands
        current_translation_twist = np.zeros(3)
        current_rotation_twist = np.zeros(3)

        while viewer.is_running():
            dt = rate.dt

            # Read spacemouse commands from queue
            if not data_queue.empty():
                try:
                    twist_command = data_queue.get_nowait()
                    current_translation_twist = twist_command['translation']
                    current_rotation_twist = twist_command['rotation']
                except queue.Empty:
                    pass

            # Integrate twist to update target pose
            T_wt = integrate_twist(
                T_wt,
                current_translation_twist,
                current_rotation_twist,
                dt
            )
            
            # Set target for IK (no need to update mocap if not using it)
            end_effector_task.set_target(T_wt)

            # Attempt to converge IK
            converge_ik(
                configuration,
                tasks,
                dt,
                SOLVER,
                POS_THRESHOLD,
                ORI_THRESHOLD,
                MAX_ITERS,
            )

            # Set robot controls
            data.ctrl[:5] = configuration.q[:5]
            data.ctrl[5] = 0

            # Step simulation
            mujoco.mj_step(model, data)

            # Get end-effector pose for arrow visualization
            ee_pos = data.site(site_id).xpos
            # Get quaternion from rotation matrix
            ee_xmat = data.site(site_id).xmat.reshape(3, 3)
            ee_quat = R.from_matrix(ee_xmat).as_quat()  # Returns [x, y, z, w]
            # Convert to [w, x, y, z] format
            ee_quat = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
            
            # Update arrow visualization in the scene
            # Access the scene through viewer and add arrows before sync
            scene = viewer.user_scn
            # Add world frame reference axes (always visible)
            add_world_frame_axes(scene, world_frame_pos, world_frame_length)
            # Add twist command arrows
            update_arrow_visualization(
                scene,
                current_translation_twist,
                current_rotation_twist,
                ee_pos, ee_quat
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
