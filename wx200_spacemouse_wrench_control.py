"""
WX200 robot teleoperation using SpaceMouse with wrench (force/torque) control.
Applies world-frame forces and torques at the end-effector and lets physics drive motion.
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

# SpaceMouse scaling for wrench control
FORCE_SCALE = 10.0   # Newtons per unit SpaceMouse input
TORQUE_SCALE = 2.0   # N⋅m per unit SpaceMouse input

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

def update_wrench_visualization(scene, force, torque, ee_pos, ee_quat):
    """
    Add arrow visualizations for wrench commands (force and torque).
    Arrows are attached to the end-effector.
    Arrow size (length and thickness) indicates magnitude of the wrench.
    Wrenches are in world frame.
    
    Args:
        scene: mjvScene to add arrows to (viewer.user_scn)
        force: [fx, fy, fz] in world frame (N)
        torque: [tx, ty, tz] in world frame (N⋅m)
        ee_pos: End-effector position in world frame
        ee_quat: End-effector quaternion in world frame [w, x, y, z]
    """
    # Clear previous arrows
    scene.ngeom = 0
    
    # Force arrow: show in world frame, starting from EE
    force_magnitude = np.linalg.norm(force)
    if force_magnitude > 0.1:  # Threshold in Newtons
        # Force is already in world frame
        force_world = force
        
        # Scale arrow length based on magnitude (show as force vector)
        arrow_scale = 0.01  # Visual scaling factor (meters per Newton)
        arrow_length = force_magnitude * arrow_scale
        
        # Arrow direction in world frame (normalized)
        arrow_dir = force_world / force_magnitude
        
        # Arrow start and end points
        arrow_start = ee_pos.copy()
        arrow_end = arrow_start + arrow_dir * arrow_length
        
        # Arrow radius scales with magnitude (thicker for larger forces)
        arrow_radius = 0.002 + force_magnitude * 0.0005  # Min 0.002, scales up
        
        # Add force arrow (blue/cyan)
        add_visual_arrow(
            scene,
            arrow_start,
            arrow_end,
            radius=arrow_radius,
            rgba=(0.0, 0.5, 1.0, 0.9)  # Bright blue/cyan for force
        )
    
    # Torque arrow: show as axis of rotation
    torque_magnitude = np.linalg.norm(torque)
    if torque_magnitude > 0.01:  # Threshold in N⋅m
        # Torque is already in world frame
        torque_world = torque
        
        # Scale arrow length based on magnitude
        arrow_scale = 0.05  # Visual scaling factor
        arrow_length = torque_magnitude * arrow_scale
        
        # Arrow direction (torque axis, normalized)
        arrow_dir = torque_world / torque_magnitude
        
        # Position arrow at end-effector (offset slightly to avoid overlap)
        arrow_start = ee_pos.copy() + np.array([0, 0, 0.05])
        arrow_end = arrow_start + arrow_dir * arrow_length
        
        # Arrow radius scales with magnitude
        arrow_radius = 0.0015 + torque_magnitude * 0.002  # Min 0.0015, scales up
        
        # Add torque arrow (red/orange)
        add_visual_arrow(
            scene,
            arrow_start,
            arrow_end,
            radius=arrow_radius,
            rgba=(1.0, 0.3, 0.0, 0.9)  # Bright red/orange for torque
        )

def apply_wrench_to_end_effector(model, data, force, torque, ee_body_id):
    """
    Apply a wrench (force and torque) to the end-effector body in world frame.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        force: [fx, fy, fz] force vector in world frame (N)
        torque: [tx, ty, tz] torque vector in world frame (N⋅m)
        ee_body_id: Body ID of the end-effector
    """
    # Get the end-effector body's xfrc_applied index
    # xfrc_applied is a 6D wrench: [fx, fy, fz, tx, ty, tz] in body frame
    
    # We need to transform the world-frame wrench to body frame
    # Get body's rotation matrix
    body_quat = data.xquat[ee_body_id]
    body_rot = R.from_quat(body_quat).as_matrix()
    
    # Transform force and torque from world frame to body frame
    force_body = body_rot.T @ force  # Transpose = inverse for rotation matrix
    torque_body = body_rot.T @ torque
    
    # Apply wrench in body frame
    data.xfrc_applied[ee_body_id, 0:3] = force_body
    data.xfrc_applied[ee_body_id, 3:6] = torque_body

def main():
    # Load model & data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Create queue for spacemouse data
    data_queue = mp.Queue(maxsize=5)
    
    # Start spacemouse reader process
    spacemouse_proc = mp.Process(
        target=spacemouse_process,
        args=(data_queue, FORCE_SCALE, TORQUE_SCALE)  # Use force/torque scales
    )
    spacemouse_proc.start()
    print("SpaceMouse process started (Wrench Control Mode)")

    # Get end-effector body ID
    # The end-effector is at link5 (attachment_site is a site, not a body)
    # We need to find the body that contains the attachment_site
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    site_body_id = model.site_bodyid[site_id]
    print(f"End-effector body ID: {site_body_id}, name: {model.body(site_body_id).name}")

    # Initialize viewer in passive mode
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Reset simulation data to the 'home' keyframe
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)

        # World frame reference position
        world_frame_pos = np.array([0.0, 0.0, 0.1])
        world_frame_length = 1.0

        rate = RateLimiter(frequency=200.0, warn=False)
        
        # Initialize wrench commands
        current_force = np.zeros(3)
        current_torque = np.zeros(3)

        while viewer.is_running():
            dt = rate.dt

            # Read spacemouse commands from queue
            if not data_queue.empty():
                try:
                    twist_command = data_queue.get_nowait()
                    # Interpret translation as force, rotation as torque
                    current_force = twist_command['translation']  # Now represents force (N)
                    current_torque = twist_command['rotation']    # Now represents torque (N⋅m)
                except queue.Empty:
                    pass

            # Apply wrench to end-effector
            apply_wrench_to_end_effector(
                model, data,
                current_force,
                current_torque,
                site_body_id
            )

            # Step simulation (physics will respond to the applied wrench)
            mujoco.mj_step(model, data)

            # Get end-effector pose for visualization
            ee_pos = data.site(site_id).xpos
            ee_xmat = data.site(site_id).xmat.reshape(3, 3)
            ee_rot = R.from_matrix(ee_xmat)
            ee_quat = ee_rot.as_quat()  # [x, y, z, w]
            ee_quat = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])  # [w, x, y, z]
            
            # Update visualization
            scene = viewer.user_scn
            # Add world frame reference axes
            add_world_frame_axes(scene, world_frame_pos, world_frame_length)
            # Add wrench visualization
            update_wrench_visualization(
                scene,
                current_force,
                current_torque,
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
