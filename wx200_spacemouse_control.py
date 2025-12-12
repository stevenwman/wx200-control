"""
WX200 robot control with SpaceMouse (clean refactored version).

This version demonstrates the clean architecture:
- SpaceMouseDriver: Handles all SpaceMouse I/O
- RobotController: Orchestrates pose control → IK → joint commands
- Main loop: Just connects everything together

This architecture makes it easy to:
1. Swap SpaceMouse with other input sources (NN, joystick, etc.)
2. Use the same controller for simulation and real robot
"""
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
from spacemouse_driver import SpaceMouseDriver
from robot_controller import RobotController

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

# SpaceMouse scaling
VELOCITY_SCALE = 0.5  # m/s per unit SpaceMouse input
ANGULAR_VELOCITY_SCALE = 0.5  # rad/s per unit SpaceMouse rotation input

# Gripper positions
GRIPPER_OPEN_POS = -0.026
GRIPPER_CLOSED_POS = 0.0

# Global trajectory history for velocity arrows (max 100 entries)
_velocity_trajectory = []
_MAX_TRAJECTORY_ARROWS = 100


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
    
    # Create Mink configuration
    configuration = mink.Configuration(model)
    
    # Initialize viewer
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # Reset simulation to home
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        
        # Get initial end-effector pose
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        initial_position = data.site(site_id).xpos.copy()
        current_site_xmat = data.site(site_id).xmat.reshape(3, 3)
        current_site_rot = R.from_matrix(current_site_xmat)
        current_site_quat = current_site_rot.as_quat()  # [x, y, z, w]
        initial_orientation_quat_wxyz = np.array([
            current_site_quat[3], 
            current_site_quat[0], 
            current_site_quat[1], 
            current_site_quat[2]
        ])  # [w, x, y, z]
        
        print(f"Initial end-effector position: {initial_position}")
        print(f"World frame: X=Red, Y=Green, Z=Blue (see axes at origin)")
        
        # Initialize SpaceMouse driver
        spacemouse = SpaceMouseDriver(
            velocity_scale=VELOCITY_SCALE,
            angular_velocity_scale=ANGULAR_VELOCITY_SCALE
        )
        spacemouse.start()
        
        # Initialize robot controller
        robot_controller = RobotController(
            model=model,
            initial_position=initial_position,
            initial_orientation_quat_wxyz=initial_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        # Initialize posture task target (required before using controller)
        robot_controller.initialize_posture_target(configuration)
        
        # Get gripper actuator ID
        gripper_l_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_l")
        print(f"Gripper actuator: gripper_l (actuator_id={gripper_l_actuator_id})")
        
        # Control loop setup
        rate = RateLimiter(frequency=200.0, warn=False)
        world_frame_pos = np.array([0.0, 0.0, 0.1])
        world_frame_length = 1.0
        
        try:
            while viewer.is_running():
                dt = rate.dt
                
                # Update SpaceMouse input
                spacemouse.update()
                
                # Get velocity commands from SpaceMouse
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                
                # Update robot controller with velocity commands
                robot_controller.update_from_velocity_command(
                    velocity_world=velocity_world,
                    angular_velocity_world=angular_velocity_world,
                    dt=dt,
                    configuration=configuration
                )
                
                # Get joint commands from controller
                joint_commands = robot_controller.get_joint_commands(configuration, num_joints=5)
                data.ctrl[:5] = joint_commands
                
                # Set gripper control
                gripper_target = spacemouse.get_gripper_target_position(
                    open_position=GRIPPER_OPEN_POS,
                    closed_position=GRIPPER_CLOSED_POS
                )
                if gripper_l_actuator_id >= 0 and gripper_l_actuator_id < model.nu:
                    data.ctrl[gripper_l_actuator_id] = gripper_target
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Update visualization
                current_site_pos = data.site(site_id).xpos
                scene = viewer.user_scn
                scene.ngeom = 0
                
                add_world_frame_axes(scene, world_frame_pos, world_frame_length)
                update_velocity_visualization(
                    scene,
                    velocity_world,
                    angular_velocity_world,
                    current_site_pos
                )
                
                viewer.sync()
                rate.sleep()
        
        finally:
            # Cleanup
            spacemouse.stop()
            print("Control loop stopped")


if __name__ == "__main__":
    main()
