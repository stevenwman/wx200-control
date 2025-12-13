"""
WX200 real robot control with SpaceMouse and live visualization.

This script combines:
- SpaceMouse control (from wx200_spacemouse_control.py)
- Real robot actuation (from wx200_robot_spacemouse_control.py)
- Live encoder feedback visualization (from robot_sim_live_sync.py)
- Startup sequence (move to home)
- Shutdown sequence (safe exit)

Flow:
1. Startup: Move robot to sim keyframe home position
2. Main loop:
   - Read SpaceMouse → Update pose controller → Solve IK → Send to robot
   - Read robot encoders → Update sim visualization (for feedback) [optional]
   - Visualize twist commands (arrows) [optional]
3. Shutdown: Execute safe exit sequence

Usage:
    python wx200_real_robot_spacemouse_control.py [--no-viz]
    
    --no-viz: Disable visualization for maximum performance
"""
from pathlib import Path
import mujoco
import mujoco.viewer  # Import at top level to avoid scoping issues
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
import time
import argparse
from spacemouse_driver import SpaceMouseDriver
from robot_controller import RobotController
from robot_joint_to_motor import JointToMotorTranslator, encoder_to_joint_angle, ENCODER_CENTER, ENCODER_MAX
from robot_driver import RobotDriver
from dynamixel_sdk import *

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

# SpaceMouse scaling
VELOCITY_SCALE = 0.5  # m/s per unit SpaceMouse input
ANGULAR_VELOCITY_SCALE = 0.5  # rad/s per unit SpaceMouse rotation input

# Gripper positions (in meters, same as sim)
GRIPPER_OPEN_POS = -0.026
GRIPPER_CLOSED_POS = 0.0

# Robot control parameters
VELOCITY_LIMIT = 30  # Speed limit for movements (0=Max, 30=Slow/Safe)
CONTROL_FREQUENCY = 50.0  # Control loop frequency (Hz)
VISUALIZATION_FREQUENCY = 20.0  # Visualization update frequency (Hz) - lower than control for performance

# Shutdown sequence poses (from exit_sequence.py)
REASONABLE_HOME_POSE = [-1, 1382, 2712, 1568, 1549, 2058, 1784]  # -1 means skip
BASE_HOME_POSE = [2040, -1, -1, -1, -1, -1, -1]
FOLDED_HOME_POSE = [2040, 846, 3249, 958, 1944, 2057, 1784]
MOVE_DELAY = 2.0  # Seconds to wait between shutdown moves

# Encoder reading
MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]
ADDR_PRESENT_POSITION = 132  # Used for reading encoders and port flushing

# Global trajectory history for velocity arrows
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
    """Add world frame coordinate axes (X, Y, Z) as colored arrows."""
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
    """
    global _velocity_trajectory
    
    # Linear velocity arrow
    vel_magnitude = np.linalg.norm(velocity_world)
    if vel_magnitude > 0.001:
        vel_world = velocity_world
        arrow_scale = 0.3
        arrow_length = vel_magnitude * arrow_scale
        arrow_dir = vel_world / vel_magnitude
        arrow_start = ee_pos.copy()
        arrow_end = arrow_start + arrow_dir * arrow_length
        arrow_radius = 0.003 + vel_magnitude * 0.01
        
        _velocity_trajectory.append({
            'start': arrow_start.copy(),
            'end': arrow_end.copy(),
            'radius': arrow_radius,
            'magnitude': vel_magnitude
        })
        
        if len(_velocity_trajectory) > _MAX_TRAJECTORY_ARROWS:
            _velocity_trajectory.pop(0)
    
    # Draw all linear velocity arrows in trajectory history
    for i, arrow_data in enumerate(_velocity_trajectory):
        age_ratio = i / max(len(_velocity_trajectory), 1)
        alpha = 0.3 + 0.5 * (1.0 - age_ratio)
        add_visual_arrow(
            scene,
            arrow_data['start'],
            arrow_data['end'],
            radius=arrow_data['radius'],
            rgba=(0.0, 1.0, 1.0, alpha)  # Cyan with fading alpha
        )
    
    # Angular velocity arrow
    omega_magnitude = np.linalg.norm(angular_velocity_world)
    if omega_magnitude > 0.01:
        omega_world = angular_velocity_world
        arrow_scale = 0.15
        arrow_length = omega_magnitude * arrow_scale
        arrow_dir = omega_world / omega_magnitude
        arrow_start = ee_pos.copy()
        arrow_end = arrow_start + arrow_dir * arrow_length
        arrow_radius = 0.004 + omega_magnitude * 0.02
        
        add_visual_arrow(
            scene,
            arrow_start,
            arrow_end,
            radius=arrow_radius,
            rgba=(1.0, 0.0, 1.0, 0.8)  # Magenta for angular velocity
        )


def read_robot_encoders(portHandler, packetHandler):
    """Read current encoder positions from all motors."""
    encoder_positions = {}
    for motor_id in MOTOR_IDS:
        try:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
                portHandler, motor_id, ADDR_PRESENT_POSITION
            )
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                encoder_positions[motor_id] = dxl_present_position
            else:
                encoder_positions[motor_id] = None
        except (IndexError, ValueError, Exception) as e:
            # Handle malformed responses or port errors gracefully
            encoder_positions[motor_id] = None
    return encoder_positions


def encoders_to_joint_angles(encoder_positions, translator):
    """
    Convert encoder positions to joint angles (inverse mapping).
    Same as in robot_sim_live_sync.py
    """
    joint_angles = np.zeros(6)  # 5 joints + gripper
    
    # Joint 0 (base-1_z) <- Motor 1 (FLIPPED)
    if 1 in encoder_positions and encoder_positions[1] is not None:
        joint_angles[0] = -encoder_to_joint_angle(encoder_positions[1])
    
    # Joint 1 (link1-2_x) <- Motors 2 and 3 (FLIPPED, opposing)
    if 2 in encoder_positions and 3 in encoder_positions:
        if encoder_positions[2] is not None and encoder_positions[3] is not None:
            motor2_enc_relative = encoder_positions[2] - translator.joint1_motor2_offset
            motor2_angle = encoder_to_joint_angle(motor2_enc_relative)
            
            motor3_enc_relative = encoder_positions[3] - translator.joint1_motor3_offset
            motor3_enc_flipped = 2 * ENCODER_CENTER - motor3_enc_relative
            motor3_angle = encoder_to_joint_angle(motor3_enc_flipped)
            
            joint_angles[1] = -(motor2_angle + motor3_angle) / 2.0
    
    # Joint 2 (link2-3_x) <- Motor 4
    if 4 in encoder_positions and encoder_positions[4] is not None:
        joint_angles[2] = encoder_to_joint_angle(encoder_positions[4])
    
    # Joint 3 (link3-4_x) <- Motor 5
    if 5 in encoder_positions and encoder_positions[5] is not None:
        joint_angles[3] = encoder_to_joint_angle(encoder_positions[5])
    
    # Joint 4 (link4-5_y) <- Motor 6 (FLIPPED)
    if 6 in encoder_positions and encoder_positions[6] is not None:
        joint_angles[4] = -encoder_to_joint_angle(encoder_positions[6])
    
    # Gripper <- Motor 7
    if 7 in encoder_positions and encoder_positions[7] is not None:
        GRIPPER_ENCODER_MIN = 1559  # Closed position
        GRIPPER_ENCODER_MAX = 2776  # Open position
        GRIPPER_ENCODER_RANGE = GRIPPER_ENCODER_MAX - GRIPPER_ENCODER_MIN  # 1217
        
        encoder = max(GRIPPER_ENCODER_MIN, min(GRIPPER_ENCODER_MAX, encoder_positions[7]))
        normalized = (encoder - GRIPPER_ENCODER_MIN) / GRIPPER_ENCODER_RANGE  # 0 to 1
        sim_gripper_range = 0.026
        joint_angles[5] = -sim_gripper_range * normalized  # 0.0 to -0.026 (closed to open)
    
    return joint_angles


def get_sim_home_pose(model):
    """Get the home pose from sim keyframe."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    
    qpos = data.qpos.copy()
    
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    position = data.site(site_id).xpos.copy()
    current_site_xmat = data.site(site_id).xmat.reshape(3, 3)
    current_site_rot = R.from_matrix(current_site_xmat)
    current_site_quat = current_site_rot.as_quat()  # [x, y, z, w]
    orientation_quat_wxyz = np.array([
        current_site_quat[3], 
        current_site_quat[0], 
        current_site_quat[1], 
        current_site_quat[2]
    ])  # [w, x, y, z]
    
    return qpos, position, orientation_quat_wxyz


def shutdown_sequence(robot_driver):
    """
    Execute safe shutdown sequence (from exit_sequence.py):
    1. Reasonable Home
    2. Base Home
    3. Folded Home
    4. Disable torque
    """
    print("\n" + "="*60)
    print("!!! SHUTDOWN SEQUENCE INITIATED !!!")
    print("="*60)
    
    if not robot_driver.connected:
        print("Robot not connected, skipping shutdown sequence")
        return
    
    portHandler = robot_driver.portHandler
    packetHandler = robot_driver.packetHandler
    
    # CRITICAL FIX: Flush port state by doing a TxRx operation
    # The control loop uses write4ByteTxOnly (fire-and-forget) which leaves port in "pending" state
    # We MUST do a TxRx operation to clear this state before shutdown uses TxRx
    print("Flushing port state (transitioning from TxOnly to TxRx mode)...")
    ADDR_PRESENT_POSITION = 132
    flush_success = False
    for attempt in range(5):
        try:
            result, comm_result, error = packetHandler.read4ByteTxRx(portHandler, 1, ADDR_PRESENT_POSITION)
            if comm_result == COMM_SUCCESS:
                print(f"Port flush successful (attempt {attempt+1})")
                flush_success = True
                break
            else:
                print(f"Port flush attempt {attempt+1} got error: {packetHandler.getTxRxResult(comm_result)}")
                time.sleep(0.2)
        except Exception as e:
            print(f"Port flush attempt {attempt+1} exception: {e}")
            time.sleep(0.2)
    
    if not flush_success:
        print("WARNING: Port flush failed - shutdown may have issues")
    
    time.sleep(0.2)  # Extra pause after flush
    
    ADDR_PROFILE_VELOCITY = 112
    ADDR_GOAL_POSITION = 116
    ADDR_TORQUE_ENABLE = 64
    
    def move_to_pose(ids, positions, speed_limit):
        """Move motors to positions. Skips any motor where position is -1."""
        # Port should already be flushed by shutdown_sequence
        # Don't call clearPort() here as it might interfere
        
        # Set speed limit and enable torque
        for dxl_id in ids:
            packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VELOCITY, speed_limit)
            packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        
        # Send move commands (ignoring -1s)
        print("  Sending position commands...")
        for dxl_id, goal_pos in zip(ids, positions):
            if goal_pos == -1:
                continue
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                portHandler, dxl_id, ADDR_GOAL_POSITION, goal_pos
            )
            if dxl_comm_result != COMM_SUCCESS:
                error_msg = packetHandler.getTxRxResult(dxl_comm_result)
                print(f"  [ID {dxl_id}] Write Error: {error_msg}")
                # If port is in use, wait and retry once (don't call clearPort as it might make it worse)
                if "Port is in use" in str(error_msg):
                    time.sleep(0.3)  # Wait longer for port to be ready
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                        portHandler, dxl_id, ADDR_GOAL_POSITION, goal_pos
                    )
                    if dxl_comm_result == COMM_SUCCESS:
                        print(f"  [ID {dxl_id}] Retry successful, moving to {goal_pos}")
            elif dxl_error != 0:
                print(f"  [ID {dxl_id}] Packet Error: {packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"  [ID {dxl_id}] Moving to {goal_pos}")
    
    # Step 1: Reasonable Home
    print("\nStep 1: Reasonable Home")
    move_to_pose(MOTOR_IDS, REASONABLE_HOME_POSE, VELOCITY_LIMIT)
    print(f"  Waiting {MOVE_DELAY} seconds for movement...")
    time.sleep(MOVE_DELAY)
    
    # Step 2: Base Home
    print("\nStep 2: Aligning Base")
    move_to_pose(MOTOR_IDS, BASE_HOME_POSE, VELOCITY_LIMIT)
    print(f"  Waiting {MOVE_DELAY} seconds for movement...")
    time.sleep(MOVE_DELAY)
    
    # Step 3: Folded Home
    print("\nStep 3: Folding to Rest")
    move_to_pose(MOTOR_IDS, FOLDED_HOME_POSE, VELOCITY_LIMIT)
    print(f"  Waiting {MOVE_DELAY} seconds for movement...")
    time.sleep(MOVE_DELAY)
    
    # Step 4: Disable torque
    print("\nStep 4: Disabling Torque")
    robot_driver.disable_torque_all()
    
    print("\nShutdown Complete. Robot is limp.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='WX200 Real Robot Control with SpaceMouse')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Disable visualization for maximum performance')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling to identify performance bottlenecks')
    args = parser.parse_args()
    
    ENABLE_VISUALIZATION = not args.no_viz
    ENABLE_PROFILING = args.profile
    
    print("WX200 Real Robot Control with SpaceMouse")
    print("="*60)
    print("Features:")
    print("- SpaceMouse control")
    if ENABLE_VISUALIZATION:
        print("- Live MuJoCo visualization (shows actual robot state)")
        print("- Twist command visualization (arrows)")
    else:
        print("- Visualization: DISABLED (maximum performance mode)")
    print("- Safe startup and shutdown sequences")
    print("="*60)
    
    # Load model for IK and visualization
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # Create Mink configuration
    configuration = mink.Configuration(model)
    
    # Get sim home pose
    home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(model)
    print(f"\nSim home pose - EE position: {home_position}")
    
    # Initialize robot driver
    robot_driver = RobotDriver()
    
    try:
        # Connect to robot
        print("\nConnecting to robot...")
        robot_driver.connect()
        
        # Initialize joint-to-motor translator
        translator = JointToMotorTranslator(
            joint1_motor2_offset=0,  # TODO: Calibrate if needed
            joint1_motor3_offset=0   # TODO: Calibrate if needed
        )
        
        # Convert sim home qpos to motor encoder positions
        home_joint_angles = home_qpos[:5]  # First 5 joints
        home_gripper_pos = home_qpos[5] if len(home_qpos) > 5 else -0.01
        
        home_motor_positions = translator.joint_commands_to_motor_positions(
            joint_angles_rad=home_joint_angles,
            gripper_position=home_gripper_pos
        )
        
        translator.set_home_encoders([
            home_motor_positions.get(1, 2048),
            home_motor_positions.get(2, 2048),
            home_motor_positions.get(3, 2048),
            home_motor_positions.get(4, 2048),
            home_motor_positions.get(5, 2048),
            home_motor_positions.get(6, 2048),
            home_motor_positions.get(7, 2048),
        ])
        
        print(f"Home motor positions: {home_motor_positions}")
        print("\n" + "="*60)
        print("STARTUP SEQUENCE: Moving robot to home position...")
        print("This may take a few seconds...")
        print("="*60)
        
        # Move robot to home position
        robot_driver.move_to_home(home_motor_positions, velocity_limit=VELOCITY_LIMIT)
        
        print("\n" + "="*60)
        print("✓ Robot is now at home position (sim keyframe)")
        print("Ready for SpaceMouse control!")
        if ENABLE_VISUALIZATION:
            print("Press ESC in viewer or Ctrl+C to stop and execute shutdown sequence")
        else:
            print("Press Ctrl+C to stop and execute shutdown sequence")
        print("="*60 + "\n")
        
        # Initialize SpaceMouse driver
        spacemouse = SpaceMouseDriver(
            velocity_scale=VELOCITY_SCALE,
            angular_velocity_scale=ANGULAR_VELOCITY_SCALE
        )
        spacemouse.start()
        
        # Initialize robot controller with home pose
        robot_controller = RobotController(
            model=model,
            initial_position=home_position,
            initial_orientation_quat_wxyz=home_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        # Initialize posture task target
        configuration.update(home_qpos)
        robot_controller.initialize_posture_target(configuration)
        
        # Initialize simulation to home
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        
        # Control loop (50 Hz)
        control_rate = RateLimiter(frequency=CONTROL_FREQUENCY, warn=False)
        
        if ENABLE_VISUALIZATION:
            # Get site ID for visualization
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            gripper_l_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_l")
            gripper_r_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_r")
            
            # Pre-allocate arrays to avoid allocations in hot loop
            _zero_array = np.zeros(3)
            _ee_pos = np.zeros(3)
            
            # Visualization rate limiting (20 Hz - lower frequency for performance)
            vis_interval = 1.0 / VISUALIZATION_FREQUENCY
            last_vis_time = time.time()
            
            # Initialize viewer with context manager
            viewer_context = mujoco.viewer.launch_passive(
                model=model, data=data, show_left_ui=False, show_right_ui=False
            )
            mujoco.mjv_defaultFreeCamera(model, viewer_context.cam)
        else:
            # No visualization - use dummy viewer context
            viewer_context = None
            site_id = None
            gripper_l_joint_id = None
            gripper_r_joint_id = None
            _zero_array = None
            _ee_pos = None
            vis_interval = None
            last_vis_time = None
        
        # Profiling setup
        if ENABLE_PROFILING:
            import collections
            profile_times = collections.defaultdict(list)
            profile_count = 0
            profile_interval = 100  # Print stats every N iterations
        
        # Flag to stop control loop before shutdown
        control_loop_active = True
        
        try:
            running = True
            while running and control_loop_active:
                loop_start = time.perf_counter()
                
                # Update running condition
                if ENABLE_VISUALIZATION:
                    running = viewer_context.is_running()
                
                dt = control_rate.dt
                current_time = time.time()
                
                # === CONTROL PATH (every loop, 50 Hz) ===
                # Update SpaceMouse input
                t0 = time.perf_counter()
                spacemouse.update()
                if ENABLE_PROFILING:
                    profile_times['spacemouse_update'].append(time.perf_counter() - t0)
                
                # Get velocity commands from SpaceMouse
                t0 = time.perf_counter()
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                if ENABLE_PROFILING:
                    profile_times['get_velocity_commands'].append(time.perf_counter() - t0)
                
                # Update robot controller with velocity commands
                t0 = time.perf_counter()
                robot_controller.update_from_velocity_command(
                    velocity_world=velocity_world,
                    angular_velocity_world=angular_velocity_world,
                    dt=dt,
                    configuration=configuration
                )
                if ENABLE_PROFILING:
                    profile_times['ik_solve'].append(time.perf_counter() - t0)
                
                # Get joint commands from controller (in radians)
                # Use view instead of copy if possible
                t0 = time.perf_counter()
                joint_commands_rad = configuration.q[:5]
                if ENABLE_PROFILING:
                    profile_times['get_joint_commands'].append(time.perf_counter() - t0)
                
                # Convert joint commands to motor positions
                t0 = time.perf_counter()
                gripper_target = spacemouse.get_gripper_target_position(
                    open_position=GRIPPER_OPEN_POS,
                    closed_position=GRIPPER_CLOSED_POS
                )
                
                motor_positions = translator.joint_commands_to_motor_positions(
                    joint_angles_rad=joint_commands_rad,
                    gripper_position=gripper_target
                )
                if ENABLE_PROFILING:
                    profile_times['joint_to_motor'].append(time.perf_counter() - t0)
                
                # Send motor commands to robot (CRITICAL PATH - must be fast)
                t0 = time.perf_counter()
                robot_driver.send_motor_positions(motor_positions, velocity_limit=VELOCITY_LIMIT)
                if ENABLE_PROFILING:
                    profile_times['send_motor_commands'].append(time.perf_counter() - t0)
                
                
                # === VISUALIZATION PATH (lower frequency, 20 Hz) ===
                # Only update visualization at lower frequency to reduce latency
                if ENABLE_VISUALIZATION and (current_time - last_vis_time) >= vis_interval:
                    # Only read encoders if control loop is still active and robot is connected
                    if not control_loop_active or not robot_driver.connected:
                        # Control loop stopped, skip visualization update
                        continue
                    
                    try:
                        # Read actual encoder positions from robot (for visualization feedback)
                        robot_encoders = read_robot_encoders(
                            robot_driver.portHandler,
                            robot_driver.packetHandler
                        )
                    except Exception as e:
                        # If encoder read fails, skip this visualization update
                        continue
                    
                    # Convert encoders to joint angles
                    robot_joint_angles = encoders_to_joint_angles(robot_encoders, translator)
                    
                    # Update simulation with actual robot joint angles (for visualization)
                    data.qpos[:5] = robot_joint_angles[:5]  # Update first 5 joints
                    
                    # Update gripper: sync gripper_l and gripper_r like in sim
                    gripper_l_pos = robot_joint_angles[5]
                    gripper_r_pos = -gripper_l_pos  # Sync: gripper_r = -gripper_l
                    
                    if gripper_l_joint_id >= 0:
                        data.qpos[gripper_l_joint_id] = gripper_l_pos
                    if gripper_r_joint_id >= 0:
                        data.qpos[gripper_r_joint_id] = gripper_r_pos
                    
                    # Forward kinematics
                    mujoco.mj_forward(model, data)
                    
                    # Get end-effector position for visualization (use view, avoid copy)
                    _ee_pos[:] = data.site(site_id).xpos
                    
                    # Clear previous visualizations
                    viewer_context.user_scn.ngeom = 0
                    
                    # Add world frame axes at origin
                    add_world_frame_axes(viewer_context.user_scn, _zero_array)
                    
                    # Visualize velocity commands
                    update_velocity_visualization(
                        viewer_context.user_scn,
                        velocity_world,
                        angular_velocity_world,
                        _ee_pos
                    )
                    
                    # Update viewer
                    viewer_context.sync()
                    
                    last_vis_time = current_time
                
                # Profiling
                if ENABLE_PROFILING:
                    profile_times['total_loop'].append(time.perf_counter() - loop_start)
                    profile_count += 1
                    
                    if profile_count >= profile_interval:
                        print("\n" + "="*60)
                        print("PERFORMANCE PROFILE (last {} iterations):".format(profile_interval))
                        print("="*60)
                        for key in sorted(profile_times.keys()):
                            times = profile_times[key]
                            if times:
                                avg = sum(times) / len(times) * 1000  # Convert to ms
                                max_time = max(times) * 1000
                                min_time = min(times) * 1000
                                print(f"  {key:25s}: avg={avg:6.2f}ms, max={max_time:6.2f}ms, min={min_time:6.2f}ms")
                        print("="*60 + "\n")
                        # Clear for next interval
                        profile_times.clear()
                        profile_count = 0
                
                control_rate.sleep()
            
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Stopping control loop...")
        
        finally:
            # Stop control loop first to release port
            control_loop_active = False
            print("Waiting for control loop to exit...")
            time.sleep(0.3)  # Wait for control loop to fully exit
            
            # Flush port by doing a TxRx operation to clear any pending TxOnly operations
            # This is critical: TxOnly leaves port in "pending" state, TxRx clears it
            if robot_driver.connected and robot_driver.portHandler:
                try:
                    print("Flushing port state (clearing pending TxOnly operations)...")
                    # Do a dummy read to flush the port - this waits for any pending operations
                    robot_driver.packetHandler.read4ByteTxRx(
                        robot_driver.portHandler, 1, ADDR_PRESENT_POSITION
                    )
                    time.sleep(0.1)
                    print("Port flushed successfully")
                except Exception as e:
                    print(f"Port flush failed (may be OK): {e}")
            
            # Cleanup
            spacemouse.stop()
            print("SpaceMouse stopped")
            if ENABLE_VISUALIZATION and viewer_context is not None:
                viewer_context.close()
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected during initialization...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always execute shutdown sequence
        # Make sure control loop has fully stopped first
        if 'control_loop_active' in locals():
            control_loop_active = False
        
        # Wait for control loop to fully exit
        print("Ensuring control loop has stopped...")
        time.sleep(0.3)
        
        # Port should already be flushed in the inner finally block
        # But do one more flush here just in case (with retries)
        if robot_driver.connected and robot_driver.portHandler:
            try:
                for attempt in range(3):
                    try:
                        result, comm_result, error = robot_driver.packetHandler.read4ByteTxRx(
                            robot_driver.portHandler, 1, ADDR_PRESENT_POSITION
                        )
                        if comm_result == COMM_SUCCESS:
                            break
                        time.sleep(0.2)
                    except:
                        time.sleep(0.2)
                time.sleep(0.1)
            except:
                pass
        
        try:
            shutdown_sequence(robot_driver)
        except Exception as e:
            print(f"Error during shutdown sequence: {e}")
            import traceback
            traceback.print_exc()
        
        # Disconnect
        robot_driver.disconnect()
        print("Robot driver disconnected")


if __name__ == "__main__":
    main()
