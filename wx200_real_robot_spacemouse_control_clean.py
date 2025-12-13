"""
WX200 real robot control with SpaceMouse (clean version, no visualization).

This script provides:
- SpaceMouse control
- Real robot actuation
- Safe startup and shutdown sequences
- Optional performance profiling

Flow:
1. Startup: Move robot to sim keyframe home position
2. Main loop:
   - Read SpaceMouse → Update pose controller → Solve IK → Send to robot
3. Shutdown: Execute safe exit sequence

Usage:
    python wx200_real_robot_spacemouse_control_clean.py [--profile]
    
    --profile: Enable performance profiling to identify bottlenecks
"""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
import time
import argparse
from spacemouse_driver import SpaceMouseDriver
from robot_controller import RobotController
from robot_joint_to_motor import JointToMotorTranslator, encoder_to_joint_angle, encoder_to_gripper_position, ENCODER_CENTER, ENCODER_MAX
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

# Gripper force feedback
GRIPPER_MOTOR_ID = 7
GRIPPER_CURRENT_LIMIT_MA = 100.0  # Disable torque if current exceeds this (mA) - adjust based on actual readings
GRIPPER_OPEN_THRESHOLD = 0.002  # Consider gripper "open" if within 2mm of open position (5-10% of range)

# Shutdown sequence poses (from exit_sequence.py)
REASONABLE_HOME_POSE = [-1, 1382, 2712, 1568, 1549, 2058, 1784]  # -1 means skip
BASE_HOME_POSE = [2040, -1, -1, -1, -1, -1, -1]
FOLDED_HOME_POSE = [2040, 846, 3249, 958, 1944, 2057, 1784]
MOVE_DELAY = 2.0  # Seconds to wait between shutdown moves

MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]
ADDR_PRESENT_POSITION = 132
ADDR_TORQUE_ENABLE = 64


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
    """Execute safe shutdown sequence using a fresh port handler."""
    print("\n" + "="*60)
    print("!!! SHUTDOWN SEQUENCE INITIATED !!!")
    print("="*60)
    
    if not robot_driver.connected:
        print("Robot not connected, skipping shutdown sequence")
        return
    
    devicename = robot_driver.devicename
    baudrate = getattr(robot_driver, 'baudrate', 1000000)
    
    if robot_driver.portHandler:
        try:
            robot_driver.portHandler.closePort()
            time.sleep(0.5)
        except:
            pass
    
    from dynamixel_sdk import PortHandler, PacketHandler
    PROTOCOL_VERSION = 2.0
    portHandler = PortHandler(devicename)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    
    if not portHandler.openPort():
        print("ERROR: Failed to open port for shutdown", flush=True)
        return
    
    if not portHandler.setBaudRate(baudrate):
        print("ERROR: Failed to set baudrate", flush=True)
        portHandler.closePort()
        return
    
    time.sleep(0.3)
    
    ADDR_PROFILE_VELOCITY = 112
    ADDR_GOAL_POSITION = 116
    ADDR_TORQUE_ENABLE = 64
    
    def move_to_pose(ids, positions, speed_limit):
        """Move motors to positions. Skips any motor where position is -1."""
        portHandler.clearPort()
        time.sleep(0.1)
        
        for dxl_id in ids:
            for retry in range(5):
                comm_result, error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VELOCITY, speed_limit)
                if comm_result == COMM_SUCCESS:
                    break
                if retry < 4:
                    time.sleep(0.2)
                    if retry == 2:
                        portHandler.clearPort()
                        time.sleep(0.1)
            
            for retry in range(5):
                comm_result, error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
                if comm_result == COMM_SUCCESS:
                    break
                if retry < 4:
                    time.sleep(0.2)
                    if retry == 2:
                        portHandler.clearPort()
                        time.sleep(0.1)
        
        for dxl_id, goal_pos in zip(ids, positions):
            if goal_pos == -1:
                continue
            
            success = False
            for retry in range(5):
                dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                    portHandler, dxl_id, ADDR_GOAL_POSITION, goal_pos
                )
                if dxl_comm_result == COMM_SUCCESS:
                    success = True
                    break
                
                if retry < 4:
                    time.sleep(0.25)
                    if retry == 2:
                        portHandler.clearPort()
                        time.sleep(0.1)
                else:
                    error_msg = packetHandler.getTxRxResult(dxl_comm_result)
                    print(f"  [ID {dxl_id}] Write Error: {error_msg}", flush=True)
            
            if success and dxl_error != 0:
                print(f"  [ID {dxl_id}] Packet Error: {packetHandler.getRxPacketError(dxl_error)}", flush=True)
    
    print("\nStep 1: Reasonable Home")
    move_to_pose(MOTOR_IDS, REASONABLE_HOME_POSE, VELOCITY_LIMIT)
    time.sleep(MOVE_DELAY)
    
    print("\nStep 2: Aligning Base")
    move_to_pose(MOTOR_IDS, BASE_HOME_POSE, VELOCITY_LIMIT)
    time.sleep(MOVE_DELAY)
    
    print("\nStep 3: Folding to Rest")
    move_to_pose(MOTOR_IDS, FOLDED_HOME_POSE, VELOCITY_LIMIT)
    time.sleep(MOVE_DELAY)
    
    print("\nStep 4: Disabling Torque")
    robot_driver.disable_torque_all()
    
    print("\nShutdown Complete. Robot is limp.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='WX200 Real Robot Control with SpaceMouse')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling to identify performance bottlenecks')
    args = parser.parse_args()
    
    ENABLE_PROFILING = args.profile
    
    print("WX200 Real Robot Control with SpaceMouse")
    print("="*60)
    print("Features:")
    print("- SpaceMouse control")
    print("- Safe startup and shutdown sequences")
    if ENABLE_PROFILING:
        print("- Performance profiling: ENABLED")
    print("="*60)
    
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    
    home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(model)
    print(f"\nSim home pose - EE position: {home_position}")
    
    robot_driver = RobotDriver()
    
    try:
        print("\nConnecting to robot...")
        robot_driver.connect()
        
        translator = JointToMotorTranslator(
            joint1_motor2_offset=0,
            joint1_motor3_offset=0
        )
        
        home_joint_angles = home_qpos[:5]
        # Open gripper first during initialization (use open position)
        home_gripper_pos = GRIPPER_OPEN_POS
        
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
        
        robot_driver.move_to_home(home_motor_positions, velocity_limit=VELOCITY_LIMIT)
        
        print("\n" + "="*60)
        print("✓ Robot is now at home position (sim keyframe)")
        print("Ready for SpaceMouse control!")
        print("Press Ctrl+C to stop and execute shutdown sequence")
        print("="*60 + "\n")
        
        spacemouse = SpaceMouseDriver(
            velocity_scale=VELOCITY_SCALE,
            angular_velocity_scale=ANGULAR_VELOCITY_SCALE
        )
        spacemouse.start()
        
        robot_controller = RobotController(
            model=model,
            initial_position=home_position,
            initial_orientation_quat_wxyz=home_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        configuration.update(home_qpos)
        robot_controller.initialize_posture_target(configuration)
        
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        
        control_rate = RateLimiter(frequency=CONTROL_FREQUENCY, warn=False)
        
        if ENABLE_PROFILING:
            import collections
            profile_times = collections.defaultdict(list)
            profile_count = 0
            profile_interval = 100
        
        control_loop_active = True
        
        # Gripper force feedback state
        gripper_last_target = None
        gripper_torque_disabled = False  # Track if we disabled torque due to current limit
        gripper_current_read_interval = 0  # Counter to read current every N loops (reduce frequency)
        
        try:
            running = True
            while running and control_loop_active:
                loop_start = time.perf_counter()
                
                dt = control_rate.dt
                current_time = time.time()
                
                t0 = time.perf_counter()
                spacemouse.update()
                if ENABLE_PROFILING:
                    profile_times['spacemouse_update'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                if ENABLE_PROFILING:
                    profile_times['get_velocity_commands'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                robot_controller.update_from_velocity_command(
                    velocity_world=velocity_world,
                    angular_velocity_world=angular_velocity_world,
                    dt=dt,
                    configuration=configuration
                )
                if ENABLE_PROFILING:
                    profile_times['ik_solve'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                joint_commands_rad = configuration.q[:5]
                if ENABLE_PROFILING:
                    profile_times['get_joint_commands'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                gripper_target = spacemouse.get_gripper_target_position(
                    open_position=GRIPPER_OPEN_POS,
                    closed_position=GRIPPER_CLOSED_POS
                )
                
                # Force feedback: If current > threshold, disable torque. Re-enable when opening.
                gripper_current_read_interval += 1
                should_read_current = (gripper_current_read_interval % 5 == 0)  # Read every 5 loops (~100ms at 50Hz)
                
                # Check if opening (moving towards open position)
                is_opening = False
                if gripper_last_target is not None:
                    is_opening = gripper_target < gripper_last_target
                
                # If torque was disabled and we're opening, re-enable torque
                if gripper_torque_disabled and is_opening:
                    try:
                        robot_driver.packetHandler.write1ByteTxRx(
                            robot_driver.portHandler, GRIPPER_MOTOR_ID, ADDR_TORQUE_ENABLE, 1
                        )
                        gripper_torque_disabled = False
                        print(f"[GRIPPER] Re-enabling torque (opening detected)", flush=True)
                    except Exception:
                        pass
                
                # Monitor current when torque is enabled and not near-open
                if not gripper_torque_disabled:
                    distance_from_open = abs(gripper_target - GRIPPER_OPEN_POS)
                    is_near_open = distance_from_open < GRIPPER_OPEN_THRESHOLD
                    
                    if not is_near_open and should_read_current:
                        # Read current from gripper motor
                        current_ma = robot_driver.read_present_current(GRIPPER_MOTOR_ID)
                        
                        if current_ma is not None:
                            try:
                                result, comm_result, error = robot_driver.packetHandler.read4ByteTxRx(
                                    robot_driver.portHandler, GRIPPER_MOTOR_ID, ADDR_PRESENT_POSITION
                                )
                                if comm_result == COMM_SUCCESS and error == 0:
                                    current_pos = result
                                    # Validate encoder value is in reasonable range
                                    if 1000 <= current_pos <= 4000:
                                        gripper_sim_pos = encoder_to_gripper_position(current_pos)
                                        gripper_percent_closed = ((gripper_sim_pos - GRIPPER_OPEN_POS) / 
                                                                 (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)) * 100
                                        
                                        # Print debug info
                                        print(f"[GRIPPER DEBUG] Current: {current_ma:6.1f}mA | "
                                              f"Encoder: {current_pos:4d} | "
                                              f"Sim Pos: {gripper_sim_pos:6.4f}m | "
                                              f"% Closed: {gripper_percent_closed:5.1f}% | "
                                              f"Target: {gripper_target:6.4f}m | "
                                              f"Limit: {GRIPPER_CURRENT_LIMIT_MA}mA", flush=True)
                                        
                                        # Simple rule: if current > limit, disable torque
                                        if current_ma > GRIPPER_CURRENT_LIMIT_MA:
                                            try:
                                                robot_driver.packetHandler.write1ByteTxRx(
                                                    robot_driver.portHandler, GRIPPER_MOTOR_ID, ADDR_TORQUE_ENABLE, 0
                                                )
                                                gripper_torque_disabled = True
                                                print(f"[GRIPPER] *** FORCE LIMIT EXCEEDED! Current: {current_ma:.1f}mA > Limit: {GRIPPER_CURRENT_LIMIT_MA}mA ***", flush=True)
                                                print(f"[GRIPPER] *** DISABLING TORQUE - Gripper will go limp. Press left button to re-enable. ***", flush=True)
                                            except Exception as e:
                                                print(f"[GRIPPER] Error disabling torque: {e}", flush=True)
                            except Exception:
                                pass
                
                gripper_last_target = gripper_target
                
                motor_positions = translator.joint_commands_to_motor_positions(
                    joint_angles_rad=joint_commands_rad,
                    gripper_position=gripper_target
                )
                
                # Remove gripper from motor commands if torque is disabled
                if gripper_torque_disabled:
                    motor_positions.pop(GRIPPER_MOTOR_ID, None)
                if ENABLE_PROFILING:
                    profile_times['joint_to_motor'].append(time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                robot_driver.send_motor_positions(motor_positions, velocity_limit=VELOCITY_LIMIT)
                if ENABLE_PROFILING:
                    profile_times['send_motor_commands'].append(time.perf_counter() - t0)
                
                if ENABLE_PROFILING:
                    profile_times['total_loop'].append(time.perf_counter() - loop_start)
                    profile_count += 1
                    
                    if profile_count >= profile_interval:
                        print("\n" + "="*60, flush=True)
                        print("PERFORMANCE PROFILE (last {} iterations):".format(profile_interval), flush=True)
                        print("="*60, flush=True)
                        for key in sorted(profile_times.keys()):
                            times = profile_times[key]
                            if times:
                                avg = sum(times) / len(times) * 1000
                                max_time = max(times) * 1000
                                min_time = min(times) * 1000
                                print(f"  {key:25s}: avg={avg:6.2f}ms, max={max_time:6.2f}ms, min={min_time:6.2f}ms", flush=True)
                        print("="*60 + "\n", flush=True)
                        profile_times.clear()
                        profile_count = 0
                
                control_rate.sleep()
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Stopping control loop...")
        
        finally:
            control_loop_active = False
            time.sleep(0.3)
            
            if robot_driver.connected and robot_driver.portHandler:
                try:
                    flush_success = False
                    for attempt in range(3):
                        try:
                            result, comm_result, error = robot_driver.packetHandler.read4ByteTxRx(
                                robot_driver.portHandler, 1, ADDR_PRESENT_POSITION
                            )
                            if comm_result == COMM_SUCCESS:
                                flush_success = True
                                break
                            time.sleep(0.3)
                        except Exception:
                            time.sleep(0.3)
                    
                    if not flush_success:
                        try:
                            robot_driver.portHandler.closePort()
                            time.sleep(0.6)
                            if robot_driver.portHandler.openPort():
                                baudrate = getattr(robot_driver, 'baudrate', 1000000)
                                if robot_driver.portHandler.setBaudRate(baudrate):
                                    robot_driver.portHandler.clearPort()
                                    time.sleep(0.4)
                                    flush_success = True
                        except Exception:
                            pass
                    
                    time.sleep(0.3)
                except Exception:
                    pass
            
            spacemouse.stop()
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected during initialization...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'control_loop_active' in locals():
            control_loop_active = False
        
        time.sleep(0.5)
        
        if robot_driver.connected and robot_driver.portHandler:
            try:
                flush_success = False
                for attempt in range(3):
                    try:
                        result, comm_result, error = robot_driver.packetHandler.read4ByteTxRx(
                            robot_driver.portHandler, 1, ADDR_PRESENT_POSITION
                        )
                        if comm_result == COMM_SUCCESS:
                            flush_success = True
                            break
                        time.sleep(0.3)
                    except:
                        time.sleep(0.3)
                
                if not flush_success:
                    try:
                        robot_driver.portHandler.closePort()
                        time.sleep(0.6)
                        if robot_driver.portHandler.openPort():
                            baudrate = getattr(robot_driver, 'baudrate', 1000000)
                            if robot_driver.portHandler.setBaudRate(baudrate):
                                robot_driver.portHandler.clearPort()
                                time.sleep(0.4)
                    except Exception:
                        pass
                
                time.sleep(0.3)
            except Exception:
                pass
        
        try:
            shutdown_sequence(robot_driver)
        except Exception as e:
            print(f"Error during shutdown sequence: {e}")
            import traceback
            traceback.print_exc()
        
        # Reboot motors after shutdown sequence to clear any errors
        # Use a fresh port handler (same approach as shutdown sequence)
        print("\nRebooting motors after shutdown to clear any errors...")
        try:
            if robot_driver.connected:
                devicename = robot_driver.devicename
                baudrate = getattr(robot_driver, 'baudrate', 1000000)
                
                # Close old port handler if it exists
                if robot_driver.portHandler:
                    try:
                        robot_driver.portHandler.closePort()
                        time.sleep(0.5)
                    except:
                        pass
                
                # Create fresh port handler for reboot
                from dynamixel_sdk import PortHandler, PacketHandler
                PROTOCOL_VERSION = 2.0
                portHandler = PortHandler(devicename)
                packetHandler = PacketHandler(PROTOCOL_VERSION)
                
                if portHandler.openPort():
                    if portHandler.setBaudRate(baudrate):
                        time.sleep(0.3)
                        
                        # Reboot all motors using fresh port handler
                        rebooted_count = 0
                        for dxl_id in MOTOR_IDS:
                            try:
                                dxl_comm_result, dxl_error = packetHandler.reboot(portHandler, dxl_id)
                                if dxl_comm_result == COMM_SUCCESS:
                                    rebooted_count += 1
                            except Exception:
                                pass
                        
                        print(f"Rebooted {rebooted_count}/{len(MOTOR_IDS)} motors")
                        time.sleep(0.5)
                        
                        portHandler.closePort()
                    else:
                        print("WARNING: Failed to set baudrate for reboot")
                else:
                    print("WARNING: Failed to open port for reboot")
        except Exception as e:
            print(f"Error rebooting motors: {e}")
            import traceback
            traceback.print_exc()
        
        robot_driver.disconnect()


if __name__ == "__main__":
    main()
