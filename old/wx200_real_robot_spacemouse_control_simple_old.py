"""
WX200 real robot control with SpaceMouse.

Flow:
1. Startup: Move robot to sim keyframe home position
2. Main loop: Read SpaceMouse → Update pose controller → Solve IK → Send to robot
3. Shutdown: Execute safe exit sequence
"""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
import time
from spacemouse_driver import SpaceMouseDriver
from robot_controller import RobotController
from robot_joint_to_motor import JointToMotorTranslator
from robot_driver import RobotDriver
from dynamixel_sdk import *

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

# SpaceMouse scaling
VELOCITY_SCALE = 0.5
ANGULAR_VELOCITY_SCALE = 0.5

# Gripper positions (in meters, same as sim)
GRIPPER_OPEN_POS = -0.026
GRIPPER_CLOSED_POS = 0.0

# Robot control parameters
VELOCITY_LIMIT = 30
CONTROL_FREQUENCY = 50.0

# Gripper incremental control
GRIPPER_INCREMENT_RATE = 0.00025

# Shutdown sequence poses
REASONABLE_HOME_POSE = [-1, 1382, 2712, 1568, 1549, 2058, 1784]
BASE_HOME_POSE = [2040, -1, -1, -1, -1, -1, -1]
FOLDED_HOME_POSE = [2040, 846, 3249, 958, 1944, 2057, 1784]
MOVE_DELAY = 1.0

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
    current_site_quat = current_site_rot.as_quat()
    orientation_quat_wxyz = np.array([
        current_site_quat[3], 
        current_site_quat[0], 
        current_site_quat[1], 
        current_site_quat[2]
    ])
    
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
    print("WX200 Real Robot Control with SpaceMouse")
    print("="*60)
    print("Features:")
    print("- SpaceMouse control")
    print("- Safe startup and shutdown sequences")
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
        control_loop_active = True
        gripper_current_position = GRIPPER_OPEN_POS
        
        try:
            while control_loop_active:
                dt = control_rate.dt
                
                spacemouse.update()
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                
                robot_controller.update_from_velocity_command(
                    velocity_world=velocity_world,
                    angular_velocity_world=angular_velocity_world,
                    dt=dt,
                    configuration=configuration
                )
                
                joint_commands_rad = configuration.q[:5]
                
                # Incremental gripper control
                left_button_pressed, right_button_pressed = spacemouse.get_gripper_button_states()
                
                if left_button_pressed:
                    gripper_current_position -= GRIPPER_INCREMENT_RATE
                    gripper_current_position = max(gripper_current_position, GRIPPER_OPEN_POS)
                elif right_button_pressed:
                    gripper_current_position += GRIPPER_INCREMENT_RATE
                    gripper_current_position = min(gripper_current_position, GRIPPER_CLOSED_POS)
                
                gripper_target = gripper_current_position
                
                motor_positions = translator.joint_commands_to_motor_positions(
                    joint_angles_rad=joint_commands_rad,
                    gripper_position=gripper_target
                )
                
                robot_driver.send_motor_positions(motor_positions, velocity_limit=VELOCITY_LIMIT)
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
        
        # Reboot motors after shutdown
        print("\nRebooting motors after shutdown to clear any errors...")
        try:
            if robot_driver.connected:
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
                
                if portHandler.openPort():
                    if portHandler.setBaudRate(baudrate):
                        time.sleep(0.3)
                        
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
        except Exception as e:
            print(f"Error rebooting motors: {e}")
            import traceback
            traceback.print_exc()
        
        robot_driver.disconnect()


if __name__ == "__main__":
    main()
