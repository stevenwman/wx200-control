"""
Robot shutdown sequence and motor reboot utilities.

Provides safe shutdown sequence and motor reboot functionality for WX200 robot.
"""
import time
from dynamixel_sdk import *
from robot_config import robot_config


def shutdown_sequence(robot_driver, velocity_limit=30):
    """
    Execute safe shutdown sequence using a fresh port handler.
    
    Args:
        robot_driver: RobotDriver instance
        velocity_limit: Speed limit for movements (0=Max, 30=Slow/Safe)
    """
    print("\n" + "="*60)
    print("!!! SHUTDOWN SEQUENCE INITIATED !!!")
    print("="*60)
    
    if not robot_driver.connected:
        print("Robot not connected, skipping shutdown sequence")
        return
    
    # Flush port before shutdown (transition from TxOnly to TxRx mode)
    # The control loop uses write4ByteTxOnly (fire-and-forget) which doesn't wait for responses.
    # This can leave the port in a "pending" state. We need to do a TxRx operation to clear
    # this state before the shutdown sequence (which uses TxRx) can work properly.
    if robot_driver.portHandler:
        try:
            flush_success = False
            # Try to flush by doing a read operation (TxRx)
            for attempt in range(3):
                try:
                    result, comm_result, error = robot_driver.packetHandler.read4ByteTxRx(
                        robot_driver.portHandler, 1, robot_config.addr_present_position
                    )
                    if comm_result == COMM_SUCCESS:
                        flush_success = True
                        break
                    time.sleep(0.3)
                except Exception:
                    time.sleep(0.3)
            
            # If flush failed, try closing and reopening the port
            if not flush_success:
                try:
                    robot_driver.portHandler.closePort()
                    time.sleep(0.6)
                    if robot_driver.portHandler.openPort():
                        if robot_driver.portHandler.setBaudRate(robot_driver.baudrate):
                            robot_driver.portHandler.clearPort()
                            time.sleep(0.4)
                            flush_success = True
                except Exception:
                    pass
            
            time.sleep(0.3)
        except Exception:
            pass
        
        # Close the old port handler (we'll create a fresh one for shutdown)
        try:
            robot_driver.portHandler.closePort()
            time.sleep(0.5)
        except:
            pass
    
    devicename = robot_driver.devicename
    baudrate = robot_driver.baudrate
    
    # Create fresh port handler for shutdown sequence
    # This is necessary because the control loop uses TxOnly (fire-and-forget) mode,
    # which can leave the port in an inconsistent state. A fresh port handler ensures
    # clean TxRx communication for the shutdown sequence.
    from dynamixel_sdk import PortHandler, PacketHandler
    portHandler = PortHandler(devicename)
    packetHandler = PacketHandler(robot_config.protocol_version)
    
    if not portHandler.openPort():
        print("ERROR: Failed to open port for shutdown", flush=True)
        return
    
    if not portHandler.setBaudRate(baudrate):
        print("ERROR: Failed to set baudrate", flush=True)
        portHandler.closePort()
        return
    
    time.sleep(0.3)
    
    def retry_write(write_func, max_retries=5, sleep_time=0.2, clear_on_retry=2, error_on_fail=False, error_prefix=""):
        """
        Retry a Dynamixel write operation with port clearing on mid-retry.
        
        Args:
            write_func: Function that returns (comm_result, error) tuple
            max_retries: Maximum number of retry attempts
            sleep_time: Time to sleep between retries
            clear_on_retry: Retry number to clear port (0-indexed, e.g., 2 = 3rd attempt)
            error_on_fail: If True, print error message on final failure
            error_prefix: Prefix for error messages (e.g., "[ID 1]")
        
        Returns:
            (comm_result, error) tuple, or (None, None) if all retries failed
        """
        for retry in range(max_retries):
            comm_result, error = write_func()
            if comm_result == COMM_SUCCESS:
                return comm_result, error
            
            if retry < max_retries - 1:
                time.sleep(sleep_time)
                if retry == clear_on_retry:
                    portHandler.clearPort()
                    time.sleep(0.1)
            elif error_on_fail:
                error_msg = packetHandler.getTxRxResult(comm_result)
                print(f"{error_prefix} Write Error: {error_msg}", flush=True)
        
        return None, None
    
    def move_to_pose(ids, positions, speed_limit):
        """Move motors to positions. Skips any motor where position is -1."""
        portHandler.clearPort()
        time.sleep(0.1)
        
        # Set profile velocity and enable torque for all motors
        for dxl_id in ids:
            retry_write(
                lambda: packetHandler.write4ByteTxRx(portHandler, dxl_id, robot_config.addr_profile_velocity, speed_limit)
            )
            retry_write(
                lambda: packetHandler.write1ByteTxRx(portHandler, dxl_id, robot_config.addr_torque_enable, 1)
            )
        
        # Move motors to target positions
        for dxl_id, goal_pos in zip(ids, positions):
            if goal_pos == -1:
                continue
            
            comm_result, dxl_error = retry_write(
                lambda: packetHandler.write4ByteTxRx(portHandler, dxl_id, robot_config.addr_goal_position, goal_pos),
                sleep_time=0.25,
                error_on_fail=True,
                error_prefix=f"  [ID {dxl_id}]"
            )
            
            if comm_result == COMM_SUCCESS and dxl_error != 0:
                print(f"  [ID {dxl_id}] Packet Error: {packetHandler.getRxPacketError(dxl_error)}", flush=True)
    
    # Ensure gripper is open during shutdown (motor 7 = last motor)
    reasonable_pose = list(robot_config.reasonable_home_pose)
    base_pose = list(robot_config.base_home_pose)
    folded_pose = list(robot_config.folded_home_pose)
    
    # Set gripper (motor 7, index 6) to open position
    if len(reasonable_pose) > 6:
        reasonable_pose[6] = robot_config.gripper_encoder_max  # Open
    if len(base_pose) > 6:
        base_pose[6] = robot_config.gripper_encoder_max  # Open
    if len(folded_pose) > 6:
        folded_pose[6] = robot_config.gripper_encoder_max  # Open
    
    print("\nStep 1: Reasonable Home")
    move_to_pose(robot_config.motor_ids, reasonable_pose, velocity_limit)
    time.sleep(robot_config.move_delay)
    
    print("\nStep 2: Aligning Base")
    move_to_pose(robot_config.motor_ids, base_pose, velocity_limit)
    time.sleep(robot_config.move_delay)
    
    print("\nStep 3: Folding to Rest")
    move_to_pose(robot_config.motor_ids, folded_pose, velocity_limit)
    time.sleep(robot_config.move_delay)
    
    print("\nStep 4: Disabling Torque")
    time.sleep(robot_config.move_delay)
    # Disable torque using the fresh port handler (robot_driver's portHandler was closed)
    print("Disabling torque on all motors...")
    for dxl_id in robot_config.motor_ids:
        try:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, robot_config.addr_torque_enable, 0)
        except Exception:
            pass
    print("Torque disabled")
    
    print("\nShutdown Complete. Robot is limp.")
    
    portHandler.closePort()


def reboot_motors(robot_driver):
    """
    Reboot all motors after shutdown to clear any errors.
    
    Args:
        robot_driver: RobotDriver instance
    """
    print("\nRebooting motors after shutdown to clear any errors...")
    try:
        if not robot_driver.connected:
            return
        
        devicename = robot_driver.devicename
        baudrate = robot_driver.baudrate
        
        if robot_driver.portHandler:
            try:
                robot_driver.portHandler.closePort()
                time.sleep(0.5)
            except:
                pass
        
        from dynamixel_sdk import PortHandler, PacketHandler
        portHandler = PortHandler(devicename)
        packetHandler = PacketHandler(robot_config.protocol_version)
        
        if portHandler.openPort():
            if portHandler.setBaudRate(baudrate):
                time.sleep(0.3)
                
                rebooted_count = 0
                for dxl_id in robot_config.motor_ids:
                    try:
                        dxl_comm_result, dxl_error = packetHandler.reboot(portHandler, dxl_id)
                        if dxl_comm_result == COMM_SUCCESS:
                            rebooted_count += 1
                    except Exception:
                        pass
                
                print(f"Rebooted {rebooted_count}/{len(robot_config.motor_ids)} motors")
                time.sleep(0.5)
                portHandler.closePort()
    except Exception as e:
        print(f"Error rebooting motors: {e}")
        import traceback
        traceback.print_exc()
