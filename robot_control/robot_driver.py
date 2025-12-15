"""
Dynamixel robot driver for sending joint commands to real hardware.
"""
import time
from dynamixel_sdk import *
from robot_control.robot_config import robot_config


class RobotDriver:
    """
    Driver for controlling Dynamixel motors on the real robot.
    """
    
    def __init__(self, devicename=None, baudrate=None):
        """
        Initialize robot driver.
        
        Args:
            devicename: Serial port device name (e.g., '/dev/ttyUSB0'). Defaults to robot_config.
            baudrate: Serial baudrate. Defaults to robot_config.
        """
        self.devicename = devicename if devicename is not None else robot_config.devicename
        self.baudrate = baudrate if baudrate is not None else robot_config.baudrate
        self.portHandler = None
        self.packetHandler = None
        self.connected = False
        self._last_velocity_limit = None  # Cache to avoid setting velocity every loop
        self.groupSyncWrite = None  # GroupSyncWrite for bulk position writes
    
    def reboot_all_motors(self):
        """
        Reboot all motors to clear error states (e.g., red flashing LED).
        This resets motors that have stalled or entered error states.
        """
        if not self.portHandler or not self.packetHandler:
            raise RuntimeError("Port not initialized. Call connect() first.")
        
        print("Rebooting all motors to clear error states...")
        rebooted_count = 0
        failed_motors = []
        
        for dxl_id in robot_config.motor_ids:
            try:
                dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, dxl_id)
                if dxl_comm_result == COMM_SUCCESS:
                    rebooted_count += 1
                else:
                    # Log which motor failed
                    error_msg = self.packetHandler.getTxRxResult(dxl_comm_result)
                    failed_motors.append((dxl_id, error_msg))
            except Exception as e:
                # Log exception for debugging
                failed_motors.append((dxl_id, str(e)))
        
        if rebooted_count > 0:
            print(f"Rebooted {rebooted_count}/{len(robot_config.motor_ids)} motors")
        else:
            print(f"WARNING: Failed to reboot any motors (0/{len(robot_config.motor_ids)})")
            if failed_motors:
                print("Failed motors:")
                for motor_id, error in failed_motors[:3]:  # Show first 3 errors
                    print(f"  Motor {motor_id}: {error}")
        
        time.sleep(0.5)  # Wait for motors to initialize after reboot
    
    def reboot_motor(self, motor_id):
        """
        Reboot a single motor to clear error states (e.g., over-torque on gripper).
        
        This is useful when a specific joint (like the gripper) has latched an error
        but you don't want to reset the entire robot.
        """
        if not self.portHandler or not self.packetHandler:
            raise RuntimeError("Port not initialized. Call connect() first.")
        
        if motor_id not in robot_config.motor_ids:
            print(f"Warning: Attempted to reboot unknown motor ID {motor_id}")
            return
        
        try:
            print(f"Rebooting motor {motor_id} to clear error state...")
            dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, motor_id)
            if dxl_comm_result != COMM_SUCCESS:
                error_msg = self.packetHandler.getTxRxResult(dxl_comm_result)
                print(f"Warning: Failed to reboot motor {motor_id}: {error_msg}")
                return
            
            # After reboot, torque is typically disabled; re-enable it for this motor.
            time.sleep(0.1)
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, motor_id, robot_config.addr_torque_enable, 1
            )
            if dxl_comm_result != COMM_SUCCESS:
                error_msg = self.packetHandler.getTxRxResult(dxl_comm_result)
                print(f"Warning: Failed to re-enable torque on motor {motor_id}: {error_msg}")
            elif dxl_error != 0:
                error_msg = self.packetHandler.getRxPacketError(dxl_error)
                print(f"Warning: Motor {motor_id} reported error after reboot: {error_msg}")
            else:
                print(f"Motor {motor_id} rebooted and torque re-enabled.")
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Exception while rebooting motor {motor_id}: {e}")
    
    def connect(self):
        """Connect to robot, reboot motors to clear errors, and enable torque on all motors."""
        self.portHandler = PortHandler(self.devicename)
        self.packetHandler = PacketHandler(robot_config.protocol_version)
        
        # Open port
        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open port {self.devicename}")
        
        # Set baudrate
        if not self.portHandler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate}")
        
        # Reboot motors to clear error states
        self.reboot_all_motors()
        
        # Enable torque on all motors
        for dxl_id in robot_config.motor_ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, robot_config.addr_torque_enable, 1
            )
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError(f"Failed to enable torque on motor {dxl_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise RuntimeError(f"Motor {dxl_id} error: {self.packetHandler.getRxPacketError(dxl_error)}")
        
        # Initialize GroupSyncWrite for bulk position writes
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler, 
            self.packetHandler, 
            robot_config.addr_goal_position, 
            4
        )
        
        self.connected = True
        print(f"Robot driver connected on {self.devicename}")
    
    def disconnect(self):
        """Disconnect from robot and disable torque."""
        if self.connected:
            try:
                self.disable_torque_all()
            except Exception:
                pass  # Port may already be closed
            
            if self.portHandler:
                try:
                    self.portHandler.closePort()
                except Exception:
                    pass
            
            self.connected = False
            print("Robot driver disconnected")
    
    def disable_torque_all(self):
        """Disable torque on all motors."""
        if not self.connected or not self.portHandler:
            return
        
        print("Disabling torque on all motors...")
        for dxl_id in robot_config.motor_ids:
            try:
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, robot_config.addr_torque_enable, 0
                )
            except Exception:
                pass
        print("Torque disabled")
    
    def set_profile_velocity(self, motor_id, velocity_limit, use_tx_only=False):
        """
        Set profile velocity (speed limit) for a motor.
        
        Args:
            motor_id: Motor ID
            velocity_limit: Speed limit (0=Max, higher=slower)
            use_tx_only: If True, use TxOnly (non-blocking) instead of TxRx (blocking)
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")
        
        if use_tx_only:
            self.packetHandler.write4ByteTxOnly(
                self.portHandler, motor_id, robot_config.addr_profile_velocity, velocity_limit
            )
        else:
            # Use TxRx with retry to ensure velocity is actually set
            for attempt in range(3):
                dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                    self.portHandler, motor_id, robot_config.addr_profile_velocity, velocity_limit
                )
                if dxl_comm_result == COMM_SUCCESS:
                    break
                if attempt < 2:
                    time.sleep(0.1)
                    self.portHandler.clearPort()
    
    def send_motor_positions(self, motor_positions, velocity_limit=30):
        """
        Send motor position commands using GroupSyncWrite for low latency.
        
        Args:
            motor_positions: dict {motor_id: encoder_position}
            velocity_limit: Speed limit for movement (0=Max, 30=Slow/Safe)
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")
        
        # Set velocity limit if it changed (uses TxRx to ensure it's actually set)
        if velocity_limit != self._last_velocity_limit:
            for motor_id in robot_config.motor_ids:
                self.set_profile_velocity(motor_id, velocity_limit, use_tx_only=False)
            self._last_velocity_limit = velocity_limit
            time.sleep(0.1)  # Allow velocity commands to be processed
        
        # Send position commands using GroupSyncWrite (bulk write for low latency)
        self.groupSyncWrite.clearParam()
        
        for motor_id, goal_pos in motor_positions.items():
            if motor_id not in robot_config.motor_ids:
                continue  # Skip invalid IDs silently in hot loop
            
            # Convert 32-bit position to 4-byte array (little-endian)
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(goal_pos)),
                DXL_HIBYTE(DXL_LOWORD(goal_pos)),
                DXL_LOBYTE(DXL_HIWORD(goal_pos)),
                DXL_HIBYTE(DXL_HIWORD(goal_pos))
            ]
            
            # Add parameter to sync write group
            dxl_addparam_result = self.groupSyncWrite.addParam(motor_id, param_goal_position)
            if not dxl_addparam_result:
                # If addParam fails, fall back to individual write
                self.packetHandler.write4ByteTxOnly(
                    self.portHandler, motor_id, robot_config.addr_goal_position, goal_pos
                )
        
        # Transmit all positions in a single packet
        dxl_comm_result = self.groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            # If sync write fails, fall back to individual writes
            for motor_id, goal_pos in motor_positions.items():
                if motor_id not in robot_config.motor_ids:
                    continue
                self.packetHandler.write4ByteTxOnly(
                    self.portHandler, motor_id, robot_config.addr_goal_position, goal_pos
                )
    
    def read_all_encoders(self, max_retries=3, retry_delay=0.1):
        """
        Read current encoder positions from all motors.
        
        Args:
            max_retries: Maximum number of retry attempts per motor
            retry_delay: Delay between retries (seconds)
            
        Returns:
            dict: {motor_id: encoder_position} for all motors, None if read failed
        """
        if not self.connected:
            return {}
        
        encoder_positions = {}
        for motor_id in robot_config.motor_ids:
            encoder_positions[motor_id] = None
            for attempt in range(max_retries):
                try:
                    dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                        self.portHandler, motor_id, robot_config.addr_present_position
                    )
                    if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                        encoder_positions[motor_id] = dxl_present_position
                        break  # Success, exit retry loop
                    elif attempt < max_retries - 1:
                        time.sleep(retry_delay)
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
        
        return encoder_positions
    
    def move_to_home(self, home_motor_positions, velocity_limit=30):
        """
        Move robot to home position (sim keyframe pose).
        
        Args:
            home_motor_positions: dict {motor_id: encoder_position} for home pose
            velocity_limit: Speed limit for movement
        """
        print("Moving robot to home position (sim keyframe)...")
        self.send_motor_positions(home_motor_positions, velocity_limit=velocity_limit)
        print("Home position command sent. Waiting for movement to complete...")
        time.sleep(3.0)  # Wait for movement to complete
        print("Ready for SpaceMouse control")
    
