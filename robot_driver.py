"""
Dynamixel robot driver for sending joint commands to real hardware.
"""
import time
from dynamixel_sdk import *
from robot_config import robot_config


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
        # Cache velocity limit to avoid setting it every loop
        self._last_velocity_limit = None
        # GroupSyncWrite for efficient bulk position writes
        self.groupSyncWrite = None
    
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
        
        # Reboot all motors first to clear any error states (red flashing LEDs)
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
        
        # Initialize GroupSyncWrite for bulk position writes (much faster than individual writes)
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler, 
            self.packetHandler, 
            robot_config.addr_goal_position, 
            4  # 4 bytes for goal position
        )
        
        self.connected = True
        print(f"Robot driver connected on {self.devicename}")
    
    def disconnect(self):
        """Disconnect from robot and disable torque."""
        if self.connected:
            # Only disable torque if port is still open (shutdown sequence may have closed it)
            # Use try-except since shutdown sequence may have already closed the port
            try:
                self.disable_torque_all()
            except Exception:
                pass  # Port may already be closed, ignore error
            
            if self.portHandler:
                try:
                    self.portHandler.closePort()
                except Exception:
                    pass  # Port may already be closed, ignore error
            
            self.connected = False
            print("Robot driver disconnected")
    
    def disable_torque_all(self):
        """Disable torque on all motors."""
        if not self.connected:
            return
        
        # Check if port handler exists
        if not self.portHandler:
            return
        
        print("Disabling torque on all motors...")
        for dxl_id in robot_config.motor_ids:
            try:
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, robot_config.addr_torque_enable, 0
                )
            except Exception:
                pass  # Ignore errors if port is closed or motor unreachable
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
            # Use TxOnly for low latency (no response wait)
            self.packetHandler.write4ByteTxOnly(
                self.portHandler, motor_id, robot_config.addr_profile_velocity, velocity_limit
            )
        else:
            # Use TxRx (blocking, waits for response) - slower but confirms write
            self.packetHandler.write4ByteTxRx(
                self.portHandler, motor_id, robot_config.addr_profile_velocity, velocity_limit
            )
    
    def set_profile_velocity_all(self, velocity_limit):
        """Set profile velocity for all motors."""
        for motor_id in robot_config.motor_ids:
            self.set_profile_velocity(motor_id, velocity_limit)
    
    def send_motor_positions(self, motor_positions, velocity_limit=30):
        """
        Send motor position commands (optimized for low latency).
        
        Uses GroupSyncWrite to send all motor positions in a single packet for maximum speed.
        This is the production implementation - robot_driver_profiling.py wraps this
        with identical control logic plus timing measurements.
        
        Args:
            motor_positions: dict {motor_id: encoder_position}
            velocity_limit: Speed limit for movement (0=Max, 30=Slow/Safe)
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")
        
        # Only set velocity limit if it changed (optimization)
        # Use TxOnly for velocity setting too - we don't need to wait for confirmation
        # This significantly reduces latency (TxRx is ~15ms per motor, TxOnly is ~1-2ms)
        if velocity_limit != self._last_velocity_limit:
            for motor_id in motor_positions.keys():
                self.set_profile_velocity(motor_id, velocity_limit, use_tx_only=True)
            self._last_velocity_limit = velocity_limit
        
        # Send position commands using GroupSyncWrite (bulk write - much faster!)
        # This sends all motor positions in a single packet instead of 7 separate packets
        # Should reduce send time from ~20ms to ~2-5ms
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
    
    def read_present_current(self, motor_id):
        """
        Read present current from a motor.
        
        Args:
            motor_id: Motor ID to read from
            
        Returns:
            Current in mA (milliamps), or None if read failed
        """
        if not self.connected:
            return None
        
        try:
            result, comm_result, error = self.packetHandler.read2ByteTxRx(
                self.portHandler, motor_id, robot_config.addr_present_current
            )
            if comm_result == COMM_SUCCESS and error == 0:
                current_raw = result
                
                # Convert unsigned 16-bit to signed 16-bit (two's complement)
                # Values >= 32768 represent negative numbers
                if current_raw >= 32768:
                    current_raw_signed = current_raw - 65536
                else:
                    current_raw_signed = current_raw
                
                # Convert to milliamps based on value magnitude
                # Large values (>1000) are likely in 0.1A units, small values use 2.69 mA/unit
                # This heuristic handles different Dynamixel firmware versions
                if abs(current_raw_signed) > 1000:
                    # Value is in 0.1A units: multiply by 0.1 to get amps, then by 1000 for mA
                    current_ma = abs(current_raw_signed) * 0.1 * 1000
                else:
                    # Standard XM430/XM540 conversion: 1 unit = 2.69 mA
                    current_ma = abs(current_raw_signed) * 2.69
                
                return current_ma
            return None
        except Exception:
            return None
    
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
