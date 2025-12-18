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
        self.groupSyncRead = None  # GroupSyncRead for bulk encoder reads
        
        # Statistics for monitoring packet drops
        self.encoder_read_stats = {
            'total_reads': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'timeout_reads': 0,
            'partial_reads': 0,  # Some motors succeeded, some failed
        }
    
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
        
        # Set packet timeout for reads
        # Note: Based on profiling, motors typically respond in 9-25ms (avg ~18ms, p95 ~25ms)
        # Setting timeout to 30ms to accommodate p95 while avoiding unnecessary waits
        # If you see many timeouts, consider checking USB latency timer (should be 1ms, not default 16ms)
        self.portHandler.setPacketTimeout(30)  # milliseconds (was 20ms, increased based on profiling)
        
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
        
        # Initialize GroupSyncRead for bulk encoder reads (4 bytes for present_position)
        self.groupSyncRead = GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            robot_config.addr_present_position,
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
    
    def read_all_encoders(self, max_retries=3, retry_delay=0.1, use_bulk_read=True, motor_ids=None):
        """
        Read current encoder positions from all motors.
        
        Uses GroupSyncRead for bulk reading (much faster) if available,
        otherwise falls back to individual reads.
        
        Args:
            max_retries: Maximum number of retry attempts (only used for fallback)
            retry_delay: Delay between retries (seconds, only used for fallback)
            use_bulk_read: If True, use GroupSyncRead for faster bulk reading
            motor_ids: List of motor IDs to read. If None, reads all motors.
            
        Returns:
            dict: {motor_id: encoder_position} for all motors, None if read failed
        """
        if not self.connected:
            return {}
        
        # Use provided motor_ids or default to all motors
        if motor_ids is None:
            motor_ids = robot_config.motor_ids
        
        # Try bulk read first (much faster - single packet)
        if use_bulk_read and self.groupSyncRead is not None:
            try:
                # Clear previous parameters
                self.groupSyncRead.clearParam()
                
                # Add specified motor IDs to sync read group
                for motor_id in motor_ids:
                    dxl_addparam_result = self.groupSyncRead.addParam(motor_id)
                    if not dxl_addparam_result:
                        # If addParam fails, fall back to individual reads
                        return self._read_all_encoders_individual(max_retries, retry_delay, motor_ids)
                
                # Transmit sync read packet
                # Note: We keep the default timeout (20ms set in connect()) as 15ms was too aggressive
                # and caused timeouts leading to fallback to slower individual reads
                self.encoder_read_stats['total_reads'] += 1
                
                # Time the actual txRxPacket call to see if we're waiting for timeout
                import time
                txrx_start = time.perf_counter()
                dxl_comm_result = self.groupSyncRead.txRxPacket()
                txrx_duration = (time.perf_counter() - txrx_start) * 1000  # Convert to ms
                
                # Track txRxPacket timing for analysis
                if not hasattr(self, '_txrx_times'):
                    self._txrx_times = []
                self._txrx_times.append(txrx_duration)
                if len(self._txrx_times) > 1000:
                    self._txrx_times.pop(0)
                
                if dxl_comm_result != COMM_SUCCESS:
                    # If sync read fails, fall back to individual reads
                    self.encoder_read_stats['failed_reads'] += 1
                    # Check if it was a timeout (check for common timeout error codes)
                    # Dynamixel SDK uses different error codes, check the error message
                    error_msg = self.packetHandler.getTxRxResult(dxl_comm_result)
                    if 'timeout' in error_msg.lower() or 'TIMEOUT' in error_msg:
                        self.encoder_read_stats['timeout_reads'] += 1
                    return self._read_all_encoders_individual(max_retries, retry_delay, motor_ids)
                
                # Read encoder values from sync read result
                encoder_positions = {}
                successful_motors = 0
                failed_motors = 0
                
                for motor_id in motor_ids:
                    # Check if data is available for this motor
                    dxl_getdata_result = self.groupSyncRead.isAvailable(
                        motor_id, robot_config.addr_present_position, 4
                    )
                    if dxl_getdata_result:
                        # Read 4-byte value (present_position)
                        dxl_present_position = self.groupSyncRead.getData(
                            motor_id, robot_config.addr_present_position, 4
                        )
                        encoder_positions[motor_id] = dxl_present_position
                        successful_motors += 1
                    else:
                        encoder_positions[motor_id] = None
                        failed_motors += 1
                
                # Track statistics
                if successful_motors == len(motor_ids):
                    self.encoder_read_stats['successful_reads'] += 1
                elif successful_motors > 0:
                    self.encoder_read_stats['partial_reads'] += 1
                    self.encoder_read_stats['failed_reads'] += 1
                else:
                    self.encoder_read_stats['failed_reads'] += 1
                
                # Warn if we're getting partial reads (some motors timed out)
                if failed_motors > 0 and self.encoder_read_stats['total_reads'] % 100 == 0:
                    failure_rate = (self.encoder_read_stats['failed_reads'] / 
                                   max(1, self.encoder_read_stats['total_reads'])) * 100
                    if failure_rate > 5.0:  # More than 5% failure rate
                        print(f"⚠️  Warning: Encoder read failure rate: {failure_rate:.1f}% "
                              f"({self.encoder_read_stats['failed_reads']}/{self.encoder_read_stats['total_reads']} reads failed)")
                        if self.encoder_read_stats['timeout_reads'] > 0:
                            timeout_rate = (self.encoder_read_stats['timeout_reads'] / 
                                          max(1, self.encoder_read_stats['failed_reads'])) * 100
                            print(f"   Timeout rate: {timeout_rate:.1f}% of failures "
                                  f"({self.encoder_read_stats['timeout_reads']} timeouts)")
                            print(f"   Consider increasing packet timeout if timeouts persist")
                
                # Fill in missing motors with None if we didn't read all
                if motor_ids != robot_config.motor_ids:
                    for motor_id in robot_config.motor_ids:
                        if motor_id not in encoder_positions:
                            encoder_positions[motor_id] = None
                
                return encoder_positions
            except Exception:
                # If bulk read fails, fall back to individual reads
                return self._read_all_encoders_individual(max_retries, retry_delay, motor_ids)
        else:
            # Use individual reads (slower but more reliable)
            return self._read_all_encoders_individual(max_retries, retry_delay, motor_ids)
    
    def _read_all_encoders_individual(self, max_retries=3, retry_delay=0.1, motor_ids=None):
        """
        Fallback method: Read encoder positions individually (slower but more reliable).
        
        Args:
            max_retries: Maximum number of retry attempts per motor
            retry_delay: Delay between retries (seconds)
            motor_ids: List of motor IDs to read. If None, reads all motors.
            
        Returns:
            dict: {motor_id: encoder_position} for all motors, None if read failed
        """
        if motor_ids is None:
            motor_ids = robot_config.motor_ids
        
        encoder_positions = {}
        for motor_id in motor_ids:
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
        
        # Fill in missing motors with None if we didn't read all
        if motor_ids != robot_config.motor_ids:
            for motor_id in robot_config.motor_ids:
                if motor_id not in encoder_positions:
                    encoder_positions[motor_id] = None
        
        return encoder_positions
    
    def get_encoder_read_stats(self):
        """
        Get statistics about encoder reads (success rate, timeouts, etc.).
        
        Returns:
            dict: Statistics about encoder reads
        """
        stats = self.encoder_read_stats.copy()
        if stats['total_reads'] > 0:
            stats['success_rate'] = (stats['successful_reads'] / stats['total_reads']) * 100
            stats['failure_rate'] = (stats['failed_reads'] / stats['total_reads']) * 100
            if stats['failed_reads'] > 0:
                stats['timeout_rate_of_failures'] = (stats['timeout_reads'] / stats['failed_reads']) * 100
            else:
                stats['timeout_rate_of_failures'] = 0.0
        
        # Add txRxPacket timing stats if available
        if hasattr(self, '_txrx_times') and len(self._txrx_times) > 0:
            import numpy as np
            txrx_array = np.array(self._txrx_times)
            stats['txrx_times_ms'] = {
                'avg': float(np.mean(txrx_array)),
                'min': float(np.min(txrx_array)),
                'max': float(np.max(txrx_array)),
                'p95': float(np.percentile(txrx_array, 95)),
                'p99': float(np.percentile(txrx_array, 99)),
                'count': len(txrx_array)
            }
            # Check if we're hitting the timeout (20ms) - if most reads are near 20ms, we're timeout-bound
            near_timeout = np.sum((txrx_array > 18.0) & (txrx_array < 22.0))
            stats['txrx_times_ms']['near_timeout_pct'] = (near_timeout / len(txrx_array)) * 100
        
        if stats['total_reads'] == 0:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['timeout_rate_of_failures'] = 0.0
        return stats
    
    def reset_encoder_read_stats(self):
        """Reset encoder read statistics."""
        self.encoder_read_stats = {
            'total_reads': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'timeout_reads': 0,
            'partial_reads': 0,
        }
    
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
    
