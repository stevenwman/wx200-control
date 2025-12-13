"""
Dynamixel robot driver for sending joint commands to real hardware.
"""
import time
from dynamixel_sdk import *

# Configuration
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

# Addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_PRESENT_CURRENT = 126  # 2 bytes, read-only

MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]


class RobotDriver:
    """
    Driver for controlling Dynamixel motors on the real robot.
    """
    
    def __init__(self, devicename=DEVICENAME, baudrate=BAUDRATE):
        """
        Initialize robot driver.
        
        Args:
            devicename: Serial port device name (e.g., '/dev/ttyUSB0')
            baudrate: Serial baudrate
        """
        self.devicename = devicename
        self.baudrate = baudrate
        self.portHandler = None
        self.packetHandler = None
        self.connected = False
        # Cache velocity limit to avoid setting it every loop
        self._last_velocity_limit = None
    
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
        
        for dxl_id in MOTOR_IDS:
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
            print(f"Rebooted {rebooted_count}/{len(MOTOR_IDS)} motors")
        else:
            print(f"WARNING: Failed to reboot any motors (0/{len(MOTOR_IDS)})")
            if failed_motors:
                print("Failed motors:")
                for motor_id, error in failed_motors[:3]:  # Show first 3 errors
                    print(f"  Motor {motor_id}: {error}")
        
        time.sleep(0.5)  # Wait for motors to initialize after reboot
    
    def connect(self):
        """Connect to robot, reboot motors to clear errors, and enable torque on all motors."""
        self.portHandler = PortHandler(self.devicename)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        
        # Open port
        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open port {self.devicename}")
        
        # Set baudrate
        if not self.portHandler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate}")
        
        # Reboot all motors first to clear any error states (red flashing LEDs)
        self.reboot_all_motors()
        
        # Enable torque on all motors
        for dxl_id in MOTOR_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1
            )
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError(f"Failed to enable torque on motor {dxl_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                raise RuntimeError(f"Motor {dxl_id} error: {self.packetHandler.getRxPacketError(dxl_error)}")
        
        self.connected = True
        print(f"Robot driver connected on {self.devicename}")
    
    def disconnect(self):
        """Disconnect from robot and disable torque."""
        if self.connected:
            self.disable_torque_all()
            if self.portHandler:
                self.portHandler.closePort()
            self.connected = False
            print("Robot driver disconnected")
    
    def disable_torque_all(self):
        """Disable torque on all motors."""
        if not self.connected:
            return
        
        print("Disabling torque on all motors...")
        for dxl_id in MOTOR_IDS:
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0
            )
        print("Torque disabled")
    
    def set_profile_velocity(self, motor_id, velocity_limit):
        """
        Set profile velocity (speed limit) for a motor.
        
        Args:
            motor_id: Motor ID
            velocity_limit: Speed limit (0=Max, higher=slower)
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")
        
        self.packetHandler.write4ByteTxRx(
            self.portHandler, motor_id, ADDR_PROFILE_VELOCITY, velocity_limit
        )
    
    def set_profile_velocity_all(self, velocity_limit):
        """Set profile velocity for all motors."""
        for motor_id in MOTOR_IDS:
            self.set_profile_velocity(motor_id, velocity_limit)
    
    def send_motor_positions(self, motor_positions, velocity_limit=30):
        """
        Send motor position commands (optimized for low latency).
        
        Args:
            motor_positions: dict {motor_id: encoder_position}
            velocity_limit: Speed limit for movement (0=Max, 30=Slow/Safe)
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")
        
        # Only set velocity limit if it changed (optimization)
        if velocity_limit != self._last_velocity_limit:
            for motor_id in motor_positions.keys():
                self.set_profile_velocity(motor_id, velocity_limit)
            self._last_velocity_limit = velocity_limit
        
        # Send position commands using TxOnly (no response wait) for low latency
        # This is fire-and-forget - much faster but no error checking
        for motor_id, goal_pos in motor_positions.items():
            if motor_id not in MOTOR_IDS:
                continue  # Skip invalid IDs silently in hot loop
            
            # Use TxOnly (transmit only, no response) for maximum speed
            # This reduces latency from ~15ms per motor to ~1-2ms per motor
            self.packetHandler.write4ByteTxOnly(
                self.portHandler, motor_id, ADDR_GOAL_POSITION, goal_pos
            )
        
        # Note: Removed clearPort() from hot loop as it can cause port conflicts
        # The port will be flushed automatically on next operation
    
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
                self.portHandler, motor_id, ADDR_PRESENT_CURRENT
            )
            if comm_result == COMM_SUCCESS and error == 0:
                current_raw = result
                # Current is signed 16-bit (two's complement)
                # Handle signed 16-bit: values >= 32768 are negative
                if current_raw >= 32768:
                    current_raw_signed = current_raw - 65536
                else:
                    current_raw_signed = current_raw
                
                # For XM430/XM540: 1 unit = 2.69 mA
                # But the raw value might be in a different format
                # Let's try: if raw value seems too high, maybe it's already in 0.1A units
                # Test: if abs(raw) > 1000, assume it's in 0.1A units (divide by 10)
                # Otherwise use the 2.69 conversion
                if abs(current_raw_signed) > 1000:
                    # Likely in 0.1A units (100mA per unit), convert to mA
                    current_ma = abs(current_raw_signed) * 0.1 * 1000  # Convert 0.1A units to mA
                else:
                    # Use standard conversion: 1 unit = 2.69 mA
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
