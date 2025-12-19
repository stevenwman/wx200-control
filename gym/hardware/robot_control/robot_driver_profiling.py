"""
Profiling wrapper for RobotDriver.

Provides toggleable profiling that can be easily enabled/disabled.

IMPORTANT: This module's send_motor_positions() is IDENTICAL to robot_driver.py's
implementation in terms of control logic. The only differences are:
- Timing measurements (time.perf_counter() calls)
- Stats recording and printing

All control behavior (GroupSyncWrite, fallbacks, etc.) is exactly the same.
Physical performance should be identical - profiling only adds timing overhead.

Usage:
    from robot_driver_profiling import create_profiled_driver
    
    robot_driver = RobotDriver()
    if ENABLE_PROFILING:
        robot_driver = create_profiled_driver(robot_driver)
"""
import time
from dynamixel_sdk import *
from robot_control.robot_config import robot_config


class RobotDriverProfiled:
    """
    Profiled wrapper for RobotDriver that measures timing in send_motor_positions.
    """
    
    def __init__(self, robot_driver, stats_interval=100):
        """
        Wrap an existing RobotDriver to add profiling.
        
        Args:
            robot_driver: Existing RobotDriver instance
            stats_interval: Print stats every N command sends
        """
        self.robot_driver = robot_driver
        self.stats_interval = stats_interval
        self.profile_data = {
            'velocity_check_time': [],
            'velocity_set_time': [],
            'position_send_time': [],
            'total_time': [],
        }
        self.profile_count = 0
    
    def send_motor_positions(self, motor_positions, velocity_limit=30):
        """
        Profiled version of send_motor_positions.
        
        This is IDENTICAL to robot_driver.py's send_motor_positions() except for timing measurements.
        All control logic (GroupSyncWrite, fallbacks, etc.) is exactly the same.
        """
        t_total_start = time.perf_counter()
        
        if not self.robot_driver.connected:
            raise RuntimeError("Not connected to robot")
        
        # Profile velocity limit check and setting
        t_vel_start = time.perf_counter()
        velocity_check_time = 0
        velocity_set_time = 0
        
        # IDENTICAL to robot_driver.py: Only set velocity limit if it changed
        if velocity_limit != self.robot_driver._last_velocity_limit:
            t_set_start = time.perf_counter()
            for motor_id in motor_positions.keys():
                self.robot_driver.set_profile_velocity(motor_id, velocity_limit, use_tx_only=True)
            velocity_set_time = time.perf_counter() - t_set_start
            self.robot_driver._last_velocity_limit = velocity_limit
        
        velocity_check_time = time.perf_counter() - t_vel_start
        
        # Profile position command sending - IDENTICAL to robot_driver.py
        t_pos_start = time.perf_counter()
        
        # IDENTICAL: Send position commands using GroupSyncWrite (bulk write)
        self.robot_driver.groupSyncWrite.clearParam()
        
        for motor_id, goal_pos in motor_positions.items():
            if motor_id not in robot_config.motor_ids:
                continue  # Skip invalid IDs silently in hot loop
            
            # IDENTICAL: Convert 32-bit position to 4-byte array (little-endian)
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(goal_pos)),
                DXL_HIBYTE(DXL_LOWORD(goal_pos)),
                DXL_LOBYTE(DXL_HIWORD(goal_pos)),
                DXL_HIBYTE(DXL_HIWORD(goal_pos))
            ]
            
            # IDENTICAL: Add parameter to sync write group
            dxl_addparam_result = self.robot_driver.groupSyncWrite.addParam(motor_id, param_goal_position)
            if not dxl_addparam_result:
                # IDENTICAL: If addParam fails, fall back to individual write
                self.robot_driver.packetHandler.write4ByteTxOnly(
                    self.robot_driver.portHandler, motor_id, robot_config.addr_goal_position, goal_pos
                )
        
        # IDENTICAL: Transmit all positions in a single packet
        dxl_comm_result = self.robot_driver.groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            # IDENTICAL: If sync write fails, fall back to individual writes
            for motor_id, goal_pos in motor_positions.items():
                if motor_id not in robot_config.motor_ids:
                    continue
                self.robot_driver.packetHandler.write4ByteTxOnly(
                    self.robot_driver.portHandler, motor_id, robot_config.addr_goal_position, goal_pos
                )
        
        position_send_time = time.perf_counter() - t_pos_start
        t_total = time.perf_counter() - t_total_start
        
        # Record profiling data (ONLY difference from robot_driver.py)
        self.profile_data['velocity_check_time'].append(velocity_check_time * 1000)
        self.profile_data['velocity_set_time'].append(velocity_set_time * 1000)
        self.profile_data['position_send_time'].append(position_send_time * 1000)
        self.profile_data['total_time'].append(t_total * 1000)
        
        self.profile_count += 1
        
        # Print stats periodically (ONLY difference from robot_driver.py)
        if self.profile_count % self.stats_interval == 0:
            self.print_profile_stats()
    
    def print_profile_stats(self):
        """Print profiling statistics."""
        print("\n" + "="*70)
        print("SEND_MOTOR_POSITIONS BREAKDOWN")
        print("="*70)
        
        for key, times in self.profile_data.items():
            if len(times) > 0:
                avg = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                print(f"{key:25s}: avg={avg:6.2f} ms, max={max_time:6.2f} ms, min={min_time:6.2f} ms")
        
        print("="*70 + "\n")
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped robot_driver."""
        return getattr(self.robot_driver, name)


def create_profiled_driver(robot_driver, stats_interval=100):
    """
    Create a profiled wrapper around a RobotDriver instance.
    
    Args:
        robot_driver: RobotDriver instance to profile
        stats_interval: Print stats every N command sends
    
    Returns:
        RobotDriverProfiled instance wrapping the original driver
    """
    return RobotDriverProfiled(robot_driver, stats_interval=stats_interval)
