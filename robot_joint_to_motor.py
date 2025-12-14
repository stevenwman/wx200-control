"""
Translation layer between IK joint commands and Dynamixel motor commands.

Handles the mapping from joint space (radians) to motor encoder space,
including the special case where joint 1 has 2 motors with flipped encoders.
"""
import numpy as np
from robot_config import robot_config

# Joint to motor mapping
JOINT_TO_MOTOR_MAP = {
    0: [1],      # base-1_z -> motor 1
    1: [2, 3],   # link1-2_x -> motors 2 and 3 (flipped)
    2: [4],      # link2-3_x -> motor 4
    3: [5],      # link3-4_x -> motor 5
    4: [6],      # link4-5_y -> motor 6
    # Gripper is motor 7 (handled separately)
}

def joint_angle_to_encoder(angle_rad, center=None, max_angle_rad=np.pi):
    """
    Convert joint angle (radians) to Dynamixel encoder position.
    
    Args:
        angle_rad: Joint angle in radians
        center: Center encoder position (defaults to robot_config.encoder_center)
        max_angle_rad: Maximum joint angle in radians (default π)
    
    Returns:
        int: Encoder position (0-4095)
    """
    if center is None:
        center = robot_config.encoder_center
    
    angle_rad = np.clip(angle_rad, -max_angle_rad, max_angle_rad)
    encoder_offset = int((angle_rad / max_angle_rad) * (robot_config.encoder_max / 2))
    encoder_pos = center + encoder_offset
    return int(np.clip(encoder_pos, 0, robot_config.encoder_max))


def encoder_to_joint_angle(encoder_pos, center=None, max_angle_rad=np.pi):
    """
    Convert Dynamixel encoder position to joint angle (radians).
    
    Args:
        encoder_pos: Encoder position (0-4095)
        center: Center encoder position (defaults to robot_config.encoder_center)
        max_angle_rad: Maximum joint angle in radians (default π)
    
    Returns:
        float: Joint angle in radians
    """
    if center is None:
        center = robot_config.encoder_center
    encoder_offset = encoder_pos - center
    angle_rad = (encoder_offset / (robot_config.encoder_max / 2)) * max_angle_rad
    return angle_rad


class JointToMotorTranslator:
    """
    Translates IK joint commands to Dynamixel motor commands.
    Handles the special case of joint 1 with 2 motors.
    """
    
    def __init__(self, joint1_motor2_offset=0, joint1_motor3_offset=0):
        """
        Initialize translator.
        
        Args:
            joint1_motor2_offset: Offset for motor 2 when joint 1 is at 0 (encoder units)
            joint1_motor3_offset: Offset for motor 3 when joint 1 is at 0 (encoder units)
        """
        self.joint1_motor2_offset = joint1_motor2_offset
        self.joint1_motor3_offset = joint1_motor3_offset
        self.home_encoders = None
    
    def set_home_encoders(self, home_encoders):
        """
        Set home encoder positions from sim keyframe.
        
        Args:
            home_encoders: List of encoder positions [motor1, motor2, motor3, motor4, motor5, motor6, motor7]
        """
        self.home_encoders = home_encoders.copy()
    
    def joint_commands_to_motor_positions(self, joint_angles_rad, gripper_position=None):
        """
        Convert IK joint angles to Dynamixel motor positions.
        
        Args:
            joint_angles_rad: Array of 5 joint angles in radians [q0, q1, q2, q3, q4]
            gripper_position: Gripper position (optional, handled separately)
        
        Returns:
            dict: {motor_id: encoder_position} for all motors
        """
        motor_positions = {}
        
        # Joint 0 (base-1_z) -> Motor 1
        # NOTE: Robot hardware has this joint flipped relative to simulation
        if len(joint_angles_rad) > 0:
            motor_positions[1] = joint_angle_to_encoder(-joint_angles_rad[0])
        
        # Joint 1 (link1-2_x) -> Motors 2 and 3 (opposing motors for same joint)
        # NOTE: Robot hardware has this joint flipped, and uses two motors with opposing encoders
        # Motor 2: direct mapping (positive joint angle -> higher encoder)
        # Motor 3: inverted mapping (positive joint angle -> lower encoder, mirrored about center)
        if len(joint_angles_rad) > 1:
            encoder_joint1 = joint_angle_to_encoder(-joint_angles_rad[1])
            motor_positions[2] = int(np.clip(encoder_joint1 + self.joint1_motor2_offset, 0, robot_config.encoder_max))
            # Motor 3 is inverted: mirror encoder_joint1 about center, then add offset
            motor3_base = robot_config.encoder_center - (encoder_joint1 - robot_config.encoder_center)
            motor_positions[3] = int(np.clip(motor3_base + self.joint1_motor3_offset, 0, robot_config.encoder_max))
        
        # Joint 2 (link2-3_x) -> Motor 4 (no flip)
        if len(joint_angles_rad) > 2:
            motor_positions[4] = joint_angle_to_encoder(joint_angles_rad[2])
        
        # Joint 3 (link3-4_x) -> Motor 5 (no flip)
        if len(joint_angles_rad) > 3:
            motor_positions[5] = joint_angle_to_encoder(joint_angles_rad[3])
        
        # Joint 4 (link4-5_y) -> Motor 6
        # NOTE: Robot hardware has this joint flipped relative to simulation
        if len(joint_angles_rad) > 4:
            motor_positions[6] = joint_angle_to_encoder(-joint_angles_rad[4])
        
        # Gripper -> Motor 7
        # NOTE: Sim uses [-0.026 (open), 0.0 (closed)], robot uses [2776 (open), 1559 (closed)]
        # The mapping is inverted: sim open (negative) -> robot open (higher encoder)
        if gripper_position is not None:
            sim_gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
            sim_normalized = (gripper_position - robot_config.gripper_open_pos) / sim_gripper_range
            gripper_range = robot_config.gripper_encoder_max - robot_config.gripper_encoder_min
            encoder = robot_config.gripper_encoder_min + (1.0 - sim_normalized) * gripper_range
            motor_positions[7] = int(encoder)
        
        return motor_positions
    
    def get_home_motor_positions(self):
        """
        Get motor positions corresponding to sim keyframe home pose.
        
        Returns:
            dict: {motor_id: encoder_position} for home pose
        """
        if self.home_encoders is None:
            raise ValueError("Home encoders not set. Call set_home_encoders() first.")
        
        return {
            motor_id: encoder_pos 
            for motor_id, encoder_pos in zip(robot_config.motor_ids, self.home_encoders)
        }


def encoder_to_gripper_position(encoder_value):
    """
    Convert gripper encoder position to sim gripper position.
    
    Args:
        encoder_value: Encoder position (gripper_encoder_min=closed, gripper_encoder_max=open)
        
    Returns:
        Gripper position in meters (gripper_open_pos=open, gripper_closed_pos=closed)
    """
    from robot_config import robot_config
    gripper_range = robot_config.gripper_encoder_max - robot_config.gripper_encoder_min
    sim_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
    encoder_normalized = (encoder_value - robot_config.gripper_encoder_min) / gripper_range
    return robot_config.gripper_open_pos + (1.0 - encoder_normalized) * sim_range
