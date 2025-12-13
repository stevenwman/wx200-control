"""
Translation layer between IK joint commands and Dynamixel motor commands.

Handles the mapping from joint space (radians) to motor encoder space,
including the special case where joint 1 has 2 motors with flipped encoders.
"""
import numpy as np

# Motor IDs (7 motors total)
MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]

# Joint to motor mapping
# Joint indices: [0=base-1_z, 1=link1-2_x, 2=link2-3_x, 3=link3-4_x, 4=link4-5_y]
# Motor IDs: [1, 2, 3, 4, 5, 6, 7]
# Special case: Joint 1 (link1-2_x) is controlled by motors 2 and 3 (flipped)
JOINT_TO_MOTOR_MAP = {
    0: [1],      # base-1_z -> motor 1
    1: [2, 3],   # link1-2_x -> motors 2 and 3 (flipped)
    2: [4],      # link2-3_x -> motor 4
    3: [5],      # link3-4_x -> motor 5
    4: [6],      # link4-5_y -> motor 6
    # Gripper is motor 7 (handled separately)
}

# Dynamixel encoder parameters
# Encoder range: 0-4095 (12-bit, 360 degrees)
ENCODER_MAX = 4095
ENCODER_CENTER = 2048  # Middle position (180 degrees)

# Joint angle to encoder conversion
# Assuming joints can rotate ±180 degrees (or ±π radians)
# Encoder = center + (angle_rad / π) * (ENCODER_MAX / 2)
def joint_angle_to_encoder(angle_rad, center=ENCODER_CENTER, max_angle_rad=np.pi):
    """
    Convert joint angle (radians) to Dynamixel encoder position.
    
    Args:
        angle_rad: Joint angle in radians
        center: Center encoder position (default 2048)
        max_angle_rad: Maximum joint angle in radians (default π)
    
    Returns:
        int: Encoder position (0-4095)
    """
    # Normalize angle to [-max_angle, max_angle]
    angle_rad = np.clip(angle_rad, -max_angle_rad, max_angle_rad)
    
    # Convert to encoder: center ± (angle / max_angle) * (max_encoder / 2)
    encoder_offset = int((angle_rad / max_angle_rad) * (ENCODER_MAX / 2))
    encoder_pos = center + encoder_offset
    
    # Clip to valid range
    encoder_pos = int(np.clip(encoder_pos, 0, ENCODER_MAX))
    return encoder_pos


def encoder_to_joint_angle(encoder_pos, center=ENCODER_CENTER, max_angle_rad=np.pi):
    """
    Convert Dynamixel encoder position to joint angle (radians).
    
    Args:
        encoder_pos: Encoder position (0-4095)
        center: Center encoder position (default 2048)
        max_angle_rad: Maximum joint angle in radians (default π)
    
    Returns:
        float: Joint angle in radians
    """
    encoder_offset = encoder_pos - center
    angle_rad = (encoder_offset / (ENCODER_MAX / 2)) * max_angle_rad
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
                                  Note: motor 3 should be flipped relative to motor 2
        """
        self.joint1_motor2_offset = joint1_motor2_offset
        self.joint1_motor3_offset = joint1_motor3_offset
        
        # Home positions from sim keyframe: qpos="0 0 0 0 0 -0.01"
        # These are the encoder positions when joints are at home
        self.home_encoders = None  # Will be set from sim keyframe
    
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
        # IMPORTANT: Robot model has this joint FLIPPED
        if len(joint_angles_rad) > 0:
            # NEGATE joint angle because robot model is flipped
            joint0_angle = -joint_angles_rad[0]
            motor_positions[1] = joint_angle_to_encoder(joint0_angle)
        
        # Joint 1 (link1-2_x) -> Motors 2 and 3 (flipped)
        # IMPORTANT: 
        # 1. These two motors control the same joint but with flipped encoders
        # 2. The robot model has this joint FLIPPED compared to simulation
        #    So we need to negate the joint angle before converting to encoders
        if len(joint_angles_rad) > 1:
            # NEGATE joint angle because robot model is flipped
            joint1_angle = -joint_angles_rad[1]
            encoder_joint1 = joint_angle_to_encoder(joint1_angle)
            
            # Motor 2: direct mapping (positive angle -> higher encoder)
            motor_positions[2] = int(np.clip(encoder_joint1 + self.joint1_motor2_offset, 0, ENCODER_MAX))
            
            # Motor 3: flipped (inverted) mapping
            # When joint angle increases, motor 2 encoder increases, motor 3 encoder decreases
            # Formula: motor3 = center - (motor2 - center) + offset
            motor3_base = ENCODER_CENTER - (encoder_joint1 - ENCODER_CENTER)
            motor_positions[3] = int(np.clip(motor3_base + self.joint1_motor3_offset, 0, ENCODER_MAX))
        
        # Joint 2 (link2-3_x) -> Motor 4
        if len(joint_angles_rad) > 2:
            motor_positions[4] = joint_angle_to_encoder(joint_angles_rad[2])
        
        # Joint 3 (link3-4_x) -> Motor 5
        if len(joint_angles_rad) > 3:
            motor_positions[5] = joint_angle_to_encoder(joint_angles_rad[3])
        
        # Joint 4 (link4-5_y) -> Motor 6
        # IMPORTANT: Robot model has this joint FLIPPED
        if len(joint_angles_rad) > 4:
            # NEGATE joint angle because robot model is flipped
            joint4_angle = -joint_angles_rad[4]
            motor_positions[6] = joint_angle_to_encoder(joint4_angle)
        
        # Gripper -> Motors (handled separately, not part of IK)
        # In sim: gripper_l and gripper_r are coupled via equality constraint (gripper_r = -gripper_l)
        # gripper_l: -0.026 (open) to 0.0 (closed) in simulation
        # gripper_r: 0.0 (open) to 0.026 (closed) - opposite direction
        # IMPORTANT: 
        # 1. Gripper motor is FLIPPED
        # 2. Real robot gripper encoder range: [1559 (closed), 2776 (open)]
        #    Measured actual range, not centered around 2048
        if gripper_position is not None:
            # gripper_position is for gripper_l in meters (-0.026 to 0.0 in sim)
            sim_gripper_range = 0.026  # Sim range
            
            # Normalize sim position: -0.026 -> 0, 0.0 -> 1
            # 0 = open, 1 = closed
            sim_normalized = (gripper_position + sim_gripper_range) / sim_gripper_range  # 0 to 1
            
            # Real robot encoder range (measured)
            GRIPPER_ENCODER_MIN = 1559  # Closed position
            GRIPPER_ENCODER_MAX = 2776  # Open position
            GRIPPER_ENCODER_RANGE = GRIPPER_ENCODER_MAX - GRIPPER_ENCODER_MIN  # 1217
            
            # Map sim [0, 1] to real robot encoder [1559, 2776]
            # sim_normalized=0 (open, -0.026) -> encoder 2776 (open)
            # sim_normalized=1 (closed, 0.0) -> encoder 1559 (closed)
            # Note: sim uses inverted convention (open is negative), so we flip the mapping
            encoder = GRIPPER_ENCODER_MIN + (1.0 - sim_normalized) * GRIPPER_ENCODER_RANGE
            motor_positions[7] = int(encoder)  # 2776 (open) to 1559 (closed)
            
            # If there's a separate gripper_r motor, it should be the opposite
            # For now, assuming motor 7 controls gripper_l, and gripper_r is mechanically coupled
            # If you have a separate motor for gripper_r, uncomment and set the motor ID:
            # gripper_r_position = -gripper_position  # Opposite of gripper_l
            # gripper_r_normalized = (-gripper_r_position) / gripper_range
            # motor_positions[GRIPPER_R_MOTOR_ID] = int(gripper_r_normalized * ENCODER_MAX)
        
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
            for motor_id, encoder_pos in zip(MOTOR_IDS, self.home_encoders)
        }


def encoder_to_gripper_position(encoder_value):
    """
    Convert gripper encoder position to sim gripper position (inverse of joint_commands_to_motor_positions).
    
    Args:
        encoder_value: Encoder position (1559=closed, 2776=open)
        
    Returns:
        Gripper position in meters (-0.026=open, 0.0=closed)
    """
    GRIPPER_ENCODER_MIN = 1559  # Closed position
    GRIPPER_ENCODER_MAX = 2776  # Open position
    GRIPPER_ENCODER_RANGE = GRIPPER_ENCODER_MAX - GRIPPER_ENCODER_MIN  # 1217
    sim_gripper_range = 0.026  # Sim range
    
    # Normalize encoder: 1559 -> 1 (closed), 2776 -> 0 (open)
    encoder_normalized = (encoder_value - GRIPPER_ENCODER_MIN) / GRIPPER_ENCODER_RANGE  # 0 to 1
    
    # Convert to sim position: 0 (open) -> -0.026, 1 (closed) -> 0.0
    sim_position = -sim_gripper_range * (1.0 - encoder_normalized)
    
    return sim_position
