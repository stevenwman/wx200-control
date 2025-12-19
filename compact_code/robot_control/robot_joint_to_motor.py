"""
Translation layer between IK joint commands and Dynamixel motor commands.

Handles the mapping from joint space (radians) to motor encoder space,
including the special case where joint 1 has 2 motors with flipped encoders.
"""
import numpy as np
from robot_control.robot_config import robot_config

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
            home_encoders: List of encoder positions [motor1, motor2, 3, 4, 5, 6, 7]
                          or dict {motor_id: encoder_position}
        """
        if isinstance(home_encoders, dict):
            # Convert dict to list in motor ID order
            self.home_encoders = [home_encoders.get(mid, robot_config.encoder_center) 
                                 for mid in robot_config.motor_ids]
        else:
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

def encoder_to_gripper_position(encoder_value):
    """
    Convert gripper encoder position to sim gripper position.
    
    Args:
        encoder_value: Encoder position (gripper_encoder_min=closed, gripper_encoder_max=open)
        
    Returns:
        Gripper position in meters (gripper_open_pos=open, gripper_closed_pos=closed)
    """
    from robot_control.robot_config import robot_config
    gripper_range = robot_config.gripper_encoder_max - robot_config.gripper_encoder_min
    sim_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
    encoder_normalized = (encoder_value - robot_config.gripper_encoder_min) / gripper_range
    return robot_config.gripper_open_pos + (1.0 - encoder_normalized) * sim_range


def encoders_to_joint_angles(encoder_positions, translator):
    """
    Convert encoder positions to joint angles (inverse mapping).
    
    This is the inverse of joint_commands_to_motor_positions - converts
    actual robot encoder readings back to joint space for simulation sync.
    
    Args:
        encoder_positions: dict {motor_id: encoder_position} from robot
        translator: JointToMotorTranslator instance (needed for joint1 offsets)
    
    Returns:
        np.ndarray: Joint angles in radians [q0, q1, q2, q3, q4, gripper]
    """
    from robot_control.robot_config import robot_config
    
    joint_angles = np.zeros(6)  # 5 joints + gripper
    
    # Joint 0 (base-1_z) <- Motor 1 (FLIPPED)
    if 1 in encoder_positions and encoder_positions[1] is not None:
        joint_angles[0] = -encoder_to_joint_angle(encoder_positions[1])
    
    # Joint 1 (link1-2_x) <- Motors 2 and 3 (FLIPPED, opposing)
    if 2 in encoder_positions and 3 in encoder_positions:
        if encoder_positions[2] is not None and encoder_positions[3] is not None:
            motor2_enc_relative = encoder_positions[2] - translator.joint1_motor2_offset
            motor2_angle = encoder_to_joint_angle(motor2_enc_relative)
            
            motor3_enc_relative = encoder_positions[3] - translator.joint1_motor3_offset
            motor3_enc_flipped = 2 * robot_config.encoder_center - motor3_enc_relative
            motor3_angle = encoder_to_joint_angle(motor3_enc_flipped)
            
            joint_angles[1] = -(motor2_angle + motor3_angle) / 2.0
    
    # Joint 2 (link2-3_x) <- Motor 4
    if 4 in encoder_positions and encoder_positions[4] is not None:
        joint_angles[2] = encoder_to_joint_angle(encoder_positions[4])
    
    # Joint 3 (link3-4_x) <- Motor 5
    if 5 in encoder_positions and encoder_positions[5] is not None:
        joint_angles[3] = encoder_to_joint_angle(encoder_positions[5])
    
    # Joint 4 (link4-5_y) <- Motor 6 (FLIPPED)
    if 6 in encoder_positions and encoder_positions[6] is not None:
        joint_angles[4] = -encoder_to_joint_angle(encoder_positions[6])
    
    # Gripper <- Motor 7
    if 7 in encoder_positions and encoder_positions[7] is not None:
        encoder = max(robot_config.gripper_encoder_min, 
                     min(robot_config.gripper_encoder_max, encoder_positions[7]))
        normalized = (encoder - robot_config.gripper_encoder_min) / \
                    (robot_config.gripper_encoder_max - robot_config.gripper_encoder_min)
        sim_gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
        joint_angles[5] = robot_config.gripper_open_pos + (1.0 - normalized) * sim_gripper_range
    
    return joint_angles


def sync_robot_to_mujoco(robot_encoders, translator, model, data, configuration):
    """
    Sync actual robot encoder positions to MuJoCo simulation.
    
    Reads encoder positions, converts to joint angles, updates MuJoCo state,
    and syncs mink Configuration. Includes safety checks for communication errors.
    
    Args:
        robot_encoders: dict {motor_id: encoder_position} from robot
        translator: JointToMotorTranslator instance
        model: MuJoCo model
        data: MuJoCo data
        configuration: mink.Configuration to update
    
    Returns:
        tuple: (robot_joint_angles, actual_position, actual_orientation_quat_wxyz)
            - robot_joint_angles: np.ndarray of joint angles [q0-q4, gripper]
            - actual_position: [x, y, z] end-effector position
            - actual_orientation_quat_wxyz: [w, x, y, z] end-effector orientation
    
    Raises:
        RuntimeError: If encoder reading fails or conversion results in invalid state
    """
    import numpy as np
    import mujoco
    from scipy.spatial.transform import Rotation as R
    
    # Check encoder reads
    successful_reads = sum(1 for v in robot_encoders.values() if v is not None)
    if successful_reads == 0:
        raise RuntimeError("Encoder reading failed - all encoders returned None")
    
    # Convert encoders to joint angles
    robot_joint_angles = encoders_to_joint_angles(robot_encoders, translator)
    
    # Safety check: abort if joint angles are all zero (indicates conversion failure)
    if np.allclose(robot_joint_angles[:5], 0, atol=1e-6):
        raise RuntimeError("Joint angle conversion failed - robot_joint_angles are all zero")
    
    # Update MuJoCo data with actual robot position
    data.qpos[:5] = robot_joint_angles[:5]
    if len(data.qpos) > 5:
        data.qpos[5] = robot_joint_angles[5]
    
    # Compute forward kinematics
    mujoco.mj_forward(model, data)
    
    # Update mink Configuration from MuJoCo data
    configuration.update(data.qpos)
    
    # Verify configuration was updated correctly
    if np.allclose(configuration.q[:5], 0, atol=1e-6):
        raise RuntimeError("Configuration update failed - configuration.q is still zero")
    
    # Get actual end-effector pose from robot
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    actual_position = data.site(site_id).xpos.copy()
    actual_site_xmat = data.site(site_id).xmat.reshape(3, 3)
    actual_site_rot = R.from_matrix(actual_site_xmat)
    actual_site_quat = actual_site_rot.as_quat()
    actual_orientation_quat_wxyz = np.array([
        actual_site_quat[3], 
        actual_site_quat[0], 
        actual_site_quat[1], 
        actual_site_quat[2]
    ])
    
    return robot_joint_angles, actual_position, actual_orientation_quat_wxyz
