"""
Robot kinematics and control logic.

Combines EndEffectorPoseController, RobotController, and JointToMotorTranslator
into a single module for the compact gym environment.
"""
import numpy as np
import mink
import mujoco
from scipy.spatial.transform import Rotation as R
from .robot_config import robot_config


# =============================================================================
# JOINT TO MOTOR TRANSLATION
# =============================================================================

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
        self.joint1_motor2_offset = joint1_motor2_offset
        self.joint1_motor3_offset = joint1_motor3_offset
        self.home_encoders = None
    
    def set_home_encoders(self, home_encoders):
        if isinstance(home_encoders, dict):
            self.home_encoders = [home_encoders.get(mid, robot_config.encoder_center) 
                                 for mid in robot_config.motor_ids]
        else:
            self.home_encoders = home_encoders.copy()
    
    def joint_commands_to_motor_positions(self, joint_angles_rad, gripper_position=None):
        """Convert IK joint angles to Dynamixel motor positions."""
        motor_positions = {}
        
        # Joint 0 (base-1_z) -> Motor 1
        if len(joint_angles_rad) > 0:
            motor_positions[1] = joint_angle_to_encoder(-joint_angles_rad[0])
        
        # Joint 1 (link1-2_x) -> Motors 2 and 3
        if len(joint_angles_rad) > 1:
            encoder_joint1 = joint_angle_to_encoder(-joint_angles_rad[1])
            motor_positions[2] = int(np.clip(encoder_joint1 + self.joint1_motor2_offset, 0, robot_config.encoder_max))
            # Motor 3 is inverted
            motor3_base = robot_config.encoder_center - (encoder_joint1 - robot_config.encoder_center)
            motor_positions[3] = int(np.clip(motor3_base + self.joint1_motor3_offset, 0, robot_config.encoder_max))
        
        # Joint 2 (link2-3_x) -> Motor 4
        if len(joint_angles_rad) > 2:
            motor_positions[4] = joint_angle_to_encoder(joint_angles_rad[2])
        
        # Joint 3 (link3-4_x) -> Motor 5
        if len(joint_angles_rad) > 3:
            motor_positions[5] = joint_angle_to_encoder(joint_angles_rad[3])
        
        # Joint 4 (link4-5_y) -> Motor 6
        if len(joint_angles_rad) > 4:
            motor_positions[6] = joint_angle_to_encoder(-joint_angles_rad[4])
        
        # Gripper -> Motor 7
        if gripper_position is not None:
            sim_gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
            sim_normalized = (gripper_position - robot_config.gripper_open_pos) / sim_gripper_range
            gripper_range = robot_config.gripper_encoder_max - robot_config.gripper_encoder_min
            encoder = robot_config.gripper_encoder_min + (1.0 - sim_normalized) * gripper_range
            motor_positions[7] = int(encoder)
        
        return motor_positions


def encoder_to_gripper_position(encoder_value):
    """Convert gripper encoder position to sim gripper position."""
    gripper_range = robot_config.gripper_encoder_max - robot_config.gripper_encoder_min
    sim_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
    encoder_normalized = (encoder_value - robot_config.gripper_encoder_min) / gripper_range
    return robot_config.gripper_open_pos + (1.0 - encoder_normalized) * sim_range


def encoders_to_joint_angles(encoder_positions, translator):
    """Convert encoder positions to joint angles (inverse mapping)."""
    joint_angles = np.zeros(6)  # 5 joints + gripper
    
    # Joint 0
    if 1 in encoder_positions and encoder_positions[1] is not None:
        joint_angles[0] = -encoder_to_joint_angle(encoder_positions[1])
    
    # Joint 1
    if 2 in encoder_positions and 3 in encoder_positions:
        if encoder_positions[2] is not None and encoder_positions[3] is not None:
            motor2_enc_relative = encoder_positions[2] - translator.joint1_motor2_offset
            motor2_angle = encoder_to_joint_angle(motor2_enc_relative)
            
            motor3_enc_relative = encoder_positions[3] - translator.joint1_motor3_offset
            motor3_enc_flipped = 2 * robot_config.encoder_center - motor3_enc_relative
            motor3_angle = encoder_to_joint_angle(motor3_enc_flipped)
            
            joint_angles[1] = -(motor2_angle + motor3_angle) / 2.0
    
    # Joint 2
    if 4 in encoder_positions and encoder_positions[4] is not None:
        joint_angles[2] = encoder_to_joint_angle(encoder_positions[4])
    
    # Joint 3
    if 5 in encoder_positions and encoder_positions[5] is not None:
        joint_angles[3] = encoder_to_joint_angle(encoder_positions[5])
    
    # Joint 4
    if 6 in encoder_positions and encoder_positions[6] is not None:
        joint_angles[4] = -encoder_to_joint_angle(encoder_positions[6])
    
    # Gripper
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
    """
    # Check encoder reads
    successful_reads = sum(1 for v in robot_encoders.values() if v is not None)
    if successful_reads == 0:
        raise RuntimeError("Encoder reading failed - all encoders returned None")
    
    # Convert encoders to joint angles
    robot_joint_angles = encoders_to_joint_angles(robot_encoders, translator)
    
    # Safety check
    if np.allclose(robot_joint_angles[:5], 0, atol=1e-6):
        raise RuntimeError("Joint angle conversion failed - robot_joint_angles are all zero")
    
    # Update MuJoCo data
    data.qpos[:5] = robot_joint_angles[:5]
    if len(data.qpos) > 5:
        data.qpos[5] = robot_joint_angles[5]
    
    # Compute forward kinematics
    mujoco.mj_forward(model, data)
    
    # Update mink Configuration
    configuration.update(data.qpos)
    
    # Verify configuration
    if np.allclose(configuration.q[:5], 0, atol=1e-6):
        raise RuntimeError("Configuration update failed - configuration.q is still zero")
    
    # Get actual end-effector pose
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


# =============================================================================
# END EFFECTOR POSE CONTROLLER
# =============================================================================

class EndEffectorPoseController:
    """
    Controller that takes world-frame velocity commands and produces target poses for IK.
    """
    
    def __init__(self, initial_position, initial_orientation_quat_wxyz):
        self.target_position = np.array(initial_position, dtype=np.float64).copy()
        self.target_quat_wxyz = np.array(initial_orientation_quat_wxyz, dtype=np.float64).copy()
        self.target_quat_wxyz = self.target_quat_wxyz / np.linalg.norm(self.target_quat_wxyz)
    
    def update_from_velocity_command(self, velocity_world, angular_velocity_world, dt):
        """Update target pose by integrating velocity commands."""
        # Integrate linear velocity
        self.target_position = self.target_position + velocity_world * dt
        
        # Integrate angular velocity
        omega_magnitude = np.linalg.norm(angular_velocity_world)
        if omega_magnitude > 1e-6:
            omega_normalized = angular_velocity_world / omega_magnitude
            theta = np.clip(omega_magnitude * dt, 0, np.pi)
            
            # Convert [w, x, y, z] -> [x, y, z, w] for scipy
            quat_xyzw = np.array([self.target_quat_wxyz[1], self.target_quat_wxyz[2], 
                                  self.target_quat_wxyz[3], self.target_quat_wxyz[0]])
            current_rot = R.from_quat(quat_xyzw)
            
            # Apply rotation via left multiplication (world frame)
            delta_rot = R.from_rotvec(omega_normalized * theta)
            new_rot = delta_rot * current_rot
            
            # Convert back: [x, y, z, w] -> [w, x, y, z]
            new_quat_xyzw = new_rot.as_quat()
            self.target_quat_wxyz = np.array([new_quat_xyzw[3], new_quat_xyzw[0], 
                                               new_quat_xyzw[1], new_quat_xyzw[2]])
            self.target_quat_wxyz = self.target_quat_wxyz / np.linalg.norm(self.target_quat_wxyz)
    
    def get_target_pose_se3(self):
        """Get the current target pose as an mink.SE3 object."""
        # mink.SE3 expects [w, x, y, z, x, y, z]
        return mink.SE3(np.concatenate([self.target_quat_wxyz, self.target_position]))
    
    def get_target_position(self):
        return self.target_position.copy()
    
    def get_target_orientation_quat_wxyz(self):
        return self.target_quat_wxyz.copy()
    
    def reset_pose(self, position, orientation_quat_wxyz):
        self.target_position = np.array(position, dtype=np.float64).copy()
        self.target_quat_wxyz = np.array(orientation_quat_wxyz, dtype=np.float64).copy()
        self.target_quat_wxyz = self.target_quat_wxyz / np.linalg.norm(self.target_quat_wxyz)


# =============================================================================
# ROBOT CONTROLLER
# =============================================================================

class RobotController:
    """
    Main robot controller that handles the control pipeline:
    Input → Pose Control → IK → Joint Commands
    """
    
    def __init__(self, model, initial_position, initial_orientation_quat_wxyz,
                 position_cost=1.0, orientation_cost=0.1, posture_cost=1e-2):
        self.model = model
        
        # Initialize pose controller
        self.pose_controller = EndEffectorPoseController(
            initial_position=initial_position,
            initial_orientation_quat_wxyz=initial_orientation_quat_wxyz
        )
        
        # Initialize IK tasks
        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=robot_config.ik_lm_damping,
        )
        self.posture_task = mink.PostureTask(model=model, cost=posture_cost)
        self.tasks = [self.end_effector_task, self.posture_task]
        
        self._posture_target_initialized = False
        
        # IK solver parameters
        self.solver = robot_config.ik_solver
        self.pos_threshold = robot_config.ik_pos_threshold
        self.ori_threshold = robot_config.ik_ori_threshold
        self.max_iters = robot_config.ik_max_iters
        
        self._prev_velocity_active = False
    
    def initialize_posture_target(self, configuration):
        self.posture_task.set_target_from_configuration(configuration)
        self._posture_target_initialized = True
    
    def _get_current_pose_from_configuration(self, configuration):
        """Get current end-effector pose from configuration using MuJoCo forward kinematics."""
        data = mujoco.MjData(self.model)
        data.qpos[:len(configuration.q)] = configuration.q
        mujoco.mj_forward(self.model, data)
        
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.end_effector_task.frame_name)
        current_position = data.site(site_id).xpos.copy()
        current_xmat = data.site(site_id).xmat.reshape(3, 3)
        
        current_rot = R.from_matrix(current_xmat)
        quat_xyzw = current_rot.as_quat()
        orientation_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return current_position, orientation_quat_wxyz
    
    def update_from_velocity_command(self, velocity_world, angular_velocity_world, dt, configuration):
        """Update controller with velocity commands and solve IK."""
        if not self._posture_target_initialized:
            self.initialize_posture_target(configuration)
        
        # Detect zero-velocity transition (user released input)
        vel_magnitude = np.linalg.norm(velocity_world)
        omega_magnitude = np.linalg.norm(angular_velocity_world)
        velocity_threshold = 1e-4
        velocity_active = (vel_magnitude >= velocity_threshold) or (omega_magnitude >= velocity_threshold)
        
        if not velocity_active and self._prev_velocity_active:
            # User just released: reset target to current pose
            current_position, current_orientation = self._get_current_pose_from_configuration(configuration)
            self.pose_controller.reset_pose(current_position, current_orientation)
        elif velocity_active:
            # User providing input: integrate velocity
            self.pose_controller.update_from_velocity_command(
                velocity_world=velocity_world,
                angular_velocity_world=angular_velocity_world,
                dt=dt
            )
        
        self._prev_velocity_active = velocity_active
        
        # Get target pose and set for IK
        target_position = self.pose_controller.get_target_position()
        target_orientation = self.pose_controller.get_target_orientation_quat_wxyz()
        
        # Clamp target position to workspace bounds
        bounds_min = np.array([robot_config.ee_bound_x[0], robot_config.ee_bound_y[0], robot_config.ee_bound_z[0]])
        bounds_max = np.array([robot_config.ee_bound_x[1], robot_config.ee_bound_y[1], robot_config.ee_bound_z[1]])
        clamped_position = np.clip(target_position, bounds_min, bounds_max)
        
        if not np.allclose(clamped_position, target_position, atol=1e-9):
            self.pose_controller.reset_pose(clamped_position, target_orientation)
        
        target_pose = self.pose_controller.get_target_pose_se3()
        self.end_effector_task.set_target(target_pose)
        
        # Solve IK
        return self._converge_ik(configuration, dt)
    
    def _converge_ik(self, configuration, dt):
        for _ in range(self.max_iters):
            vel = mink.solve_ik(configuration, self.tasks, dt, self.solver, 1e-3)
            configuration.integrate_inplace(vel, dt)
            
            err = self.tasks[0].compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            
            if pos_achieved and ori_achieved:
                return True
        return False
    
    def get_joint_commands(self, configuration, num_joints=5):
        return configuration.q[:num_joints].copy()
    
    def get_target_pose(self):
        return self.pose_controller.get_target_pose_se3()
    
    def reset_pose(self, position, orientation_quat_wxyz):
        self.pose_controller.reset_pose(position, orientation_quat_wxyz)
