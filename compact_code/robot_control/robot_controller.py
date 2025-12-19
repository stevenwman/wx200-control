"""
Robot controller that orchestrates: input → pose control → IK → actuation.

This controller is agnostic to the input source (SpaceMouse, NN, etc.) and
can work with both simulation and real robot hardware.
"""
import numpy as np
import mink
import mujoco
from scipy.spatial.transform import Rotation as R
from robot_control.ee_pose_controller import EndEffectorPoseController
from robot_control.robot_config import robot_config


class RobotController:
    """
    Main robot controller that handles the control pipeline:
    Input → Pose Control → IK → Joint Commands
    """
    
    def __init__(self, model, initial_position, initial_orientation_quat_wxyz,
                 position_cost=1.0, orientation_cost=0.1, posture_cost=1e-2):
        """
        Initialize robot controller.
        
        Args:
            model: MuJoCo model
            initial_position: [x, y, z] initial end-effector position in world frame
            initial_orientation_quat_wxyz: [w, x, y, z] initial orientation quaternion
            position_cost: Cost weight for position tracking
            orientation_cost: Cost weight for orientation tracking
            posture_cost: Cost weight for posture task
        """
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
        
        # Initialize posture task target from initial configuration
        # This needs to be set before using the controller
        self._posture_target_initialized = False
        
        # IK solver parameters
        self.solver = robot_config.ik_solver
        self.pos_threshold = robot_config.ik_pos_threshold
        self.ori_threshold = robot_config.ik_ori_threshold
        self.max_iters = robot_config.ik_max_iters
        
        # Track previous velocity state to detect zero-velocity transitions
        self._prev_velocity_active = False
    
    def initialize_posture_target(self, configuration):
        """
        Initialize posture task target from configuration.
        Must be called once before using the controller.
        
        Args:
            configuration: mink.Configuration to use for initial posture
        """
        self.posture_task.set_target_from_configuration(configuration)
        self._posture_target_initialized = True
    
    def _get_current_pose_from_configuration(self, configuration):
        """
        Get current end-effector pose from configuration using MuJoCo forward kinematics.
        
        Args:
            configuration: mink.Configuration
            
        Returns:
            tuple: (position, orientation_quat_wxyz) where orientation is [w, x, y, z]
        """
        # Create temporary MuJoCo data to compute forward kinematics
        data = mujoco.MjData(self.model)
        data.qpos[:len(configuration.q)] = configuration.q
        
        # Forward kinematics
        mujoco.mj_forward(self.model, data)
        
        # Get site transform
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.end_effector_task.frame_name)
        current_position = data.site(site_id).xpos.copy()
        current_xmat = data.site(site_id).xmat.reshape(3, 3)
        
        # Convert rotation matrix to quaternion [w, x, y, z]
        current_rot = R.from_matrix(current_xmat)
        quat_xyzw = current_rot.as_quat()  # [x, y, z, w]
        orientation_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return current_position, orientation_quat_wxyz
    
    def update_from_velocity_command(self, velocity_world, angular_velocity_world, dt, configuration):
        """
        Update controller with velocity commands and solve IK.
        
        When velocity is zero (user released input), reset target to current pose
        to prevent continued motion after release. This provides smooth, responsive control.
        
        Args:
            velocity_world: [vx, vy, vz] linear velocity in world frame (m/s)
            angular_velocity_world: [wx, wy, wz] angular velocity in world frame (rad/s)
            dt: Time step (s)
            configuration: mink.Configuration to update
        
        Returns:
            bool: True if IK converged, False otherwise
        """
        # Ensure posture target is initialized
        if not self._posture_target_initialized:
            self.initialize_posture_target(configuration)
        
        # Detect zero-velocity transition (user released input) for one-time target reset
        vel_magnitude = np.linalg.norm(velocity_world)
        omega_magnitude = np.linalg.norm(angular_velocity_world)
        velocity_threshold = 1e-4
        velocity_active = (vel_magnitude >= velocity_threshold) or (omega_magnitude >= velocity_threshold)
        
        if not velocity_active and self._prev_velocity_active:
            # User just released: reset target to current pose (one-time, prevents continued motion)
            current_position, current_orientation = self._get_current_pose_from_configuration(configuration)
            self.pose_controller.reset_pose(current_position, current_orientation)
        elif velocity_active:
            # User providing input: integrate velocity to update target
            self.pose_controller.update_from_velocity_command(
                velocity_world=velocity_world,
                angular_velocity_world=angular_velocity_world,
                dt=dt
            )
        
        # Update previous state
        self._prev_velocity_active = velocity_active
        
        # Get target pose and set for IK
        # Clamp target position to configured workspace bounds for safety
        target_position = self.pose_controller.get_target_position()
        target_orientation = self.pose_controller.get_target_orientation_quat_wxyz()
        
        bounds_min = np.array([
            robot_config.ee_bound_x[0],
            robot_config.ee_bound_y[0],
            robot_config.ee_bound_z[0],
        ])
        bounds_max = np.array([
            robot_config.ee_bound_x[1],
            robot_config.ee_bound_y[1],
            robot_config.ee_bound_z[1],
        ])
        clamped_position = np.clip(target_position, bounds_min, bounds_max)
        
        # If clamped, update the pose controller to avoid accumulating out-of-bounds targets
        if not np.allclose(clamped_position, target_position, atol=1e-9):
            self.pose_controller.reset_pose(clamped_position, target_orientation)
        
        # Set IK target (pose controller now holds the clamped position)
        target_pose = self.pose_controller.get_target_pose_se3()
        self.end_effector_task.set_target(target_pose)
        
        # Solve IK
        return self._converge_ik(configuration, dt)
    
    def _converge_ik(self, configuration, dt):
        """
        Run IK solver to converge to target pose.
        
        Uses iterative solver with early termination. Optimized for low latency
        (fewer iterations per cycle, relies on next cycle for continued convergence).
        
        Args:
            configuration: mink.Configuration to update
            dt: Time step (s)
        
        Returns:
            bool: True if converged, False otherwise
        """
        for _ in range(self.max_iters):
            vel = mink.solve_ik(configuration, self.tasks, dt, self.solver, 1e-3)
            configuration.integrate_inplace(vel, dt)
            
            # Check convergence (early termination if we're close enough)
            err = self.tasks[0].compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            
            if pos_achieved and ori_achieved:
                return True
        return False
    
    def get_joint_commands(self, configuration, num_joints=5):
        """
        Get joint position commands from IK solution.
        
        Args:
            configuration: mink.Configuration (should be updated after IK)
            num_joints: Number of joints to return commands for
        
        Returns:
            np.ndarray: Joint position commands [q1, q2, ..., qN]
        """
        return configuration.q[:num_joints].copy()
    
    def get_target_pose(self):
        """Get current target pose from pose controller."""
        return self.pose_controller.get_target_pose_se3()
    
    def reset_pose(self, position, orientation_quat_wxyz):
        """
        Reset target pose to a new value.
        
        Args:
            position: [x, y, z] position in world frame
            orientation_quat_wxyz: [w, x, y, z] orientation quaternion
        """
        self.pose_controller.reset_pose(position, orientation_quat_wxyz)
