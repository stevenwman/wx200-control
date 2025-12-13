"""
Robot controller that orchestrates: input → pose control → IK → actuation.

This controller is agnostic to the input source (SpaceMouse, NN, etc.) and
can work with both simulation and real robot hardware.
"""
import numpy as np
import mink
from ee_pose_controller import EndEffectorPoseController


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
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=model, cost=posture_cost)
        self.tasks = [self.end_effector_task, self.posture_task]
        
        # Initialize posture task target from initial configuration
        # This needs to be set before using the controller
        self._posture_target_initialized = False
        
        # IK parameters (optimized for low latency)
        self.solver = "quadprog"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 5  # Reduced from 20 for lower latency (50Hz control, can iterate next frame)
    
    def initialize_posture_target(self, configuration):
        """
        Initialize posture task target from configuration.
        Must be called once before using the controller.
        
        Args:
            configuration: mink.Configuration to use for initial posture
        """
        self.posture_task.set_target_from_configuration(configuration)
        self._posture_target_initialized = True
    
    def update_from_velocity_command(self, velocity_world, angular_velocity_world, dt, configuration):
        """
        Update controller with velocity commands and solve IK.
        
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
        
        # Update pose controller with velocity commands
        self.pose_controller.update_from_velocity_command(
            velocity_world=velocity_world,
            angular_velocity_world=angular_velocity_world,
            dt=dt
        )
        
        # Get target pose and set for IK
        target_pose = self.pose_controller.get_target_pose_se3()
        self.end_effector_task.set_target(target_pose)
        
        # Solve IK
        return self._converge_ik(configuration, dt)
    
    def _converge_ik(self, configuration, dt):
        """
        Run IK solver to converge to target pose.
        
        Args:
            configuration: mink.Configuration to update
            dt: Time step (s)
        
        Returns:
            bool: True if converged, False otherwise
        """
        for _ in range(self.max_iters):
            vel = mink.solve_ik(configuration, self.tasks, dt, self.solver, 1e-3)
            configuration.integrate_inplace(vel, dt)
            
            # Check convergence
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
