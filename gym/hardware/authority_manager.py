"""
Hardware Authority Manager for WX200 environments.

Provides a clean API for managing hardware authority transfer between
training and evaluation environments.
"""
from typing import Optional


class HardwareAuthorityManager:
    """
    Manages hardware authority for WX200 robot environments.
    
    Provides methods to:
    - Release authority from training env before eval
    - Release authority from eval env after eval
    - Check authority status
    - Reset authority state
    """
    
    @staticmethod
    def unwrap_to_base(env):
        """
        Unwrap environment to get base WX200GymEnvBase instance.
        
        Uses duck typing to identify WX200 environments by checking for the
        _hardware_authority class attribute, which is unique to WX200GymEnvBase.
        This works with any wrapper version (wx200_env_utils, wx200_env_utils_position_targets, etc.).
        
        Args:
            env: Environment (may be wrapped)
        
        Returns:
            WX200GymEnvBase instance or None if not a WX200 environment
        """
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Check if this is a WX200 environment by looking for the _hardware_authority class attribute
        # This is a unique identifier for WX200GymEnvBase regardless of which module it comes from
        env_class = type(base_env)
        if hasattr(env_class, '_hardware_authority') and hasattr(base_env, 'has_authority'):
            return base_env
        
        return None
    
    @staticmethod
    def _unwrap_to_base(env):
        """Private alias for backward compatibility."""
        return HardwareAuthorityManager.unwrap_to_base(env)
    
    @staticmethod
    def is_wx200_env(env_name: str) -> bool:
        """Check if environment name is a WX200 hardware environment."""
        return env_name.startswith("wx200") or env_name.startswith("WX200")
    
    @staticmethod
    def reset_authority():
        """
        Reset class-level authority (useful at script start).
        
        Note: This will only reset authority for classes that have been imported.
        For a more robust reset, pass an environment instance to unwrap_to_base()
        and reset that instance's class directly.
        """
        # Try to reset authority in known modules (may not cover all wrapper versions)
        # If you have a specific environment instance, it's better to reset via that instance
        try:
            from envs import wx200_env_utils_position_targets
            wx200_env_utils_position_targets.WX200GymEnvBase._hardware_authority = None
        except ImportError:
            pass
        try:
            from envs.wx200_env_utils import WX200GymEnvBase
            WX200GymEnvBase._hardware_authority = None
        except ImportError:
            pass
    
    @staticmethod
    def release_train_authority_for_eval(train_env, step: Optional[int] = None):
        """
        Release authority from training environment before eval.
        
        Args:
            train_env: Training environment (may be wrapped)
            step: Optional step number for logging
        """
        base_env = HardwareAuthorityManager.unwrap_to_base(train_env)
        if base_env is None:
            return
        
        # Get the actual class from the base_env instance (works for both modules)
        WX200GymEnvBase = type(base_env)
        
        # Release authority if this instance currently has it (check both has_authority flag and class-level variable)
        if WX200GymEnvBase._hardware_authority == base_env:
            if step is not None:
                print(f"Releasing hardware authority from training env before eval (step {step})...")
            else:
                print("Releasing hardware authority from training env before eval...")
            base_env.close()  # This will release authority and shut down hardware
            print("✓ Training env authority released")
        elif base_env.has_authority:
            # Inconsistent state: has_authority is True but class-level variable doesn't point to it
            # Force release anyway
            print(f"⚠️  Warning: Training env has_authority=True but class-level authority != self. Forcing release...")
            base_env.close()
            print("✓ Training env authority released")
        elif step is not None:
            print(f"✓ Training env already released authority (step {step})")
    
    @staticmethod
    def release_eval_authority_after_eval(eval_env, step: Optional[int] = None):
        """
        Release authority from eval environment after eval.
        
        Args:
            eval_env: Evaluation environment (may be wrapped)
            step: Optional step number for logging
        """
        base_eval = HardwareAuthorityManager.unwrap_to_base(eval_env)
        if base_eval is None:
            return
        
        # Get the actual class from the base_eval instance (works for both modules)
        WX200GymEnvBase = type(base_eval)
        
        # Release authority if this instance currently has it (check class-level variable)
        if WX200GymEnvBase._hardware_authority == base_eval:
            if step is not None:
                print(f"Releasing hardware authority from eval env after eval (step {step})...")
            else:
                print("Releasing hardware authority from eval env after eval...")
            base_eval.close()  # This will release authority and shut down hardware
            print("✓ Eval env authority released")
        elif base_eval.has_authority:
            # Inconsistent state: has_authority is True but class-level variable doesn't point to it
            # Force release anyway
            print(f"⚠️  Warning: Eval env has_authority=True but class-level authority != self. Forcing release...")
            base_eval.close()
            print("✓ Eval env authority released")
        elif step is not None:
            print(f"✓ Eval env already released authority (step {step})")
    
    @staticmethod
    def check_train_authority(train_env, context: str = ""):
        """
        Verify training environment has hardware authority.
        
        Args:
            train_env: Training environment (may be wrapped)
            context: Optional context string for error message
        
        Raises:
            RuntimeError: If training env does not have authority
        """
        base_env = HardwareAuthorityManager.unwrap_to_base(train_env)
        if base_env is None:
            return
        
        # Get the actual class from the base_env instance (works for both modules)
        WX200GymEnvBase = type(base_env)
        
        if not base_env.has_authority or WX200GymEnvBase._hardware_authority != base_env:
            raise RuntimeError(
                f"Training environment must have hardware authority {context}. "
                f"has_authority={base_env.has_authority}, "
                f"current_authority={WX200GymEnvBase._hardware_authority}"
            )
    
    @staticmethod
    def check_eval_no_authority(eval_env, step: Optional[int] = None):
        """
        Verify eval environment does NOT have hardware authority.
        
        Args:
            eval_env: Evaluation environment (may be wrapped)
            step: Optional step number for logging
        
        Raises:
            RuntimeError: If eval env has authority when it shouldn't
        """
        base_eval = HardwareAuthorityManager.unwrap_to_base(eval_env)
        if base_eval is None:
            return
        
        # Get the actual class from the base_eval instance (works for both modules)
        WX200GymEnvBase = type(base_eval)
        
        if base_eval.has_authority or WX200GymEnvBase._hardware_authority == base_eval:
            raise RuntimeError(
                "Eval environment should NOT have hardware authority. "
                "This would cause robot control conflicts."
            )
        
        if step is not None:
            print(f"✓ Authority check passed before eval (step {step})")
    
    @staticmethod
    def verify_initial_state(env, env_name: str):
        """
        Verify environment starts with no authority (for initial checks).
        
        Args:
            env: Environment (may be wrapped)
            env_name: Name for logging
        
        Raises:
            RuntimeError: If env has authority when it shouldn't
        """
        base_env = HardwareAuthorityManager.unwrap_to_base(env)
        if base_env is None:
            return
        
        # Get the actual class from the base_env instance (works for both modules)
        WX200GymEnvBase = type(base_env)
        
        has_auth = base_env.has_authority
        current_authority = WX200GymEnvBase._hardware_authority
        
        if has_auth or current_authority == base_env:
            raise RuntimeError(
                f"{env_name} should start with no authority but has_authority={has_auth}, "
                f"current_authority={current_authority}"
            )
        
        print(f"✓ {env_name} correctly starts with no authority")
    
    @staticmethod
    def emergency_stop(env):
        """
        Emergency stop: immediately disable torque without full shutdown.
        
        Args:
            env: Environment (may be wrapped) that currently has authority
        """
        base_env = HardwareAuthorityManager.unwrap_to_base(env)
        if base_env is None:
            return
        
        if base_env.has_authority and base_env._hardware_initialized and base_env.robot_base is not None:
            try:
                base_env.robot_base.robot_driver.emergency_disable_torque()
            except Exception as e:
                print(f"⚠️  Error during emergency stop: {e}")
                import traceback
                traceback.print_exc()
