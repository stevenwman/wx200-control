from os.path import expanduser
import os

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import imageio
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic import DATASET_REGISTRY
from utils.datasets import Dataset


def is_robomimic_env(env_name):
    """determine if an env is robomimic"""
    try:
        if ("low_dim" not in env_name) and ("image" not in env_name):
            return False
        task, dataset_type, hdf5_type = env_name.split("-")
        return task in ("lift", "can", "square", "transport", "tool_hang") and dataset_type in ("mh", "ph")
    except ValueError:
        return False

low_dim_keys = {"low_dim": ('robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object')}
ObsUtils.initialize_obs_modality_mapping_from_dict(low_dim_keys)


def _get_max_episode_length(env_name):
    if env_name.startswith("lift"):
        return 300
    elif env_name.startswith("can"):
        return 300
    elif env_name.startswith("square"):
        return 400
    elif env_name.startswith("transport"):
        return 800
    elif env_name.startswith("tool_hang"):
        return 1000
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


def make_env(env_name, seed=0):
    """
    NOTE: should get_dataset() first, so that the metadata is downloaded before creating the environment
    """
    dataset_path = _check_dataset_exists(env_name)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    max_episode_length = _get_max_episode_length(env_name)
    
    if "image" in env_name:
        shape_meta = {
            'obs': {
                'rgb': {
                    'shape': (3, 96, 96)
                },
                'state': {
                    'shape': (9,)
                }
            },
            'action': {
                'shape': (7,)
            }
        }
        
        if "square" in env_name:
            image_keys = ["agentview_image"]
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
        
        obs_keys = {"low_dim": ['robot0_eef_pos',
                                    'robot0_eef_quat',
                                    'robot0_gripper_qpos',
                                    'object'],
                        'rgb': image_keys}
        ObsUtils.initialize_obs_modality_mapping_from_dict(obs_keys)
        
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=True,
        )
        env = RobomimicImageWrapper(env, shape_meta=shape_meta, image_keys=image_keys, max_episode_length=max_episode_length)
    else:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
        )
        env = RobomimicLowdimWrapper(env, low_dim_keys=low_dim_keys["low_dim"], max_episode_length=max_episode_length)
    env.seed(seed)
    env.env.hard_reset = False
    return env

def _check_dataset_exists(env_name):
    # enforce that the dataset exists
    task, dataset_type, hdf5_type = env_name.split("-")
    if hdf5_type == "image":
        file_name = "image_v15.hdf5"
    elif dataset_type == "mg":
        file_name = "low_dim_sparse_v15.hdf5"
    else:
        file_name = f"low_dim_v15.hdf5"
    dataset_root = os.environ.get("ROBOMIMIC_DATASET_ROOT")

    if dataset_root is None:
        username = os.environ.get("USER")
        if username:
            dataset_root = os.path.join("/data/user_data", username, "robomimic")
        else:
            dataset_root = os.path.join(expanduser("~"), ".robomimic")

    dataset_path = os.path.join(
        dataset_root,
        task,
        dataset_type,
        file_name,
    )
    # assert os.path.exists(dataset_path)

    return dataset_path

def get_dataset(env, env_name):
    dataset_path = _check_dataset_exists(env_name)

    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])

    print(f"the size of the dataset is {num_timesteps}")
    example_action = env.action_space.sample()

    # Check if this is an image environment
    is_image_env = "image" in env_name

    # data holder
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []

    # go through and add to the data holder
    for ep in demos:
        a = np.array(rm_dataset["data/{}/actions".format(ep)])
        dones = np.array(rm_dataset["data/{}/dones".format(ep)])
        r = np.array(rm_dataset["data/{}/rewards".format(ep)])
        
        if is_image_env:
            # Load image observations only
            if "square" in env_name:
                image_keys = ["agentview_image"]
            else:
                raise ValueError(f"Unsupported image environment: {env_name}")
            
            obs_data = []
            next_obs_data = []
            for key in image_keys:
                # Load images (H, W, C) and convert to float32
                obs_images = np.array(rm_dataset[f"data/{ep}/obs/{key}"]).astype(np.float32)  # (T, H, W, C)
                next_obs_images = np.array(rm_dataset[f"data/{ep}/next_obs/{key}"]).astype(np.float32)  # (T, H, W, C)
                
                obs_data.append(obs_images)
                next_obs_data.append(next_obs_images)
            
            # Concatenate multiple camera views along the channel dimension (last axis) if present
            obs = np.concatenate(obs_data, axis=3) if len(obs_data) > 1 else obs_data[0]            # (T, H, W, C_total)
            next_obs = np.concatenate(next_obs_data, axis=3) if len(next_obs_data) > 1 else next_obs_data[0]
        else:
            # Load low-dimensional observations
            obs, next_obs = [], []
            for k in low_dim_keys["low_dim"]:
                obs.append(np.array(rm_dataset[f"data/{ep}/obs/{k}"]))
            for k in low_dim_keys["low_dim"]:
                next_obs.append(np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]))
            obs = np.concatenate(obs, axis=-1)
            next_obs = np.concatenate(next_obs, axis=-1)

        observations.append(obs.astype(np.float32))
        actions.append(a.astype(np.float32))
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))
        next_observations.append(next_obs.astype(np.float32))
    
    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )


class RobomimicLowdimWrapper(gym.Env):
    """
    Environment wrapper for Robomimic environments with state observations.
    Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
    """
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
    ):
        self.env = env
        self.obs_keys = low_dim_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces - use [-1, 1]
        low = np.full(env.action_dimension, fill_value=-1.)
        high = np.full(env.action_dimension, fill_value=1.)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self):
        raw_obs = self.env.get_observation()
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            return self.normalize_obs(raw_obs)
        return raw_obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options=None, **kwargs):
        """Ignore passed-in arguments like seed"""

        self.t = 0
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Handle None options (common when called from vectorized environments)
        if options is None:
            options = {}

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            # always reset to the same state to be compatible with gym
            self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            self.env.reset()
        else:
            # random reset
            self.env.reset()

        return self.get_observation(), {}

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            obs = self.normalize_obs(raw_obs)
        else:
            obs = raw_obs

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # print(obs, reward, done, info)
        if reward > 0.:
            done = True
            info["success"] = 1
            info['return'] = self.episode_return
            info['length'] = self.episode_length
        else:
            info["success"] = 0

        if done:
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}



class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        
        # set up normalization for actions only (no state normalization needed for pixel-only)
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.image_keys = image_keys
        
        # Set up observation space for pixel-only observations
        # Shape should be (H, W, C_total) for concatenated camera views where C_total = 3 * num_cameras
        # Note: Flax's Conv layer expects channel-last (NHWC) format, so we keep images in HWC order here.
        rgb_shape = shape_meta["obs"]["rgb"]["shape"]  # (3, 96, 96) as (C, H, W)
        channels, height, width = rgb_shape
        num_cameras = len(image_keys)
        full_shape = (height, width, channels * num_cameras)
        
        self.observation_space = Box(
            low=0,
            high=1,
            shape=full_shape,
            dtype=np.float32,
        )

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        # Return only pixel observations, concatenating multiple camera views if present.
        # Keep images in HWC format (channel-last) as expected by Flax.
        obs_data = []
        for key in self.image_keys:
            if key in raw_obs:
                img = raw_obs[key]  # (H, W, C)
                # Ensure float32 dtype for neural network consumption
                img = img.astype(np.float32)
                obs_data.append(img)
        
        # Concatenate multiple camera views along the channel dimension (last axis).
        if len(obs_data) > 1:
            obs = np.concatenate(obs_data, axis=2)  # (H, W, C_total)
        else:
            obs = obs_data[0]  # (H, W, C)
        
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options=None, **kwargs):
        """Reset environment and return observation and info"""
        if options is None:
            options = {}
            
        self.t = 0
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1
        
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs), {}

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Check for success
        if reward > 0.:
            done = True
            info["success"] = 1
        else:
            info["success"] = 0

        if done:
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}

if __name__ == "__main__":
    # for testing
    env = make_env("square-mh-image")
    dataset = get_dataset(env, "square-mh-image")
    print(dataset)
