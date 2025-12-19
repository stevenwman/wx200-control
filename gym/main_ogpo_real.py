import glob
from absl import app, flags
from ml_collections import config_flags
# Removed AsyncVectorEnv - real world is single-threaded only
from gymnasium.wrappers import RecordEpisodeStatistics
import tqdm
import wandb
import os
import json
import random
import numpy as np
import time
import jax
import pickle
import shutil
import signal
import sys

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

import flax
from utils.flax_utils import save_agent
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger
from envs.robomimic_utils import is_robomimic_env
# from envs.wx200_env_utils import is_wx200_env
from envs.wx200_env_utils_position_targets import is_wx200_env
from envs.env_utils import make_env_and_datasets, GYMNASIUM_ROBOTICS_ADROITHAND_ENVS
from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from agents import agents
from evaluation_ogpo_real import evaluate

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('run_name', 'Debug', 'Run name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('project', 'test', 'Project name.')
flags.DEFINE_string('wandb_name', 'test', 'WandB name.')
flags.DEFINE_integer('ep_resume', 0, 'Resume from episode number.')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('calql_steps', 0, 'Number of calql steps.')
flags.DEFINE_integer('q_warmup_steps', 0, 'Number of q_warmup steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval_bc', 200000, 'Evaluation interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('preemption_backup_interval', -1, 'Save interval for preemption backup.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')
flags.DEFINE_integer('step_data_log_interval', 10, 'Step data logging interval.')
flags.DEFINE_integer('best_of_n', 8, 'best of n actions')
flags.DEFINE_bool('clip_bc', False, "Clip BC to 50%")

flags.DEFINE_integer('utd_warmup', 1, "update to data ratio warmup")
flags.DEFINE_integer('utd_online', 1, "update to data ratio online")
flags.DEFINE_integer('n_eval_envs', 32, 'Number of parallel evaluation environments.')
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for 1B datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_float('discount', 0.99, 'discount factor')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_bool('log', True, "enable logging")
flags.DEFINE_bool('plot_q_vs_mc', False, "Plot Q vs MC returns")
flags.DEFINE_string('restore_critic_path', None, "restore critic path")
flags.DEFINE_string('restore_actor_path', None, "restore actor path")

flags.DEFINE_float('offline_ratio', 0.0, "ratio of offline data to use in mixed training (-1 = current naive scheme, 0.0 = pure online, 1.0 = pure offline)")
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')

flags.DEFINE_integer('success_buffer_batch_size', 256, "batch size of the success buffer.")
flags.DEFINE_bool('use_success_buffer', False, "whether to use the success buffer in the bc loss")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        if FLAGS.log:
            assert prefix in self.csv_loggers, prefix
            self.csv_loggers[prefix].log(data, step=step)
            self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)
            
def create_success_buffer_batch(train_dataset, replay_buffer, batch_size, offline_ratio, seq_len, discount):
    success_batch = replay_buffer.sample_sequence(
        batch_size=batch_size,
        sequence_length=seq_len,
        discount=discount,
        # by_score=True,
        success_only=True,
    )
    return success_batch

def get_checkpoint_dir(seed, env_name):
    user = os.environ.get('USER', 'unknown_user')
    checkpoint_path = os.path.join('/data/user_data', user, 'ogpo', f'{seed}_{env_name}')
    print(f"checkpoint dir: {checkpoint_path}")
    return checkpoint_path

def save_rolling_checkpoint(agent, replay_buffer, step, online_loop_step, seed, env_name):
    checkpoint_dir = get_checkpoint_dir(seed, env_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    agent_path = os.path.join(checkpoint_dir, f'{step}_agent.pkl')
    buffer_path = os.path.join(checkpoint_dir, f'{step}_buffer.pkl')
    meta_path = os.path.join(checkpoint_dir, f'{step}_meta.json')

    # Save new checkpoint first
    with open(agent_path, 'wb') as f:
        pickle.dump(flax.serialization.to_state_dict(agent), f)
        
    with open(buffer_path, 'wb') as f:
        pickle.dump(replay_buffer, f)
        
    with open(meta_path, 'w') as f:
        json.dump({'online_loop_step': online_loop_step}, f)
        
    print(f"Saved rolling checkpoint to {checkpoint_dir} at step {step} (online i={online_loop_step})")
    
    # Cleanup old checkpoints
    for filename in os.listdir(checkpoint_dir):
        if str(step) in filename:
            continue
        file_path = os.path.join(checkpoint_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def load_rolling_checkpoint(seed, env_name, agent_structure):
    checkpoint_dir = get_checkpoint_dir(seed, env_name)
    if not os.path.exists(checkpoint_dir):
        return None, None, None, None

    # Find latest step
    files = glob.glob(os.path.join(checkpoint_dir, "*_agent.pkl"))
    if not files:
        return None, None, None, None
        
    # Extract steps
    steps = []
    for f in files:
        try:
            basename = os.path.basename(f)
            step = int(basename.split('_')[0])
            steps.append(step)
        except:
            continue
            
    if not steps:
        return None, None, None, None
        
    latest_step = max(steps)
    
    agent_path = os.path.join(checkpoint_dir, f'{latest_step}_agent.pkl')
    buffer_path = os.path.join(checkpoint_dir, f'{latest_step}_buffer.pkl')
    meta_path = os.path.join(checkpoint_dir, f'{latest_step}_meta.json')
    
    if not os.path.exists(buffer_path):
        print(f"Buffer checkpoint missing for step {latest_step}")
        return None, None, None, None

    print(f"Loading rolling checkpoint from step {latest_step}")
    
    with open(agent_path, 'rb') as f:
        agent_state = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent_structure, agent_state)
    
    with open(buffer_path, 'rb') as f:
        replay_buffer = pickle.load(f)
    
    online_loop_step = 0
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            online_loop_step = meta.get('online_loop_step', 0)
        
    return agent, replay_buffer, latest_step, online_loop_step


# Import authority manager for clean API
from envs.hardware.authority_manager import HardwareAuthorityManager

# Alias for backward compatibility
is_wx200_env = HardwareAuthorityManager.is_wx200_env

def check_train_authority(env, context=""):
    """Helper function to verify training environment HAS hardware authority."""
    # Use authority manager which works with any WX200 wrapper version
    HardwareAuthorityManager.check_train_authority(env, context=context)

def main(_):
    # Global reference for signal handler to access environments
    global_envs = {'env': None, 'eval_env': None}
    
    # Global interrupt flag for graceful interrupt handling
    _interrupt_requested = False
    _interrupt_count = 0
    
    # Setup signal handler for Ctrl+C
    def signal_handler(sig, frame):
        nonlocal _interrupt_requested, _interrupt_count
        _interrupt_count += 1
        
        if _interrupt_count == 1:
            # First Ctrl+C: Emergency stop and set interrupt flag
            print("\n\n" + "="*60)
            print("⚠️  Ctrl+C detected! Emergency stopping robot...")
            print("="*60)
            
            _interrupt_requested = True
            
            # Emergency stop robot if it's a hardware environment
            if is_wx200_env(FLAGS.env_name):
                try:
                    # Emergency stop training env
                    if global_envs['env'] is not None:
                        try:
                            HardwareAuthorityManager.emergency_stop(global_envs['env'])
                        except Exception as e:
                            print(f"⚠️  Error during emergency stop (train env): {e}")
                    
                    # Emergency stop eval env
                    if global_envs['eval_env'] is not None:
                        try:
                            HardwareAuthorityManager.emergency_stop(global_envs['eval_env'])
                        except Exception as e:
                            print(f"⚠️  Error during emergency stop (eval env): {e}")
                except Exception as e:
                    print(f"⚠️  Error during emergency stop: {e}")
                    import traceback
                    traceback.print_exc()
            
            print("⚠️  Interrupt flag set. Current operation will be interrupted.")
            print("⚠️  Press Ctrl+C again to fully shutdown and exit.")
            print("="*60 + "\n")
        else:
            # Second Ctrl+C: Full shutdown and exit
            print("\n\n" + "="*60)
            print("⚠️  Second Ctrl+C detected! Executing full robot shutdown...")
            print("="*60)
            
            # Shutdown robot if it's a hardware environment
            if is_wx200_env(FLAGS.env_name):
                try:
                    # Shutdown Training Env
                    if global_envs['env'] is not None:
                        print("Shutting down training environment...")
                        try:
                            global_envs['env'].close()
                            print("✓ Training environment shutdown completed")
                        except Exception as e:
                            print(f"⚠️  Error shutting down training env: {e}")
                            import traceback
                            traceback.print_exc()

                    # Shutdown Eval Env
                    if global_envs['eval_env'] is not None:
                        print("Shutting down eval environment...")
                        try:
                            global_envs['eval_env'].close()
                            print("✓ Eval environment shutdown completed")
                        except Exception as e:
                            print(f"⚠️  Error shutting down eval env: {e}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"⚠️  Error during robot shutdown: {e}")
                    import traceback
                    traceback.print_exc()
            
            print("Exiting...")
            sys.exit(0)
    
    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    
    if FLAGS.wandb_name=="test":
        exp_name = get_exp_name(FLAGS.seed)
    else:
        exp_name = FLAGS.wandb_name + "seed" + str(FLAGS.seed)
    if FLAGS.log:
        setup_wandb(project=FLAGS.project, group=FLAGS.run_group, name=exp_name, mode=os.environ.get("WANDB_MODE", "online"))
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        flag_dict = get_flag_dict()
        with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
            json.dump(flag_dict, f)

    config = FLAGS.agent

    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        # Disable action clipping for WX200 (no normalization)
        action_clip = None if is_wx200_env(FLAGS.env_name) else 1e-5
        
        # Reset class-level authority BEFORE creating environments (in case of leftover state from previous run)
        if HardwareAuthorityManager.is_wx200_env(FLAGS.env_name):
            HardwareAuthorityManager.reset_authority()
        
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, action_clip_eps=action_clip)
    
    # Store environment references for signal handler
    global_envs['env'] = env
    global_envs['eval_env'] = eval_env
    
    # Verify eval environment does NOT have authority (for WX200 environments)
    # Note: Training env may have already claimed authority during dataset loading, which is fine
    if HardwareAuthorityManager.is_wx200_env(FLAGS.env_name):
        print("\n" + "="*60)
        print("Verifying eval environment has no hardware authority...")
        print("="*60)
        HardwareAuthorityManager.check_eval_no_authority(eval_env)
        print("="*60 + "\n")
        
    # Removed parallel eval envs - real world is single-threaded only
        
    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    # Create numpy RNG for datasets (separate from global seed for reproducibility)
    dataset_rng = np.random.default_rng(FLAGS.seed + 20000)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = FLAGS.ep_resume
    eval_step = 0

    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length
    config["discount"] = FLAGS.discount
    config['best_of_n'] = FLAGS.best_of_n
    
    # spesify min max reward
    print(is_robomimic_env(FLAGS.env_name))
    if is_robomimic_env(FLAGS.env_name):
        config['q_min'] = -1.0 / (1 - config['discount'])
        config['q_max'] = 0.0 / (1 - config['discount'])
    elif FLAGS.env_name.startswith('PutBallintoBowl'):
        config['q_min'] = 0.0 / (1 - config['discount'])
        config['q_max'] = 1.0 / (1 - config['discount'])
    elif FLAGS.env_name.startswith('cube-triple'):
        config['q_min'] = -3.0 / (1 - config['discount'])
        config['q_max'] = 0.0 / (1 - config['discount'])
    elif FLAGS.env_name.startswith('D4RL/kitchen'):
        config['q_min'] = 0 / (1 - config['discount'])
        config['q_max'] = 4 / (1 - config['discount'])
    elif FLAGS.env_name in GYMNASIUM_ROBOTICS_ADROITHAND_ENVS:
        config['q_min'] = -50
        config['q_max'] = 10
    elif FLAGS.env_name.startswith('stacking') or FLAGS.env_name.startswith('gate_insertion'):
        config['q_min'] = -100
        config['q_max'] = 100
    elif is_wx200_env(FLAGS.env_name):
        # WX200 hardware: sparse reward, -1 per step, 0 on success
        config['q_min'] = -500.0 / (1 - config['discount'])  # Max episode length is 500
        config['q_max'] = 0.0 / (1 - config['discount'])
    else:
        raise ValueError(f"Unknown environment: {FLAGS.env_name}, please specify min max reward")

    # handle dataset
    def process_train_dataset(ds, numpy_rng):
        """
        Process the train dataset to
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(numpy_rng=numpy_rng, **ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                numpy_rng=numpy_rng,
                **{k: v[:new_size] for k, v in ds.items()}
            )

        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)

        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)

        return ds

    train_dataset = process_train_dataset(train_dataset, dataset_rng)
    train_dataset.p_aug = FLAGS.p_aug
    mc_returns = train_dataset.compute_mc_returns(FLAGS.discount)
    train_dataset_dict = {k: v for k, v in train_dataset.items()}
    train_dataset_dict['mc_returns'] = mc_returns
    train_dataset = Dataset.create(numpy_rng=dataset_rng, **train_dataset_dict)
    train_dataset.p_aug = FLAGS.p_aug
    
    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)

    # Try to load checkpoint
    try:
        resumed_agent, resumed_buffer, resumed_step, resumed_online_step = load_rolling_checkpoint(FLAGS.seed, FLAGS.env_name, agent)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        resumed_agent, resumed_buffer, resumed_step, resumed_online_step = None, None, None, None
    is_resumed = resumed_agent is not None
    if is_resumed:
        agent = resumed_agent
        log_step = resumed_step
        print(f"Resumed training from step {log_step} (online_step={resumed_online_step})")
    else:
        print("No checkpoint found, starting from scratch")

    # Setup logging.
    prefixes = ["eval", "env", "eval_sde", "step_data"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")


    logger = LoggingHelper(csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))for prefix in prefixes}, wandb_logger=wandb,)
    
    if not is_resumed:
        if (FLAGS.restore_actor_path is not None) and (FLAGS.restore_actor_path != "None"):
            agent.restore_actor_params(FLAGS.restore_actor_path)
        if (FLAGS.restore_critic_path is not None) and (FLAGS.restore_critic_path != "None"):
            agent.restore_critic_params(FLAGS.restore_critic_path)

        # BC
        for i in tqdm.tqdm(range(log_step, FLAGS.offline_steps + log_step), initial=log_step, total=FLAGS.offline_steps + log_step, dynamic_ncols=True):
            log_step += 1

            if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
                train_dataset, val_dataset = make_ogbench_env_and_datasets(FLAGS.env_name, dataset_path=dataset_paths[dataset_idx],
                                                                                compact_dataset=False, dataset_only=True, cur_env=env,)
                train_dataset = process_train_dataset(train_dataset, dataset_rng)

            batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length,discount=discount)
            agent, offline_info = agent.update(batch)

            if i % FLAGS.step_data_log_interval == 0:
                logger.log({
                    'phase': 'bc',
                    'batch_obs_mean': float(batch['observations'].mean()),
                    'batch_obs_std': float(batch['observations'].std()),
                    'batch_action_mean': float(batch['actions'].mean()),
                    'batch_action_std': float(batch['actions'].std()),
                    'reward': 0.0,  # Not applicable in BC phase
                    'done': 0.0,    # Not applicable in BC phase
                }, "step_data", step=log_step)

            if i % FLAGS.log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)

            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                save_agent(agent, FLAGS.save_dir, log_step)

            # eval
            if i == FLAGS.offline_steps - 1 or \
                (FLAGS.eval_interval_bc != 0 and i > 0 and i % (FLAGS.eval_interval_bc) == 0):
                # during eval, the action chunk is executed fully
                eval_step += 1
                
                # Release authority from training env before eval (for WX200 environments)
                if is_wx200_env(FLAGS.env_name):
                    HardwareAuthorityManager.release_train_authority_for_eval(env, step=log_step)
                
                eval_info, _, _ = evaluate(
                    agent=agent, 
                    env=eval_env, 
                    global_step=log_step, 
                    action_dim=action_dim,
                    num_eval_episodes=FLAGS.eval_episodes, 
                    num_video_episodes=FLAGS.video_episodes, 
                    video_frame_skip=FLAGS.video_frame_skip,
                    actor_fn=agent.compute_flow_actions, 
                    env_name=FLAGS.env_name, 
                    plot=(eval_step%2==0) and FLAGS.plot_q_vs_mc,
                    save_observations_path="live_observations.pkl"
                )
                
                # Release authority from eval env after eval (for WX200 environments)
                if is_wx200_env(FLAGS.env_name):
                    HardwareAuthorityManager.release_eval_authority_after_eval(eval_env, step=log_step)
                print(eval_info)
                logger.log(eval_info, "eval", step=log_step)
                if FLAGS.clip_bc and (eval_info['success'] >= 0.45):
                    save_agent(agent, FLAGS.save_dir, log_step)
                    print("breaking due to bc clipping")
                    break
        
        # Offline RL (CalQL)
        if FLAGS.calql_steps > 0:
            for i in tqdm.tqdm(range(log_step, FLAGS.calql_steps + log_step), initial=log_step, total=FLAGS.calql_steps + log_step, dynamic_ncols=True):
                log_step += 1
                if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
                    dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                    print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
                    train_dataset, val_dataset = make_ogbench_env_and_datasets(
                        FLAGS.env_name,
                        dataset_path=dataset_paths[dataset_idx],
                        compact_dataset=False,
                        dataset_only=True,
                        cur_env=env,
                    )
                    train_dataset = process_train_dataset(train_dataset, dataset_rng)

                batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length,
                                                    discount=discount, ret_next_act=True, ret_mc=True)
                agent, offline_info = agent.calql_update(batch)

                if i % FLAGS.step_data_log_interval == 0:
                    logger.log({
                        'phase': 'calql',
                        'batch_obs_mean': float(batch['observations'].mean()),
                        'batch_obs_std': float(batch['observations'].std()),
                        'batch_action_mean': float(batch['actions'].mean()),
                        'batch_action_std': float(batch['actions'].std()),
                        'reward': 0.0,  # Not applicable in CalQL phase
                        'done': 0.0,    # Not applicable in CalQL phase
                    }, "step_data", step=log_step)

                if i % (FLAGS.log_interval // 5) == 0:
                    logger.log(offline_info, "offline_agent", step=log_step)

                # eval
                if i == FLAGS.calql_steps - 1 or (FLAGS.eval_interval_bc != 0 and i > 0 and i % FLAGS.eval_interval_bc == 0):
                    # during eval, the action chunk is executed fully
                    eval_step += 1
                    
                    # Release authority from training env before eval (for WX200 environments)
                    if is_wx200_env(FLAGS.env_name):
                        HardwareAuthorityManager.release_train_authority_for_eval(env, step=log_step)
                    
                    eval_info, _, _ = evaluate(agent=agent, env=eval_env, global_step=log_step, action_dim=action_dim,
                        num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=FLAGS.video_episodes, video_frame_skip=FLAGS.video_frame_skip,
                        actor_fn=agent.compute_flow_actions, env_name=FLAGS.env_name, plot=(eval_step%2==0) and FLAGS.plot_q_vs_mc)
                    
                    # Release authority from eval env after eval (for WX200 environments)
                    if is_wx200_env(FLAGS.env_name):
                        HardwareAuthorityManager.release_eval_authority_after_eval(eval_env, step=log_step)
                    print(eval_info)
                    logger.log(eval_info, "eval", step=log_step)
                
        
        save_agent(agent, FLAGS.save_dir, log_step)
    # **UNDO COmment
    # eval after offline training
    # print("Evaluating with ODE")
    # eval_info, _, _ = evaluate(
    #     agent=agent,
    #     env=eval_env,
    #     global_step=log_step,
    #     action_dim=example_batch["actions"].shape[-1],
    #     num_eval_episodes=FLAGS.eval_episodes,
    #     video_frame_skip=FLAGS.video_frame_skip,
    #     actor_fn=agent.compute_flow_actions, 
    # )
    # print("Success Rate: ", eval_info["success"].item())
    # logger.log(eval_info, "eval", step=log_step)
    
    # print("Evaluating with Stochastic Flow")
    # eval_info, _, _ = evaluate(
    #     agent=agent,
    #     env=eval_env,
    #     global_step=log_step,
    #     action_dim=example_batch["actions"].shape[-1],
    #     num_eval_episodes=FLAGS.eval_episodes,
    #     video_frame_skip=FLAGS.video_frame_skip,
    # )
    # print("Success Rate: ", eval_info["success"].item())
    # logger.log(eval_info, "eval_sde", step=log_step)

    # if FLAGS.offline_ratio >= 0.0:
    #     # is we use ratio mixing, we seperate the replay buffers for offline and online data
    # else:
    #     replay_buffer = ReplayBuffer.create_from_initial_dataset(
    #         dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    #     )
    
    example_transition = dict(
        observations=example_batch['observations'],
        actions=example_batch['actions'],
        rewards=np.array(0.0),
        terminals=np.array(0.0),
        masks=np.array(1.0),
        next_observations=example_batch['observations'],
    )
    if is_resumed:
        replay_buffer = resumed_buffer
        replay_buffer.p_aug = FLAGS.p_aug
    else:
        # Create separate RNG for replay buffer sampling
        replay_buffer_rng = np.random.default_rng(FLAGS.seed + 30000)
        replay_buffer = ReplayBuffer.create(example_transition, FLAGS.buffer_size, numpy_rng=replay_buffer_rng)
        replay_buffer.p_aug = FLAGS.p_aug
    
    # Training env will claim authority on first reset if hardware is available
    try:
        ob, _ = env.reset(seed=FLAGS.seed + 10000)
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt during initial env.reset(). Emergency stopping...")
        HardwareAuthorityManager.emergency_stop(env)
        _interrupt_requested = False  # Reset flag
        print("⚠️  Exiting...")
        sys.exit(1)
    
    # Verify authority was claimed after reset (for WX200 environments)
    if is_wx200_env(FLAGS.env_name):
        check_train_authority(env, context="after first reset")
        print("✓ Training environment authority verified after reset")
    trajectory_buffer = []
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]
    
    # reset actor optimizer
    if not is_resumed:
        agent = agent.reset_optimizers_with_lr()
        print(f"Reset actor optimizer with learning rate {config['ppo_lr']}")
        
    # Online RL
    update_info = {}
    episode_counter = 0  # Track episode resets for deterministic seeding
    start_online_step = 1
    if is_resumed:
        if resumed_online_step > 0:
            start_online_step = resumed_online_step + 1
        else:
             # Fallback if online step wasn't saved (legacy checkpoint)
             baseline = FLAGS.ep_resume + FLAGS.offline_steps + FLAGS.calql_steps
             start_online_step = log_step - baseline + 1
        print(f"Resuming online loop at i={start_online_step}")
        
    for i in tqdm.tqdm(range(start_online_step, FLAGS.online_steps + 1), initial=start_online_step, total=FLAGS.online_steps, dynamic_ncols=True):
        # Check for interrupt flag
        if _interrupt_requested:
            print("\n⚠️  Interrupt requested. Stopping current training step...")
            HardwareAuthorityManager.emergency_stop(env)
            _interrupt_requested = False  # Reset flag
            print("⚠️  Skipping current step and continuing training...")
            continue
        
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            if FLAGS.best_of_n > 1:
                # ** TODO: add best of n argument functionality somehow
                action = agent.sample_actions_BON(observations=ob, rng=key)
            else:
                action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)

        # Verify training env has authority before step (for WX200 environments)
        # Only check periodically to avoid excessive logging
        if is_wx200_env(FLAGS.env_name) and i % 1000 == 0:
            check_train_authority(env, context="for online training steps")
        
        try:
            next_ob, int_reward, terminated, truncated, info = env.step(action)
            if is_wx200_env(FLAGS.env_name):
                env.render()
            done = terminated or truncated
        except KeyboardInterrupt:
            print("\n⚠️  KeyboardInterrupt during env.step(). Emergency stopping...")
            HardwareAuthorityManager.emergency_stop(env)
            _interrupt_requested = False  # Reset flag
            print("⚠️  Skipping current step and continuing training...")
            continue

        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        trajectory_buffer.append(transition)

        if i % FLAGS.step_data_log_interval == 0:
            try:
                logger.log({
                    'phase': 'online',
                    'batch_obs_mean': float(np.mean(ob)),
                    'batch_obs_std': float(np.std(ob)),
                    'batch_action_mean': float(np.mean(action)),
                    'batch_action_std': float(np.std(action)),
                    'reward': float(int_reward),
                    'done': float(done),
                }, "step_data", step=log_step)
            except Exception as e:
                print(f"Warning: Failed to log step data at step {log_step}: {e}")

        # done
        if done:
            is_success = float(info.get('success', 0.0))
            score = is_success
            for traj_transition in trajectory_buffer:
                traj_transition['is_success'] = is_success
                traj_transition['score'] = score if score !=0 else 1e-9
                replay_buffer.add_transition(traj_transition)

            trajectory_buffer = []
            episode_counter += 1
            
            # Training env should already have authority, but verify
            if is_wx200_env(FLAGS.env_name):
                check_train_authority(env, context="before episode reset")
            
            try:
                ob, _ = env.reset(seed=FLAGS.seed + 10000 + episode_counter)
                action_queue = []  # reset the action queue
            except KeyboardInterrupt:
                print("\n⚠️  KeyboardInterrupt during env.reset(). Emergency stopping...")
                HardwareAuthorityManager.emergency_stop(env)
                _interrupt_requested = False  # Reset flag
                print("⚠️  Skipping episode reset and continuing training...")
                # Try to continue with current observation
                continue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            if i < FLAGS.q_warmup_steps:
                utd_ratio = FLAGS.utd_warmup
            else:
                utd_ratio = FLAGS.utd_online
            
            if FLAGS.offline_ratio >= 0.0:
                # Mixed training with specified offline_ratio
                offline_batch_size = int(config['batch_size'] * FLAGS.offline_ratio) # this would be zero
                online_batch_size = config['batch_size'] - offline_batch_size
                
                if offline_batch_size > 0:
                    offline_batch = train_dataset.sample_sequence(offline_batch_size * utd_ratio,
                                sequence_length=FLAGS.horizon_length, discount=discount)
                    offline_batch = jax.tree.map(lambda x: x.reshape((
                        utd_ratio, offline_batch_size) + x.shape[1:]), offline_batch)
                
                if online_batch_size > 0:
                    online_batch = replay_buffer.sample_sequence(online_batch_size * utd_ratio,
                                sequence_length=FLAGS.horizon_length, discount=discount)
                    online_batch = jax.tree.map(lambda x: x.reshape((
                        utd_ratio, online_batch_size) + x.shape[1:]), online_batch)
                
                # Combine batches
                if offline_batch_size > 0 and online_batch_size > 0:
                    batch = jax.tree.map(lambda off, on: np.concatenate([off, on], axis=1), 
                                       offline_batch, online_batch)
                elif offline_batch_size > 0:
                    batch = offline_batch # this is zero
                else:
                    batch = online_batch
                batch_bc = None
                batch_success = None
                success_flag = False
                if FLAGS.use_success_buffer:
                    num_successful = replay_buffer._traj_success_mask[:replay_buffer.size].sum()
                    min_required = FLAGS.success_buffer_batch_size * utd_ratio * FLAGS.horizon_length
                    if num_successful >= min_required:
                        try:
                            batch_success = create_success_buffer_batch(
                                train_dataset, 
                                replay_buffer, 
                                FLAGS.success_buffer_batch_size*utd_ratio,
                                offline_ratio=0,
                                seq_len=FLAGS.horizon_length,
                                discount=discount,
                            )
                            batch_success = jax.tree.map(
                                lambda x: x.reshape((utd_ratio, FLAGS.success_buffer_batch_size)+x.shape[1:]),
                                batch_success
                            )
                            success_flag = True
                        except ValueError as e:
                            print(f"Failed to sample success buffer: {e}")
                            success_flag = False
       
            else:
                batch = replay_buffer.sample_sequence(config['batch_size'] * utd_ratio,sequence_length=FLAGS.horizon_length, discount=discount)
                batch = jax.tree.map(lambda x: x.reshape((utd_ratio, config['batch_size']) + x.shape[1:]), batch)
                batch_bc = None
                batch_success = None
                success_flag = False
                # We don't want to do CalQL during online finetuning for now
                # batch_bc = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount, ret_next_act=True, ret_mc=True)
                # batch_bc = jax.tree.map(lambda x: x.reshape((utd_ratio, config['batch_size']) + x.shape[1:]), batch_bc)
                if FLAGS.use_success_buffer:
                    batch_success = create_success_buffer_batch(
                        train_dataset, 
                        replay_buffer, 
                        FLAGS.success_buffer_batch_size*utd_ratio,
                        offline_ratio=0,
                        seq_len=FLAGS.horizon_length,
                        discount=discount,
                    )
                    batch_success = jax.tree.map(
                          lambda x: x.reshape((utd_ratio, FLAGS.success_buffer_batch_size)+x.shape[1:]),
                          batch_success
                    )
                    success_flag = True

            if i < FLAGS.q_warmup_steps:
                agent, update_info["online_agent"] = agent.batch_q_warmup_update(batch, batch_bc)
            else:
                if batch_success is None:
                    batch_success = batch 
                    success_flag = False
                agent, update_info["online_agent"] = agent.batch_update(batch, batch_bc, batch_success, success_flag)

        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_step += 1
            
            # Release authority from training env before eval (for WX200 environments)
            if HardwareAuthorityManager.is_wx200_env(FLAGS.env_name):
                HardwareAuthorityManager.release_train_authority_for_eval(env, step=log_step)
            
            print("Evaluating with ODE")
            eval_info, _, _ = evaluate(agent=agent, env=eval_env, global_step=log_step, action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=FLAGS.video_episodes, video_frame_skip=FLAGS.video_frame_skip,
                actor_fn=agent.compute_flow_actions, env_name=FLAGS.env_name, plot=(eval_step%2==0) and FLAGS.plot_q_vs_mc)
            print("Success Rate: ", eval_info.get("success", 0.0))
            logger.log(eval_info, "eval", step=log_step)

            # Eval env should still have authority for SDE eval (no need to release/reclaim between ODE and SDE)
            
            print("Evaluating with Stochastic Flow")
            eval_info, _, _ = evaluate(agent=agent, env=eval_env, global_step=log_step, action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=FLAGS.video_episodes, video_frame_skip=FLAGS.video_frame_skip,
                env_name=FLAGS.env_name, plot=(eval_step%2==0) and FLAGS.plot_q_vs_mc)
            print("Success Rate: ", eval_info.get("success", 0.0))
            
            # Release authority from eval env after all evals (for WX200 environments)
            if HardwareAuthorityManager.is_wx200_env(FLAGS.env_name):
                HardwareAuthorityManager.release_eval_authority_after_eval(eval_env, step=log_step)
            logger.log(eval_info, "eval_sde", step=log_step)

        # saving
        if FLAGS.log and FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)
            
        # Preemption saving
        if FLAGS.preemption_backup_interval > 0 and i % FLAGS.preemption_backup_interval == 0:
            print(f"Saving rolling checkpoint at step {log_step}")
            save_rolling_checkpoint(agent, replay_buffer, log_step, i, FLAGS.seed, FLAGS.env_name)

    # Final Save
    save_agent(agent, FLAGS.save_dir, log_step)
    # Delete the preemption backup
    if FLAGS.preemption_backup_interval > 0:
        print(f"Deleting preemption backup at step {log_step}")
        backup_path = os.path.join(FLAGS.save_dir, f"preemption_backup_{FLAGS.seed}_{FLAGS.env_name}")
        if os.path.exists(backup_path):
            if os.path.isfile(backup_path):
                os.remove(backup_path)
            else:
                shutil.rmtree(backup_path)
    if FLAGS.log:
        for key, csv_logger in logger.csv_loggers.items():
            csv_logger.close()
    
    # Cleanup: shutdown robot if hardware environment
    if is_wx200_env(FLAGS.env_name):
        try:
            print("\nShutting down robot...")
            if global_envs['env'] is not None and hasattr(global_envs['env'], 'close'):
                global_envs['env'].close()
            if global_envs['eval_env'] is not None and hasattr(global_envs['eval_env'], 'close'):
                global_envs['eval_env'].close()
            print("✓ Robot shutdown complete")
        except Exception as e:
            print(f"⚠️  Warning: Error during robot cleanup: {e}")

if __name__ == '__main__':
    try:
        app.run(main)
    except KeyboardInterrupt:
        # Additional safety net for KeyboardInterrupt
        print("\n⚠️  KeyboardInterrupt caught at top level")
        # Signal handler should have already handled shutdown, but try again just in case
        sys.exit(0)