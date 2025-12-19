from collections import defaultdict
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import jax
import numpy as np
import tqdm
from functools import partial
import cv2
import sys
from envs.hardware.authority_manager import HardwareAuthorityManager

def check_interrupt_flag():
    """Check if interrupt flag is set in main_ogpo_real module."""
    try:
        if 'main_ogpo_real' in sys.modules:
            main_module = sys.modules['main_ogpo_real']
            if hasattr(main_module, '_interrupt_requested') and main_module._interrupt_requested:
                return True
    except (AttributeError, KeyError):
        pass
    return False


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper to split RNG before each call to f(rng=...)."""
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)
    return wrapped


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_async(d, idx, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v[idx]))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def discounted_returns_from_prefix(rewards, indices, discount=0.99):
    T = len(rewards)
    disc_pows = discount ** np.arange(T)                  # [1, γ, γ^2, ...]
    prefix = np.cumsum((np.array(rewards) - 1.0) * disc_pows)           # S_t = Σ_{k=0}^t γ^k r_k

    def G(t):
        total = prefix[-1]
        prev  = 0.0 if t == 0 else prefix[t-1]
        # G_t = (Σ_{k=t}^{T-1} γ^k r_k) / γ^t = (total - prev) / γ^t
        return float((total - prev) / disc_pows[t])

    return [G(t) for t in indices]


def visualize_q_accuracy(time_series_data, scatter_data, global_step, suffix, dir_suffix, save_dir='./plots'):
    if not scatter_data:
        return

    save_dir = os.path.join(save_dir, 'q_accuracy_plots', dir_suffix)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)
    reward_desc = "MC return-to-go (episode end)"
    fig.suptitle(f'Q-Function Accuracy Analysis ({reward_desc})\nStep {global_step} ({suffix})', fontsize=12)

    # Subplot 1: per-episode time series
    ax1 = axes[0]
    for i, episode in enumerate(time_series_data):
        timesteps = np.arange(len(episode['q_preds']))
        color = plt.cm.plasma(i / max(1, len(time_series_data) - 1))
        if len(timesteps) == 0:
            continue
        ax1.plot(timesteps, episode['q_preds'], marker='s', linestyle='--',
                 markersize=4, color=color, alpha=0.5, label='Q-Pred' if i == 0 else None)
        ax1.plot(timesteps, episode['mc_returns'], marker='^', linestyle='-',
                 markersize=4, color=color, alpha=0.5, label='MC Return' if i == 0 else None)

    ax1.set_title('Q-Value vs MC Return at Decision Points')
    ax1.set_xlabel('Decision Point Index (per episode)')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: scatter correlation
    ax2 = axes[1]
    q_preds = [d[0] for d in scatter_data]
    mc_returns = [d[1] for d in scatter_data]

    ax2.scatter(q_preds, mc_returns, alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    if q_preds and mc_returns:
        min_val = min(min(q_preds), min(mc_returns))
        max_val = max(max(q_preds), max(mc_returns))
        ax2.plot([min_val, max_val], [min_val, max_val], '--', alpha=0.75, linewidth=2, label='y = x')

    ax2.set_title('Predicted Q vs MC Return (All Episodes)')
    ax2.set_xlabel('Predicted Q-Value')
    ax2.set_ylabel('MC Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')

    if len(q_preds) > 1:
        correlation = np.corrcoef(q_preds, mc_returns)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    save_path = os.path.join(save_dir, f'q_accuracy_step_{global_step}_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Q-function accuracy plot saved to: {save_path}")

def evaluate(
    agent,
    env,
    global_step,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
    actor_fn=None,
    env_name='',
    plot=False,
    save_observations_path=None,  # Optional: path to save observations for distribution comparison
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        actor_fn: Optional custom actor function. If None, uses agent.sample_actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if actor_fn is None:
        suffix = "SDE"
        actor_fn = agent.sample_actions
    else:
        suffix = "ODE"
   
    actor_fn = supply_rng(actor_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    renders = []
    
    scatter_data = []   # list of (q_pred, mc_return) across all episodes
    time_series_data = []  # per-episode lists
    
    # Check if this is a hardware environment (wx200)
    is_hardware_env = HardwareAuthorityManager.is_wx200_env(env_name)
    
    # Collect observations if saving enabled
    all_observations = []
    
    interrupted = False
    for i in tqdm.tqdm(range(num_eval_episodes + num_video_episodes), dynamic_ncols=True):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        try:
            observation, info = env.reset()
            if save_observations_path is not None:
                all_observations.append(observation.copy())
        except KeyboardInterrupt:
            print("\n⚠️  KeyboardInterrupt during env.reset(). Emergency stopping...")
            HardwareAuthorityManager.emergency_stop(env)
            interrupted = True
            break
        
        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)
        
        rewards = []                     # per-step rewards
        decision_indices = []            # step indices where a new chunk was decided
        q_preds_at_decisions = []        # Q(s_t, a_t) at each decision point
        action_chunk_lens = defaultdict(int)

        action_queue = []
        policy_call_count = 0
        action_execution_count = 0
        policy_call_steps = []
        
        try:
            while not done:
                # Check for interrupt flag (from main_ogpo_real) before continuing
                if check_interrupt_flag():
                    print("\n⚠️  Interrupt detected. Stopping evaluation episode...")
                    HardwareAuthorityManager.emergency_stop(env)
                    done = True
                    terminated = True
                    truncated = False
                    info['success'] = 0.0
                    info['interrupted'] = True
                    break
                
                if len(action_queue) == 0:
                    # Policy called - record this
                    policy_call_count += 1
                    policy_call_steps.append(step)
                    # print(f"[Episode {i+1}] Policy call #{policy_call_count} at step {step} (interval: {step - policy_call_steps[-2] if len(policy_call_steps) > 1 else step})")
                    
                    action = actor_fn(observations=observation) #, temperature=eval_temperature)
                    action = np.array(action).reshape(-1, action_dim)
                    action_chunk_len = action.shape[0]
                    # print(f"  -> Policy output: {action_chunk_len} actions (shape: {action.shape})")
                    for a in action:
                        action_queue.append(a)
                        
                    q_pred = agent.get_q_values(jax.device_put(observation[np.newaxis, ...]), 
                                                jax.device_put(action[np.newaxis, ...]))[0]
                    q_preds_at_decisions.append(float(q_pred))
                    decision_indices.append(step)
                    action_chunk_lens[f"action_chunk_length_{action_chunk_len}"] += 1
                    info['action_chunk_length'] = action_chunk_lens
                
                action = action_queue.pop(0)
                action_execution_count += 1
                if action_execution_count % 10 == 0 or len(action_queue) == 0:
                    pass
                    # print(f"  [Step {step}] Executing action #{action_execution_count} (queue size: {len(action_queue)})")
                
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)

                # For WX200, do not clip actions (no normalization). For others, clip to [-1, 1].
                if is_hardware_env:
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    
                    # Check for interrupt flag AFTER step (to avoid printing after interrupt)
                    if check_interrupt_flag():
                        print("\n⚠️  Interrupt detected. Stopping evaluation episode...")
                        HardwareAuthorityManager.emergency_stop(env)
                        done = True
                        terminated = True
                        truncated = False
                        info['success'] = 0.0
                        info['interrupted'] = True
                        break
                    
                    # Only print if not interrupted
                    # Check ArUco marker visibility and print observation (red if not all visible)
                    all_markers_visible = True
                    try:
                        # Unwrap to get base environment using authority manager's method
                        base_env = HardwareAuthorityManager.unwrap_to_base(env)
                        # Check if it's a WX200 environment and has visibility info
                        if base_env is not None and hasattr(base_env, 'aruco_obs_dict') and 'aruco_visibility' in base_env.aruco_obs_dict:
                            visibility = base_env.aruco_obs_dict['aruco_visibility']
                            all_markers_visible = np.all(visibility == 1.0)
                    except:
                        pass  # If we can't check visibility, assume all visible
                    
                    # Denormalize action for display (show actual control values, not normalized [-1, 1])
                    action_display = action.copy()  # Default to normalized action
                    try:
                        if base_env is not None and hasattr(base_env, '_denormalize_action'):
                            # Denormalize to show actual control values
                            # For position targets: returns (target_position, target_orientation_quat_wxyz, gripper_target)
                            # For velocity deltas: returns (velocity_world, angular_velocity_world, gripper_target)
                            denorm_result = base_env._denormalize_action(action)
                            if len(denorm_result) == 3:
                                # Check if this is position targets (first element is 3D position array) or velocity (first element is 3D velocity array)
                                if len(denorm_result[0]) == 3:
                                    # Could be either - check if it's position targets by looking at the action space
                                    # Position targets: action is [x, y, z, qw, qx, qy, qz, gripper] (8D)
                                    # Velocity deltas: action is [vx, vy, vz, wx, wy, wz, gripper] (7D)
                                    if len(action) == 8:
                                        # Position targets
                                        target_position, target_orientation_quat_wxyz, gripper_target = denorm_result
                                        action_display = np.concatenate([
                                            target_position,           # [x, y, z] in m
                                            target_orientation_quat_wxyz,  # [qw, qx, qy, qz]
                                            [gripper_target]           # gripper in meters
                                        ])
                                    else:
                                        # Velocity deltas (7D)
                                        velocity_world, angular_velocity_world, gripper_target = denorm_result
                                        action_display = np.concatenate([
                                            velocity_world,            # [vx, vy, vz] in m/s
                                            angular_velocity_world,    # [wx, wy, wz] in rad/s
                                            [gripper_target]           # gripper in meters
                                        ])
                                else:
                                    # Fallback: assume velocity-based
                                    velocity_world, angular_velocity_world, gripper_target = denorm_result
                                    action_display = np.concatenate([
                                        velocity_world,
                                        angular_velocity_world,
                                        [gripper_target]
                                    ])
                    except Exception as e:
                        # If denormalization fails, show normalized action
                        pass
                    
                    # Check interrupt flag ONE MORE TIME right before actual print (last chance to stop)
                    if check_interrupt_flag():
                        print("\n⚠️  Interrupt detected. Stopping evaluation episode...")
                        HardwareAuthorityManager.emergency_stop(env)
                        done = True
                        terminated = True
                        truncated = False
                        info['success'] = 0.0
                        info['interrupted'] = True
                        break
                    
                    # Final interrupt check - if set, skip ALL printing
                    if check_interrupt_flag():
                        print("\n⚠️  Interrupt detected. Stopping evaluation episode...")
                        HardwareAuthorityManager.emergency_stop(env)
                        done = True
                        terminated = True
                        truncated = False
                        info['success'] = 0.0
                        info['interrupted'] = True
                        break
                    
                    # Print observation and action (red if not all markers visible)
                    # Only print if interrupt flag is NOT set
                    if not check_interrupt_flag():
                        RED = '\033[91m'
                        RESET = '\033[0m'
                        if all_markers_visible:
                            print(f"[Episode {i+1}, Step {step}] Observation: {next_observation}")
                            # Check again before second print
                            if not check_interrupt_flag():
                                # Print normalized action from network AND denormalized
                                print(f"[Episode {i+1}, Step {step}] Action (normalized from network): {action}")
                                print(f"[Episode {i+1}, Step {step}] Action (denormalized): {action_display}")
                        else:
                            visibility_str = f"Visibility: {base_env.aruco_obs_dict.get('aruco_visibility', 'unknown')}" if hasattr(base_env, 'aruco_obs_dict') else ""
                            print(f"{RED}[Episode {i+1}, Step {step}] Observation: {next_observation} {visibility_str}{RESET}")
                            # Check again before second print
                            if not check_interrupt_flag():
                                # Print normalized action from network AND denormalized
                                print(f"{RED}[Episode {i+1}, Step {step}] Action (normalized from network): {action}{RESET}")
                                print(f"{RED}[Episode {i+1}, Step {step}] Action (denormalized): {action_display}{RESET}")
                    
                    # Render to show ArUco tracking and capture E-Stop key
                    env.render()
                    
                    # Check for E-stop and terminate episode if detected
                    if info.get("estop", False):
                        print("Episode terminated by E-Stop.")
                        done = True
                        terminated = True
                        truncated = False
                        # Mark as interrupted/failed
                        info['success'] = 0.0
                        info['interrupted'] = True
                    else:
                        rewards.append(float(reward))
                        done = terminated or truncated
                else:
                    next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
                    rewards.append(float(reward))
                    done = terminated or truncated
                step += 1

                # For hardware environments, always render to show camera feed
                if is_hardware_env or (should_render and (step % video_frame_skip == 0 or done)):
                    frame = env.render()
                    if frame is not None and should_render:
                        render.append(frame.copy())

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
            
            # Collect observation for distribution comparison
            if save_observations_path is not None:
                all_observations.append(observation.copy())
        except KeyboardInterrupt:
            print(f"\n⚠️  KeyboardInterrupt during episode {i+1}. Emergency stopping...")
            HardwareAuthorityManager.emergency_stop(env)
            interrupted = True
            # Mark episode as failed
            info['success'] = 0.0
            info['interrupted'] = True
            done = True
        
        # Print episode statistics
        if interrupted:
            print(f"\n{'='*60}")
            print(f"Episode {i+1}/{num_eval_episodes} INTERRUPTED (marked as failure).")
            print(f"  Total steps: {step}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Episode {i+1}/{num_eval_episodes} completed.")
            print(f"  Total steps: {step}")
            print(f"  Policy calls: {policy_call_count}")
            print(f"  Actions executed: {action_execution_count}")
            if policy_call_count > 0:
                avg_interval = np.mean([policy_call_steps[j] - policy_call_steps[j-1] 
                                       for j in range(1, len(policy_call_steps))]) if len(policy_call_steps) > 1 else step
                print(f"  Average policy call interval: {avg_interval:.1f} steps")
                print(f"  Policy call steps: {policy_call_steps}")
            print(f"{'='*60}")
        
        # For hardware environments, show final frame and prompt user for success via GUI
        if is_hardware_env and i < num_eval_episodes and not interrupted:
            # Show final frame with success/failure prompt overlay
            final_frame = env.render()
            if final_frame is not None:
                # Add prompt text to frame
                cv2.putText(final_frame, "Episode Complete! Press 'y' for SUCCESS, 'n' for FAILURE", 
                           (10, final_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(final_frame, f"Episode {i+1}/{num_eval_episodes} - Length: {step} steps", 
                           (10, final_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Resize for display
                disp = cv2.resize(
                    final_frame,
                    (final_frame.shape[1] // 2, final_frame.shape[0] // 2)
                )
                cv2.imshow('WX200 Evaluation - ArUco Markers', disp)
                cv2.waitKey(100)  # Small delay to ensure frame is shown
            else:
                print("⚠️  Warning: Camera window not available. Cannot show final frame.")
            
            if final_frame is not None:
                print("Please look at the camera window and evaluate the trajectory.")
                print("Press 'y' for SUCCESS or 'n' for FAILURE in the camera window.")
            else:
                print("Camera window not available. Please enter success/failure via terminal.")
                print("Enter 'y' for SUCCESS or 'n' for FAILURE: ", end='', flush=True)
            
            # Wait for keyboard input from OpenCV window or terminal
            success = 0.0
            if final_frame is not None:
                # Use OpenCV window for input
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('y') or key == ord('Y'):
                        success = 1.0
                        print("Marked as SUCCESS")
                        break
                    elif key == ord('n') or key == ord('N'):
                        success = 0.0
                        print("Marked as FAILURE")
                        break
                    elif key == 27:  # ESC key
                        print("Evaluation cancelled")
                        success = 0.0
                        break
            else:
                # Fallback to terminal input
                user_input = input().strip().lower()
                success = 1.0 if user_input == 'y' or user_input == 'yes' else 0.0
                print(f"Marked as {'SUCCESS' if success > 0 else 'FAILURE'}")
            
            info['success'] = success
            print(f"{'='*60}\n")
        elif interrupted:
            # Episode was interrupted, mark as failed
            info['success'] = 0.0
            info['interrupted'] = True

        if not interrupted:
            mc_returns = discounted_returns_from_prefix(rewards, decision_indices, discount=0.99)
            time_series_data.append({
                'q_preds': q_preds_at_decisions,
                'mc_returns': mc_returns,
            })
            scatter_data.extend(list(zip(q_preds_at_decisions, mc_returns)))

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            if not interrupted:
                trajs.append(traj)
        else:
            if render:
                renders.append(np.array(render))
        
        if interrupted:
            print("\n⚠️  Evaluation interrupted. Returning partial results.")
            break

    if "kitchen" in env_name.lower():
        final_stats = {}
        for k, v in stats.items():
            if not v:
                continue
            
            first_element = v[0]
            if isinstance(first_element, (int, float, np.number)):
                final_stats[k] = np.mean(v)
            else:
                final_stats[k] = v[-1]
    else:
        for k, v in stats.items():
            stats[k] = np.mean(v)
        final_stats = stats

    if plot:
        visualize_q_accuracy(time_series_data, scatter_data, global_step, suffix, dir_suffix=env_name)
    
    # Save observations if requested
    if save_observations_path is not None and len(all_observations) > 0:
        import pickle
        observations_array = np.array(all_observations)
        save_data = {
            'observations': observations_array,
            'num_episodes': num_eval_episodes,
            'global_step': global_step,
        }
        with open(save_observations_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\n✓ Saved {len(all_observations)} observations to: {save_observations_path}")
        print(f"  Observation shape: {observations_array.shape}")
        print(f"  Use compare_obs_distributions.py to compare with dataset observations")

    return final_stats, trajs, renders

def visualize_q_diagnostics(scatter_data, episode_q_mc_sequences, global_step, suffix, dir_suffix, save_dir='./plots'):
    """
    Visualizes Q-function diagnostics with three subplots:
    1. Q vs. MC scatter plot, colored by value.
    2. Q-value drop vs. MC-return drop scatter plot.
    3. t-SNE embedding of (Q, MC) pairs.
    """
    if not scatter_data:
        return

    save_dir = os.path.join(save_dir, 'q_diagnostics_plots', dir_suffix)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig.suptitle(f'Q-Function Diagnostics\nStep {global_step} ({suffix})', fontsize=16)

    q_preds = np.array([d[0] for d in scatter_data])
    mc_returns = np.array([d[1] for d in scatter_data])
    ax1 = axes[0]
    sc = ax1.scatter(q_preds, mc_returns, c=mc_returns, cmap='plasma', alpha=0.5, s=15, edgecolors='none')
    fig.colorbar(sc, ax=ax1, label='MC Return Value')
    min_val, max_val = min(q_preds.min(), mc_returns.min()), max(q_preds.max(), mc_returns.max())
    ax1.plot([min_val, max_val], [min_val, max_val], '--', color='red', alpha=0.8, linewidth=2, label='y = x')
    ax1.set_title('Predicted Q vs. MC Return')
    ax1.set_xlabel('Predicted Q-Value')
    ax1.set_ylabel('Actual MC Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', 'box')

    # --- Subplot 2: Value Drop Comparison ---
    ax2 = axes[1]
    q_diffs, mc_diffs = [], []
    for q_seq, mc_seq in episode_q_mc_sequences:
        if len(q_seq) > 1:
            q_diffs.extend(q_seq[:-1] - q_seq[1:])
            mc_diffs.extend(mc_seq[:-1] - mc_seq[1:])
    
    ax2.scatter(q_diffs, mc_diffs, alpha=0.2, s=15, edgecolors='none', color='royalblue')
    min_val, max_val = min(min(q_diffs), min(mc_diffs)), max(max(q_diffs), max(mc_diffs))
    ax2.plot([min_val, max_val], [min_val, max_val], '--', color='red', alpha=0.8, linewidth=2, label='y = x')
    ax2.set_title('Value Drop Consistency')
    ax2.set_xlabel("Predicted Drop: $Q_t - Q_{t+1}$")
    ax2.set_ylabel("Actual Drop: $MC_t - MC_{t+1}$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')

    # --- Subplot 3: t-SNE Visualization ---
    ax3 = axes[2]
    # Combine Q and MC into a 2D array and scale it
    X = np.vstack([q_preds, mc_returns]).T
    X_scaled = StandardScaler().fit_transform(X)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', max_iter=500)
    embeddings = tsne.fit_transform(X_scaled)
    
    # Plot the t-SNE embeddings, colored by MC return value
    sc_tsne = ax3.scatter(embeddings[:, 0], embeddings[:, 1], c=mc_returns, cmap='viridis', alpha=0.6, s=15)
    fig.colorbar(sc_tsne, ax=ax3, label='MC Return Value')
    ax3.set_title('t-SNE of (Q, MC) Pairs')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', 'box')

    save_path = os.path.join(save_dir, f'step_{global_step}_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Q-function diagnostics plot saved to: {save_path}")
    
# Removed evaluate_parallel - real world is single-threaded only
