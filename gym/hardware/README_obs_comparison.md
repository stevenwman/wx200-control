# Observation Distribution Comparison Guide

This guide explains how to gather evidence on observation distribution mismatch between dataset and live robot observations.

## Step 1: Run Evaluation with Observation Saving

When running evaluation, add the `save_observations_path` parameter to save live robot observations:

```python
eval_info, _, _ = evaluate(
    agent=agent,
    env=eval_env,
    global_step=log_step,
    action_dim=action_dim,
    num_eval_episodes=3,  # Use small number for testing
    num_video_episodes=0,
    video_frame_skip=FLAGS.video_frame_skip,
    actor_fn=agent.compute_flow_actions,
    env_name=FLAGS.env_name,
    plot=False,
    save_observations_path="live_observations_step800000.pkl"  # Add this parameter
)
```

Or modify `main_ogpo_real.py` to add this parameter to the evaluate calls.

## Step 2: Run Comparison Script

After saving observations, run the comparison script:

```bash
cd /home/steven/Desktop/work/research/action_chunk_q_learning
python envs/hardware/compare_obs_distributions.py
```

Make sure to update `LIVE_OBS_FILE` in the script to point to your saved observations file.

## Step 3: Analyze Results

The script will:

1. **Load both datasets** - Dataset observations and live robot observations
2. **Compute statistics** - Mean, std, min, max, percentiles for each dimension
3. **Compare distributions** - Flag dimensions where differences exceed thresholds
4. **Visualize** - Create histogram plots comparing distributions side-by-side

### Understanding the Output

- **Statistics table**: Shows per-dimension statistics for both distributions
- **Comparison table**: Shows differences and flags dimensions with significant shifts
  - `MEAN_OFF`: Mean value is significantly different
  - `STD_OFF`: Standard deviation is significantly different  
  - `RANGE_OFF`: Min/max values are significantly different
- **Plots**: Histograms showing distribution overlap (or lack thereof)

### What to Look For

- **Large mean differences** (>2 standard deviations): Indicates systematic offset
- **Large std differences**: Indicates different variability
- **Range mismatches**: Observations outside training distribution
- **Multiple flagged dimensions**: Suggests systematic distribution shift

## Quick Test Without Live Observations

To just see dataset statistics:

```bash
python envs/hardware/compare_obs_distributions.py
```

The script will show dataset-only statistics if `LIVE_OBS_FILE` is not set.
