"""
Compare observation distributions between dataset and live robot observations.

This script:
1. Loads observations from the dataset
2. Optionally loads saved live robot observations (if available)
3. Compares statistics (mean, std, min, max, percentiles) per dimension
4. Visualizes distributions side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from envs.wx200_env_utils_position_targets import get_dataset

def load_dataset_observations(dataset_path, num_samples=None):
    """Load observations from the dataset."""
    print(f"Loading dataset observations from: {dataset_path}")
    dataset = get_dataset(dataset_path)
    
    if num_samples is not None:
        observations = dataset['observations'][:num_samples]
    else:
        observations = dataset['observations']
    
    print(f"Loaded {len(observations)} observations")
    print(f"  Shape: {observations.shape}")
    print(f"  Dtype: {observations.dtype}")
    
    return observations

def load_live_observations(obs_file):
    """Load saved live robot observations from a pickle file."""
    print(f"Loading live observations from: {obs_file}")
    with open(obs_file, 'rb') as f:
        data = pickle.load(f)
    
    observations = data['observations']
    print(f"Loaded {len(observations)} live observations")
    print(f"  Shape: {observations.shape}")
    print(f"  Dtype: {observations.dtype}")
    
    return observations

def compute_statistics(observations, name="Observations"):
    """Compute detailed statistics for observations."""
    stats = {
        'mean': np.mean(observations, axis=0),
        'std': np.std(observations, axis=0),
        'min': np.min(observations, axis=0),
        'max': np.max(observations, axis=0),
        'median': np.median(observations, axis=0),
        'p25': np.percentile(observations, 25, axis=0),
        'p75': np.percentile(observations, 75, axis=0),
        'p5': np.percentile(observations, 5, axis=0),
        'p95': np.percentile(observations, 95, axis=0),
    }
    
    print(f"\n{name} Statistics (per dimension):")
    print(f"{'Dim':<6} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12} {'P5':<12} {'P95':<12}")
    print("-" * 100)
    
    for i in range(observations.shape[1]):
        print(f"{i:<6} "
              f"{stats['mean'][i]:<12.6f} "
              f"{stats['std'][i]:<12.6f} "
              f"{stats['min'][i]:<12.6f} "
              f"{stats['max'][i]:<12.6f} "
              f"{stats['median'][i]:<12.6f} "
              f"{stats['p5'][i]:<12.6f} "
              f"{stats['p95'][i]:<12.6f}")
    
    return stats

def compare_statistics(dataset_stats, live_stats, threshold_std=2.0):
    """Compare statistics and flag dimensions with significant differences."""
    print("\n" + "="*100)
    print("DIMENSION-WISE COMPARISON")
    print("="*100)
    print(f"{'Dim':<6} {'Mean Diff':<15} {'Std Diff':<15} {'Min Diff':<15} {'Max Diff':<15} {'Flag':<20}")
    print("-" * 100)
    
    num_dims = len(dataset_stats['mean'])
    flagged_dims = []
    
    for i in range(num_dims):
        mean_diff = live_stats['mean'][i] - dataset_stats['mean'][i]
        std_diff = live_stats['std'][i] - dataset_stats['std'][i]
        min_diff = live_stats['min'][i] - dataset_stats['min'][i]
        max_diff = live_stats['max'][i] - dataset_stats['max'][i]
        
        # Flag if difference exceeds threshold * dataset std
        dataset_std = dataset_stats['std'][i]
        mean_flag = abs(mean_diff) > threshold_std * dataset_std
        std_flag = abs(std_diff) > threshold_std * dataset_std
        range_flag = (abs(min_diff) > threshold_std * dataset_std) or (abs(max_diff) > threshold_std * dataset_std)
        
        flags = []
        if mean_flag:
            flags.append("MEAN_OFF")
        if std_flag:
            flags.append("STD_OFF")
        if range_flag:
            flags.append("RANGE_OFF")
        
        flag_str = ", ".join(flags) if flags else "OK"
        
        if flags:
            flagged_dims.append(i)
        
        print(f"{i:<6} "
              f"{mean_diff:<15.6f} "
              f"{std_diff:<15.6f} "
              f"{min_diff:<15.6f} "
              f"{max_diff:<15.6f} "
              f"{flag_str:<20}")
    
    print("\n" + "="*100)
    print(f"SUMMARY: {len(flagged_dims)}/{num_dims} dimensions flagged as potentially problematic")
    if flagged_dims:
        print(f"Flagged dimensions: {flagged_dims}")
    print("="*100)
    
    return flagged_dims

def plot_comparison(dataset_obs, live_obs, dim_indices=None, save_path=None):
    """Plot side-by-side histograms for specified dimensions."""
    if dim_indices is None:
        # Plot all dimensions
        dim_indices = list(range(min(dataset_obs.shape[1], live_obs.shape[1])))
    
    num_dims = len(dim_indices)
    cols = 4
    rows = (num_dims + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if num_dims == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, dim in enumerate(dim_indices):
        ax = axes[idx]
        
        # Plot histograms
        ax.hist(dataset_obs[:, dim], bins=50, alpha=0.5, label='Dataset', density=True, color='blue')
        ax.hist(live_obs[:, dim], bins=50, alpha=0.5, label='Live Robot', density=True, color='red')
        
        # Add vertical lines for means
        dataset_mean = np.mean(dataset_obs[:, dim])
        live_mean = np.mean(live_obs[:, dim])
        ax.axvline(dataset_mean, color='blue', linestyle='--', linewidth=2, label=f'Dataset mean: {dataset_mean:.4f}')
        ax.axvline(live_mean, color='red', linestyle='--', linewidth=2, label=f'Live mean: {live_mean:.4f}')
        
        ax.set_xlabel(f'Dim {dim}')
        ax.set_ylabel('Density')
        ax.set_title(f'Dim {dim}: Dataset vs Live')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_dims, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()

def main():
    # Configuration
    DATASET_PATH = "/home/steven/Desktop/work/research/action_chunk_q_learning/envs/hardware/merged_data_states_position_targets.npz"
    LIVE_OBS_FILE = None  # Set this to path of saved live observations pickle file
    NUM_DATASET_SAMPLES = 1000  # Number of dataset samples to compare
    THRESHOLD_STD = 2.0  # Flag dimensions where difference exceeds this many std devs
    
    # Load dataset observations
    dataset_obs = load_dataset_observations(DATASET_PATH, num_samples=NUM_DATASET_SAMPLES)
    
    # Load live observations (if available)
    if LIVE_OBS_FILE and Path(LIVE_OBS_FILE).exists():
        live_obs = load_live_observations(LIVE_OBS_FILE)
        
        # Compute statistics
        dataset_stats = compute_statistics(dataset_obs, "Dataset")
        live_stats = compute_statistics(live_obs, "Live Robot")
        
        # Compare statistics
        flagged_dims = compare_statistics(dataset_stats, live_stats, threshold_std=THRESHOLD_STD)
        
        # Plot comparison
        plot_comparison(dataset_obs, live_obs, dim_indices=flagged_dims if flagged_dims else None,
                       save_path="obs_distribution_comparison.png")
        
        # Also plot all dimensions
        plot_comparison(dataset_obs, live_obs, dim_indices=None,
                       save_path="obs_distribution_comparison_all.png")
    else:
        print("\n" + "="*100)
        print("LIVE OBSERVATIONS NOT PROVIDED")
        print("="*100)
        print("To compare distributions, you need to:")
        print("1. Run evaluation with observation logging enabled")
        print("2. Save observations to a pickle file")
        print("3. Update LIVE_OBS_FILE path in this script")
        print("\nDataset statistics only:")
        compute_statistics(dataset_obs, "Dataset")

if __name__ == "__main__":
    main()
