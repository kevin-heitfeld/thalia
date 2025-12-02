"""
Experiment utilities for brain region experiments.

Common functions for data loading, metrics, and visualization.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any


def get_project_root() -> Path:
    """Get the project root directory."""
    # exp_utils.py is in experiments/scripts/regions/
    # So we need 3 parents to get to thalia/
    return Path(__file__).parent.parent.parent.parent


def get_results_dir() -> Path:
    """Get the results directory, creating if needed."""
    results = get_project_root() / "experiments" / "results" / "regions"
    results.mkdir(parents=True, exist_ok=True)
    return results


def load_mnist_subset(
    n_samples: int = 1000,
    digits: Optional[List[int]] = None,
    flatten: bool = True,
    normalize: bool = True,
    downsample: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a subset of MNIST data.
    
    Args:
        n_samples: Number of samples to load
        digits: List of digit classes to include (None = all)
        flatten: Whether to flatten images to 1D
        normalize: Whether to normalize to [0, 1]
        downsample: Factor to downsample images (e.g., 4 -> 7x7 from 28x28)
        
    Returns:
        images: (n_samples, dim) or (n_samples, h, w)
        labels: (n_samples,)
    """
    data_dir = get_project_root() / "data" / "MNIST" / "raw"
    
    # Load training images
    with open(data_dir / "train-images-idx3-ubyte", "rb") as f:
        f.read(16)  # Skip header
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    
    # Load training labels
    with open(data_dir / "train-labels-idx1-ubyte", "rb") as f:
        f.read(8)  # Skip header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Filter by digit class if specified
    if digits is not None:
        mask = np.isin(labels, digits)
        images = images[mask]
        labels = labels[mask]
    
    # Subsample
    if n_samples < len(images):
        indices = np.random.choice(len(images), n_samples, replace=False)
        images = images[indices]
        labels = labels[indices]
    
    # Downsample images
    if downsample is not None:
        new_size = 28 // downsample
        downsampled = np.zeros((len(images), new_size, new_size), dtype=np.float32)
        for i in range(new_size):
            for j in range(new_size):
                downsampled[:, i, j] = images[
                    :, 
                    i*downsample:(i+1)*downsample, 
                    j*downsample:(j+1)*downsample
                ].mean(axis=(1, 2))
        images = downsampled
    
    # Normalize
    if normalize:
        images = images.astype(np.float32) / 255.0
    
    # Flatten
    if flatten:
        images = images.reshape(len(images), -1)
    
    return images, labels


def create_xor_dataset(
    n_samples: int = 1000,
    noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create XOR classification dataset.
    
    Args:
        n_samples: Number of samples
        noise: Gaussian noise std
        
    Returns:
        X: (n_samples, 2) input coordinates
        y: (n_samples,) XOR labels (0 or 1)
    """
    # Generate random points in [0, 1]^2
    X = np.random.rand(n_samples, 2)
    
    # XOR: True if exactly one coordinate > 0.5
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(np.float32)
    
    # Add noise
    X = X + np.random.randn(*X.shape) * noise
    X = np.clip(X, 0, 1).astype(np.float32)
    
    return X, y


def create_bandit_env(
    n_arms: int = 4,
    reward_probs: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Create a multi-armed bandit environment.
    
    Args:
        n_arms: Number of arms
        reward_probs: Probability of reward for each arm
        
    Returns:
        Environment dict with pull() function
    """
    if reward_probs is None:
        reward_probs = np.random.rand(n_arms)
    reward_probs = np.array(reward_probs)
    
    def pull(arm: int) -> float:
        """Pull an arm and get reward."""
        return float(np.random.rand() < reward_probs[arm])
    
    return {
        "n_arms": n_arms,
        "reward_probs": reward_probs,
        "optimal_arm": int(np.argmax(reward_probs)),
        "pull": pull
    }


def create_sequence_patterns(
    n_sequences: int = 20,
    pattern_size: int = 64,
    sparsity: float = 0.1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cue-target pattern pairs for sequence memory.
    
    Args:
        n_sequences: Number of pattern pairs
        pattern_size: Dimension of each pattern
        sparsity: Fraction of active units
        
    Returns:
        List of (cue, target) pairs
    """
    n_active = int(pattern_size * sparsity)
    patterns = []
    
    for _ in range(n_sequences):
        # Random sparse cue
        cue = np.zeros(pattern_size, dtype=np.float32)
        cue[np.random.choice(pattern_size, n_active, replace=False)] = 1.0
        
        # Random sparse target
        target = np.zeros(pattern_size, dtype=np.float32)
        target[np.random.choice(pattern_size, n_active, replace=False)] = 1.0
        
        patterns.append((cue, target))
    
    return patterns


def create_delay_task(
    n_samples: int = 100,
    n_classes: int = 4,
    pattern_size: int = 32,
    delay_range: Tuple[int, int] = (5, 50)
) -> List[Dict[str, Any]]:
    """
    Create delayed match-to-sample trials.
    
    Args:
        n_samples: Number of trials
        n_classes: Number of distinct samples
        pattern_size: Dimension of sample patterns
        delay_range: (min_delay, max_delay) in timesteps
        
    Returns:
        List of trial dicts with sample, delay, test, is_match
    """
    # Create prototype patterns for each class
    prototypes = [np.random.rand(pattern_size).astype(np.float32) for _ in range(n_classes)]
    
    trials = []
    for _ in range(n_samples):
        # Random sample class
        sample_class = np.random.randint(n_classes)
        sample = prototypes[sample_class]
        
        # Random delay
        delay = np.random.randint(delay_range[0], delay_range[1] + 1)
        
        # 50% match, 50% non-match
        is_match = np.random.rand() > 0.5
        if is_match:
            test = sample.copy()
        else:
            other_class = np.random.choice([c for c in range(n_classes) if c != sample_class])
            test = prototypes[other_class]
        
        trials.append({
            "sample": sample,
            "sample_class": sample_class,
            "delay": delay,
            "test": test,
            "is_match": is_match
        })
    
    return trials


def compute_selectivity(
    responses: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Compute selectivity of each neuron to each class.
    
    Args:
        responses: (n_samples, n_neurons) activity matrix
        labels: (n_samples,) class labels
        
    Returns:
        selectivity: (n_neurons, n_classes) mean response per class
    """
    classes = np.unique(labels)
    n_neurons = responses.shape[1]
    selectivity = np.zeros((n_neurons, len(classes)))
    
    for i, c in enumerate(classes):
        mask = labels == c
        selectivity[:, i] = responses[mask].mean(axis=0)
    
    return selectivity


def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """Compute entropy of probability distribution."""
    probs = np.clip(probs, eps, 1 - eps)
    probs = probs / probs.sum()
    return -np.sum(probs * np.log2(probs))


def cumulative_regret(
    actions: List[int],
    rewards: List[float],
    optimal_arm: int,
    optimal_prob: float
) -> np.ndarray:
    """
    Compute cumulative regret over trials.
    
    Args:
        actions: List of chosen arms
        rewards: List of received rewards
        optimal_arm: Index of best arm
        optimal_prob: Reward probability of best arm
        
    Returns:
        cumulative_regret: (n_trials,) cumulative regret at each step
    """
    regret = []
    cum = 0
    for a, r in zip(actions, rewards):
        # Regret is expected reward of optimal - expected reward of chosen
        instant_regret = optimal_prob - r
        cum += instant_regret
        regret.append(cum)
    return np.array(regret)


def save_results(
    name: str,
    results: Dict[str, Any],
    include_timestamp: bool = True
) -> Path:
    """
    Save experiment results to file.
    
    Args:
        name: Experiment name
        results: Dict of results to save
        include_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to saved file
    """
    import json
    from datetime import datetime
    
    results_dir = get_results_dir()
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
    else:
        filename = f"{name}.json"
    
    filepath = results_dir / filename
    
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def plot_weights_grid(
    weights: np.ndarray,
    n_cols: int = 8,
    title: str = "Learned Weights"
) -> None:
    """
    Plot weight vectors as a grid of images.
    
    Args:
        weights: (n_neurons, input_dim) weight matrix
        n_cols: Number of columns in grid
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    n_neurons = weights.shape[0]
    input_dim = weights.shape[1]
    
    # Infer image dimensions
    side = int(np.sqrt(input_dim))
    if side * side != input_dim:
        print(f"Warning: input_dim {input_dim} is not a perfect square")
        return
    
    n_rows = (n_neurons + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).flatten()
    
    for i in range(len(axes)):
        if i < n_neurons:
            img = weights[i].reshape(side, side)
            axes[i].imshow(img, cmap='viridis')
        axes[i].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(
    values: np.ndarray,
    xlabel: str = "Trial",
    ylabel: str = "Value",
    title: str = "Learning Curve",
    smooth_window: int = 50
) -> None:
    """
    Plot a learning curve with optional smoothing.
    
    Args:
        values: (n_trials,) values to plot
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        smooth_window: Window size for smoothing (0 = no smoothing)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(values, alpha=0.3, color='blue', label='Raw')
    
    if smooth_window > 0 and len(values) > smooth_window:
        smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
        plt.plot(range(smooth_window-1, len(values)), smoothed, color='blue', label=f'Smoothed ({smooth_window})')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
