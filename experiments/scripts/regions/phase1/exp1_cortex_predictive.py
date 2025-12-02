"""
Experiment 1: Cortex - Predictive Coding for Visual Representation Learning

This experiment tests whether the Cortex brain region with predictive coding
enabled can learn useful visual representations from MNIST digits.

The Cortex class now supports two learning modes:
1. Hebbian STDP (default) - unsupervised correlation-based learning
2. Predictive Coding - minimize prediction error (this experiment)

Key Principles:
1. The cortex predicts its input from its representation
2. Prediction errors drive representation updates
3. Learning minimizes prediction error
4. Representations should capture input structure

Success Criteria:
1. Reconstruction error decreases >50% during training
2. Learned representations preserve neighborhood structure (kNN within 10% of raw)
3. Some clustering emerges (NMI > 0.1)

Based on:
- Rao & Ballard (1999) - Predictive Coding in Visual Cortex
- Friston (2005) - Free Energy Principle
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.base import LearningRule
from experiments.scripts.regions.exp_utils import (
    load_mnist_subset,
    save_results,
    get_results_dir,
)


def train_cortex_predictive(
    cortex: Cortex,
    images: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train the Cortex with predictive coding on images.
    
    Args:
        cortex: Cortex instance with learning_rule=LearningRule.PREDICTIVE
        images: Training images (n_samples, n_features)
        n_epochs: Number of training epochs
        batch_size: Mini-batch size
        verbose: Print progress
        
    Returns:
        History dict with training metrics
    """
    n_samples = len(images)
    device = cortex.device
    
    error_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        epoch_errors = []
        
        for batch_start in range(0, n_samples, batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            
            for idx in batch_idx:
                # Get single image as rate-coded input
                image = torch.tensor(images[idx], dtype=torch.float32, device=device)
                
                # Run predictive inference
                cortex.predictive_forward(image)
                
                # Learn from this sample
                metrics = cortex.learn(image, image)  # output_spikes not used in PC mode
                epoch_errors.append(metrics.get("pred_error", 0.0))
        
        avg_error = np.mean(epoch_errors)
        error_history.extend(epoch_errors)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}: pred_error={avg_error:.4f}")
    
    return {"error_history": error_history}


def get_representations(cortex: Cortex, images: np.ndarray) -> np.ndarray:
    """Extract representations from trained cortex.
    
    Args:
        cortex: Trained Cortex with predictive coding
        images: Input images
        
    Returns:
        representations: (n_samples, n_output)
    """
    device = cortex.device
    representations = []
    
    with torch.no_grad():
        for i in range(len(images)):
            image = torch.tensor(images[i], dtype=torch.float32, device=device)
            repr_tensor = cortex.predictive_forward(image)
            representations.append(repr_tensor.squeeze().cpu().numpy())
    
    return np.array(representations)


def get_reconstructions(cortex: Cortex, images: np.ndarray) -> np.ndarray:
    """Get reconstructions of input images.
    
    Args:
        cortex: Trained Cortex with predictive coding
        images: Input images
        
    Returns:
        reconstructions: (n_samples, n_input)
    """
    device = cortex.device
    reconstructions = []
    
    with torch.no_grad():
        for i in range(len(images)):
            image = torch.tensor(images[i], dtype=torch.float32, device=device)
            cortex.predictive_forward(image)
            recon = cortex.get_prediction()
            reconstructions.append(recon.cpu().numpy())
    
    return np.array(reconstructions)


def evaluate_representations(representations: np.ndarray, labels: np.ndarray) -> Dict:
    """Evaluate quality of learned representations."""
    # Handle NaN/Inf
    if not np.isfinite(representations).all():
        return {"linear_acc": 0.0, "knn_acc": 0.0, "nmi": 0.0, "ari": 0.0}
    
    # Linear classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(representations, labels)
    linear_acc = clf.score(representations, labels)
    
    # kNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(representations, labels)
    knn_acc = knn.score(representations, labels)
    
    # Clustering metrics
    n_classes = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(representations)
    
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    
    return {
        "linear_acc": linear_acc,
        "knn_acc": knn_acc,
        "nmi": nmi,
        "ari": ari,
    }


def compute_baselines(images: np.ndarray, labels: np.ndarray, n_components: int) -> Dict:
    """Compute baseline metrics for comparison."""
    # Raw pixels
    clf_raw = LogisticRegression(max_iter=1000, random_state=42)
    clf_raw.fit(images, labels)
    raw_linear_acc = clf_raw.score(images, labels)
    
    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(images, labels)
    raw_knn_acc = knn_raw.score(images, labels)
    
    # Random projection
    np.random.seed(42)
    random_proj = np.random.randn(images.shape[1], n_components) / np.sqrt(n_components)
    random_repr = images @ random_proj
    
    clf_random = LogisticRegression(max_iter=1000, random_state=42)
    clf_random.fit(random_repr, labels)
    random_linear_acc = clf_random.score(random_repr, labels)
    
    knn_random = KNeighborsClassifier(n_neighbors=5)
    knn_random.fit(random_repr, labels)
    random_knn_acc = knn_random.score(random_repr, labels)
    
    # PCA
    pca = PCA(n_components=min(n_components, images.shape[1]), random_state=42)
    pca_repr = pca.fit_transform(images)
    
    clf_pca = LogisticRegression(max_iter=1000, random_state=42)
    clf_pca.fit(pca_repr, labels)
    pca_linear_acc = clf_pca.score(pca_repr, labels)
    
    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(pca_repr, labels)
    pca_knn_acc = knn_pca.score(pca_repr, labels)
    
    return {
        "raw_linear_acc": raw_linear_acc,
        "raw_knn_acc": raw_knn_acc,
        "random_linear_acc": random_linear_acc,
        "random_knn_acc": random_knn_acc,
        "pca_linear_acc": pca_linear_acc,
        "pca_knn_acc": pca_knn_acc,
    }


def plot_results(
    cortex: Cortex,
    history: Dict,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    save_dir: Path,
):
    """Generate visualization plots."""
    # Plot learning curves
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["error_history"])
    ax.set_xlabel("Sample")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Prediction Error During Training")
    plt.tight_layout()
    plt.savefig(save_dir / "exp1_learning.png", dpi=150)
    plt.close()
    
    # Plot reconstructions
    reconstructions = get_reconstructions(cortex, test_images[:5])
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    img_size = int(np.sqrt(test_images.shape[1]))
    
    for i in range(5):
        axes[0, i].imshow(test_images[i].reshape(img_size, img_size), cmap='gray')
        axes[0, i].set_title(f"Original (y={test_labels[i]})")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructions[i].reshape(img_size, img_size), cmap='gray')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / "exp1_reconstructions.png", dpi=150)
    plt.close()
    
    # Plot prediction weights as receptive fields
    W = cortex.prediction_weights.detach().cpu().numpy()
    n_show = min(16, W.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    for i in range(n_show):
        ax = axes[i // 4, i % 4]
        rf = W[i].reshape(img_size, img_size)
        ax.imshow(rf, cmap='RdBu_r')
        ax.axis('off')
    
    plt.suptitle("Learned Receptive Fields (Prediction Weights)")
    plt.tight_layout()
    plt.savefig(save_dir / "exp1_receptive_fields.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Exp1: Cortex Predictive Coding")
    parser.add_argument("--n-train", type=int, default=2000, help="Training samples")
    parser.add_argument("--n-test", type=int, default=1000, help="Test samples")
    parser.add_argument("--n-epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--inference-steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--downsample", type=int, default=4, help="Downsample factor")
    parser.add_argument("--n-output", type=int, default=64, help="Output/representation size")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXPERIMENT 1: Cortex - Predictive Coding")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading MNIST data...")
    total_needed = args.n_train + args.n_test
    all_images, all_labels = load_mnist_subset(
        n_samples=total_needed,
        downsample=args.downsample,
    )
    train_images = all_images[:args.n_train]
    train_labels = all_labels[:args.n_train]
    test_images = all_images[args.n_train:]
    test_labels = all_labels[args.n_train:]
    
    n_input = train_images.shape[1]
    img_size = int(np.sqrt(n_input))
    print(f"  Input dimension: {n_input} ({img_size}x{img_size})")
    print(f"  Training samples: {len(train_images)}")
    print(f"  Test samples: {len(test_images)}")
    
    # Create Cortex with predictive coding
    print("\n[2/5] Creating Cortex with Predictive Coding...")
    config = CortexConfig(
        n_input=n_input,
        n_output=args.n_output,
        # Disable Hebbian mechanisms
        hebbian_lr=0.0,
        # Enable predictive coding via learning_rule
        learning_rule=LearningRule.PREDICTIVE,
        predictive_lr=args.lr,
        predictive_inference_steps=args.inference_steps,
        predictive_tau=10.0,
        predictive_dt=0.1,
        # Disable other mechanisms for clean test
        sfa_enabled=False,
        lateral_inhibition=False,
        kwta_k=0,
        diagonal_bias=0.0,
    )
    cortex = Cortex(config)
    print(f"  Input: {n_input}, Output: {args.n_output}")
    print(f"  Predictive LR: {config.predictive_lr}")
    print(f"  Inference steps: {config.predictive_inference_steps}")
    
    # Train
    print(f"\n[3/5] Training for {args.n_epochs} epochs...")
    history = train_cortex_predictive(
        cortex=cortex,
        images=train_images,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        verbose=True,
    )
    
    # Evaluate
    print("\n[4/5] Evaluating representations...")
    
    # Get representations
    representations = get_representations(cortex, test_images)
    
    print("  Computing baselines...")
    baselines = compute_baselines(test_images, test_labels, n_components=args.n_output)
    
    print("  Evaluating learned representations...")
    metrics = evaluate_representations(representations, test_labels)
    
    # Compute reconstruction error
    reconstructions = get_reconstructions(cortex, test_images)
    recon_error = np.mean((test_images - reconstructions) ** 2)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    CLASSIFICATION ACCURACY                          │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Raw Pixels (baseline)     │   Linear: {baselines['raw_linear_acc']*100:.1f}%  │  kNN: {baselines['raw_knn_acc']*100:.1f}%  │")
    print(f"│  Random Projection         │   Linear: {baselines['random_linear_acc']*100:.1f}%  │  kNN: {baselines['random_knn_acc']*100:.1f}%  │")
    print(f"│  PCA ({args.n_output}d)                │   Linear: {baselines['pca_linear_acc']*100:.1f}%  │  kNN: {baselines['pca_knn_acc']*100:.1f}%  │")
    print(f"│  Cortex PC ({args.n_output}d)          │   Linear: {metrics['linear_acc']*100:.1f}%  │  kNN: {metrics['knn_acc']*100:.1f}%  │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    CLUSTERING QUALITY                               │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Cortex NMI:  {metrics['nmi']:.3f}  (0=no info, 1=perfect)                     │")
    print(f"│  Cortex ARI:  {metrics['ari']:.3f}  (-1 to 1, 0=random)                        │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Check reconstruction improvement
    early_errors = history["error_history"][:100]
    late_errors = history["error_history"][-100:]
    early_error = np.mean(early_errors) if early_errors else 1.0
    late_error = np.mean(late_errors) if late_errors else 1.0
    error_improvement = (early_error - late_error) / early_error * 100 if early_error > 0 else 0
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    RECONSTRUCTION                                   │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Initial pred error: {early_error:.4f}                                         │")
    print(f"│  Final pred error:   {late_error:.4f}                                         │")
    print(f"│  Improvement:        {error_improvement:.1f}%                                            │")
    print(f"│  Test MSE:           {recon_error:.4f}                                         │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Save plots
    print("\n[5/5] Generating visualizations...")
    results_dir = get_results_dir()
    plot_results(cortex, history, test_images, test_labels, results_dir)
    print(f"  Saved plots to {results_dir}")
    
    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)
    
    passed = 0
    total = 3
    
    # Criterion 1: Reconstruction error decreased significantly
    if error_improvement > 50:
        print(f"  1. Pred error improved >50%: {error_improvement:.1f}% → ✓")
        passed += 1
    else:
        print(f"  1. Pred error improved >50%: {error_improvement:.1f}% → ✗")
    
    # Criterion 2: kNN competitive with raw pixels
    knn_ratio = metrics["knn_acc"] / baselines["raw_knn_acc"]
    if knn_ratio > 0.9:
        print(f"  2. kNN within 10% of raw: {metrics['knn_acc']*100:.1f}% vs {baselines['raw_knn_acc']*100:.1f}% → ✓")
        passed += 1
    else:
        print(f"  2. kNN within 10% of raw: {metrics['knn_acc']*100:.1f}% vs {baselines['raw_knn_acc']*100:.1f}% → ✗")
    
    # Criterion 3: Some clustering structure
    if metrics["nmi"] > 0.1:
        print(f"  3. NMI > 0.1: {metrics['nmi']:.3f} → ✓")
        passed += 1
    else:
        print(f"  3. NMI > 0.1: {metrics['nmi']:.3f} → ✗")
    
    if passed >= 2:
        print(f"\nOverall: ✓ PASSED ({passed}/{total} criteria met)")
    else:
        print(f"\nOverall: ✗ FAILED ({passed}/{total} criteria met)")
    
    # Save results
    results = {
        "config": {
            "n_input": n_input,
            "n_output": args.n_output,
            "predictive_lr": config.predictive_lr,
            "inference_steps": config.predictive_inference_steps,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
        },
        "metrics": {
            "linear_acc": metrics["linear_acc"],
            "knn_acc": metrics["knn_acc"],
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
            "error_improvement": error_improvement,
            "test_recon_mse": recon_error,
        },
        "baselines": baselines,
        "passed": passed >= 2,
    }
    
    save_results(name="exp1_cortex_predictive", results=results)


if __name__ == "__main__":
    main()
