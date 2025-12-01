"""Weight matrix analysis and evaluation functions.

This module provides functions for analyzing trained weight matrices
to evaluate learning success and network structure.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch
import numpy as np


def compute_diagonal_score(
    weights: torch.Tensor,
    expected_mapping: Optional[np.ndarray] = None,
) -> Tuple[int, int, List[int]]:
    """Compute how many neurons learned the expected input→output mapping.

    For a feedforward weight matrix, this checks if each output neuron's
    strongest input comes from the expected source.

    Args:
        weights: Weight matrix of shape (n_output, n_input)
        expected_mapping: Array of expected input indices for each output.
            If None, assumes linear mapping (output i ← input i*n_input/n_output).

    Returns:
        Tuple of (correct_count, total, actual_mapping_list):
        - correct_count: Number of outputs with correct strongest input
        - total: Total number of output neurons
        - actual_mapping_list: List of actual strongest input for each output

    Example:
        >>> weights = torch.randn(10, 20)  # 10 outputs, 20 inputs
        >>> correct, total, mapping = compute_diagonal_score(weights)
        >>> print(f"Diagonal score: {correct}/{total}")
    """
    n_output, n_input = weights.shape

    # Default to linear mapping if not specified
    if expected_mapping is None:
        expected_mapping = np.linspace(0, n_input - 1, n_output).round().astype(int)

    # Find strongest input for each output
    max_per_output = weights.argmax(dim=1).cpu().numpy()

    # Count matches
    correct = sum(
        1 for i in range(n_output)
        if max_per_output[i] == expected_mapping[i]
    )

    return correct, n_output, max_per_output.tolist()


def compute_paired_diagonal_score(
    weights: torch.Tensor,
) -> Tuple[int, int, List[int], str]:
    """Compute diagonal score for N:1 input:output ratio, checking all valid mappings.
    
    When n_input = ratio * n_output, the network might learn any of the valid mappings:
    - For ratio=2: output i responds to input 2i or 2i+1
    - For ratio=4: output i responds to input 4i, 4i+1, 4i+2, or 4i+3
    - For ratio=1: output i responds to input i (exact diagonal)
    
    This function checks EACH neuron against ANY valid input in its group.
    A neuron is correct if it responds to any input in [ratio*i, ratio*i + ratio-1].
    
    Args:
        weights: Weight matrix of shape (n_output, n_input) where n_input = ratio * n_output
            for some integer ratio >= 1
        
    Returns:
        Tuple of (best_score, total, actual_mapping, pattern_name):
        - best_score: Number of neurons matching any valid input in their group
        - total: Total number of output neurons  
        - actual_mapping: List of actual strongest input for each output
        - pattern_name: Description of the ratio pattern (e.g., "ratio=4 (4i to 4i+3)")
        
    Example:
        >>> weights = torch.randn(10, 20)  # 10 outputs, 20 inputs (ratio=2)
        >>> score, total, mapping, pattern = compute_paired_diagonal_score(weights)
        >>> print(f"Diagonal score ({pattern}): {score}/{total}")
        
        >>> weights = torch.randn(5, 20)  # 5 outputs, 20 inputs (ratio=4)
        >>> score, total, mapping, pattern = compute_paired_diagonal_score(weights)
        >>> print(f"Diagonal score ({pattern}): {score}/{total}")
    """
    n_output, n_input = weights.shape
    
    # Compute ratio and verify it's an integer
    if n_output == 0:
        return 0, 0, [], "empty"
    
    ratio = n_input / n_output
    if ratio != int(ratio) or ratio < 1:
        raise ValueError(
            f"Expected n_input to be an integer multiple of n_output, "
            f"got n_input={n_input}, n_output={n_output} (ratio={ratio:.2f})"
        )
    ratio = int(ratio)
    
    # Find strongest input for each output
    max_per_output = weights.argmax(dim=1).cpu().numpy()
    
    # Count neurons that match ANY valid input in their group
    # For output neuron i, valid inputs are: ratio*i, ratio*i+1, ..., ratio*i+(ratio-1)
    score = sum(
        1 for i in range(n_output) 
        if ratio * i <= max_per_output[i] < ratio * (i + 1)
    )
    
    # Generate pattern name
    if ratio == 1:
        pattern_name = "exact diagonal"
    elif ratio == 2:
        pattern_name = "2i or 2i+1"
    else:
        pattern_name = f"ratio={ratio} ({ratio}i to {ratio}i+{ratio-1})"
    
    return score, n_output, max_per_output.tolist(), pattern_name


@dataclass
class RecurrentAnalysis:
    """Results of recurrent weight structure analysis."""
    pattern_type: str  # "circular" or "gapped"
    correct_count: int
    total_analyzed: int
    connections: List[Dict[str, Any]]  # Details for each neuron

    @property
    def accuracy(self) -> float:
        """Fraction of correct connections."""
        return self.correct_count / self.total_analyzed if self.total_analyzed > 0 else 0.0


def analyze_recurrent_structure(
    recurrent_weights: torch.Tensor,
    pattern_type: str = "gapped",
    n_analyze: int = 5,
) -> RecurrentAnalysis:
    """Analyze learned recurrent weight structure for sequential patterns.

    For sequence learning, we expect recurrent weights to form:
    - CIRCULAR: i→(i+1)%n for all neurons (full loop)
    - GAPPED: i→i+1 for i<n-1, last neuron has weak outgoing

    Args:
        recurrent_weights: Recurrent weight matrix (n_output, n_output)
        pattern_type: "circular" or "gapped" pattern expectation
        n_analyze: Number of neurons to analyze (from start)

    Returns:
        RecurrentAnalysis with connection details and accuracy
    """
    n_output = recurrent_weights.shape[0]
    n_analyze = min(n_analyze, n_output)

    connections = []
    correct_count = 0

    for i in range(n_analyze):
        # Find strongest outgoing connection (excluding self)
        outgoing = recurrent_weights[i, :].clone()
        outgoing[i] = -999  # Exclude self-connection
        strongest_target = int(outgoing.argmax().item())
        strongest_weight = outgoing[strongest_target].item()

        conn_info: Dict[str, Any] = {
            "source": i,
            "target": strongest_target,
            "weight": strongest_weight,
        }

        if pattern_type == "circular":
            # Circular: expect i→(i+1)%n for all
            expected_target = (i + 1) % n_output
            is_correct = strongest_target == expected_target
            conn_info["expected"] = expected_target
            conn_info["correct"] = is_correct
            if is_correct:
                correct_count += 1
        else:  # gapped
            # Gapped: expect i→i+1 for i<n-1
            if i < n_output - 1:
                expected_target = i + 1
                is_correct = strongest_target == expected_target
                conn_info["expected"] = expected_target
                conn_info["correct"] = is_correct
                if is_correct:
                    correct_count += 1
            else:
                # Last neuron: weak outgoing is good
                conn_info["expected"] = None
                conn_info["correct"] = strongest_weight < 0.3

        connections.append(conn_info)

    return RecurrentAnalysis(
        pattern_type=pattern_type,
        correct_count=correct_count,
        total_analyzed=n_analyze,
        connections=connections,
    )


@dataclass
class WeightStatistics:
    """Statistics about a weight matrix."""
    min: float
    max: float
    mean: float
    std: float
    sparsity: float  # Fraction of weights below threshold

    def __str__(self) -> str:
        return (
            f"WeightStats(min={self.min:.4f}, max={self.max:.4f}, "
            f"mean={self.mean:.4f}, std={self.std:.4f}, sparsity={self.sparsity:.2%})"
        )


def compute_weight_statistics(
    weights: torch.Tensor,
    sparsity_threshold: float = 0.01,
) -> WeightStatistics:
    """Compute statistics about a weight matrix.

    Args:
        weights: Weight tensor of any shape
        sparsity_threshold: Threshold below which weights count as "sparse"

    Returns:
        WeightStatistics with min, max, mean, std, sparsity
    """
    flat = weights.flatten()

    return WeightStatistics(
        min=float(flat.min().item()),
        max=float(flat.max().item()),
        mean=float(flat.mean().item()),
        std=float(flat.std().item()),
        sparsity=float((flat.abs() < sparsity_threshold).float().mean().item()),
    )


def print_recurrent_analysis(analysis: RecurrentAnalysis) -> None:
    """Print recurrent analysis results in a readable format."""
    print(f"  Learned Sequential Structure (pattern: {analysis.pattern_type}):")

    for conn in analysis.connections:
        source = conn["source"]
        target = conn["target"]
        weight = conn["weight"]
        expected = conn["expected"]
        correct = conn["correct"]

        if expected is None:
            expected_str = "none (end)"
            match = "○" if correct else "?"
        else:
            expected_str = str(expected)
            match = "✓" if correct else "✗"

        print(f"    Neuron {source} → strongest: {target} (w={weight:.3f}) "
              f"[expected: {expected_str}] {match}")

    print(f"    Recurrent accuracy: {analysis.correct_count}/{analysis.total_analyzed} correct")
