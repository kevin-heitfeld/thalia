"""
Low-dimensional manifold analysis for neural trajectories.

High-dimensional neural activity often lies on low-dimensional manifolds.
This module provides tools to:
- Project activity onto low-dimensional spaces
- Visualize thought trajectories
- Identify attractor basins
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn


class ActivityTracker:
    """Track and analyze neural activity over time.

    Stores activity history and provides analysis tools for
    understanding network dynamics.

    Example:
        >>> tracker = ActivityTracker()
        >>> for t in range(1000):
        ...     spikes, _ = network(input)
        ...     tracker.record(spikes)
        >>> trajectory = tracker.get_trajectory()
        >>> pca_coords = tracker.project_pca(n_components=3)
    """

    def __init__(self, max_history: int = 10000):
        """Initialize tracker.

        Args:
            max_history: Maximum timesteps to store
        """
        self.max_history = max_history
        self.history: List[torch.Tensor] = []
        self.timestamps: List[int] = []
        self._t = 0

    def record(self, activity: torch.Tensor) -> None:
        """Record activity at current timestep.

        Args:
            activity: Activity tensor, shape (batch, n_neurons) or (n_neurons,)
        """
        if activity.dim() > 1:
            activity = activity.mean(dim=0)  # Average over batch

        self.history.append(activity.detach().clone())
        self.timestamps.append(self._t)
        self._t += 1

        # Trim if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.timestamps = self.timestamps[-self.max_history:]

    def reset(self) -> None:
        """Clear history."""
        self.history = []
        self.timestamps = []
        self._t = 0

    def get_trajectory(self, smooth_window: int = 1) -> torch.Tensor:
        """Get activity trajectory as tensor.

        Args:
            smooth_window: Window size for smoothing (1 = no smoothing)

        Returns:
            Trajectory, shape (n_timesteps, n_neurons)
        """
        if len(self.history) == 0:
            return torch.tensor([])

        trajectory = torch.stack(self.history)

        if smooth_window > 1:
            # Simple moving average
            kernel = torch.ones(smooth_window) / smooth_window
            # Pad and convolve each neuron
            padded = torch.nn.functional.pad(
                trajectory.T.unsqueeze(1),
                (smooth_window//2, smooth_window//2),
                mode='replicate'
            )
            smoothed = torch.nn.functional.conv1d(
                padded,
                kernel.view(1, 1, -1)
            ).squeeze(1).T
            trajectory = smoothed

        return trajectory

    def project_pca(
        self,
        n_components: int = 3,
        trajectory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project trajectory onto principal components.

        Args:
            n_components: Number of PCA dimensions
            trajectory: Trajectory to project (uses stored if None)

        Returns:
            (projected_coords, explained_variance)
            - projected_coords: shape (n_timesteps, n_components)
            - explained_variance: shape (n_components,)
        """
        if trajectory is None:
            trajectory = self.get_trajectory()

        if len(trajectory) == 0:
            return torch.tensor([]), torch.tensor([])

        # Center data
        mean = trajectory.mean(dim=0)
        centered = trajectory - mean

        # SVD-based PCA
        U, S, V = torch.svd(centered)

        # Project onto top components
        projected = U[:, :n_components] * S[:n_components]

        # Explained variance
        total_var = (S ** 2).sum()
        explained = (S[:n_components] ** 2) / total_var

        return projected, explained

    def distance_to_patterns(
        self,
        patterns: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute distance from trajectory to stored patterns.

        Args:
            patterns: List of pattern tensors

        Returns:
            Distances, shape (n_timesteps, n_patterns)
        """
        trajectory = self.get_trajectory()
        if len(trajectory) == 0 or len(patterns) == 0:
            return torch.tensor([])

        patterns_stack = torch.stack(patterns)  # (n_patterns, n_neurons)

        # Compute euclidean distance
        # trajectory: (T, N), patterns: (P, N)
        diff = trajectory.unsqueeze(1) - patterns_stack.unsqueeze(0)  # (T, P, N)
        distances = torch.norm(diff, dim=-1)  # (T, P)

        return distances

    def find_transitions(
        self,
        patterns: List[torch.Tensor],
        threshold: float = 0.5
    ) -> List[Tuple[int, int, int]]:
        """Find transitions between attractor states.

        Args:
            patterns: Stored patterns
            threshold: Distance threshold for being "in" an attractor

        Returns:
            List of (timestep, from_pattern, to_pattern) tuples
        """
        distances = self.distance_to_patterns(patterns)
        if len(distances) == 0:
            return []

        # Normalize distances
        distances = distances / (distances.max() + 1e-8)

        # Find closest pattern at each timestep
        closest = distances.argmin(dim=1)
        in_attractor = distances.min(dim=1).values < threshold

        # Find transition points
        transitions = []
        current_state = -1

        for t in range(len(closest)):
            if in_attractor[t]:
                if closest[t].item() != current_state:
                    if current_state != -1:
                        transitions.append((t, current_state, closest[t].item()))
                    current_state = closest[t].item()

        return transitions


class ThoughtTrajectory:
    """Represents a sequence of thoughts as attractor transitions.

    A thought trajectory is a path through concept space,
    capturing the flow of mental activity.
    """

    def __init__(self):
        self.states: List[int] = []  # Sequence of attractor indices
        self.times: List[int] = []   # Timestamps of state entries
        self.durations: List[int] = []  # Time spent in each state

    def add_state(self, state: int, time: int) -> None:
        """Record entering a state.

        Args:
            state: Attractor index
            time: Timestamp
        """
        if len(self.states) > 0 and self.states[-1] == state:
            return  # Still in same state

        # Update duration of previous state
        if len(self.times) > 0:
            self.durations.append(time - self.times[-1])

        self.states.append(state)
        self.times.append(time)

    def get_sequence(self) -> List[int]:
        """Get sequence of visited states."""
        return self.states.copy()

    def get_transitions(self) -> List[Tuple[int, int]]:
        """Get list of (from, to) transitions."""
        transitions = []
        for i in range(len(self.states) - 1):
            transitions.append((self.states[i], self.states[i+1]))
        return transitions

    def mean_dwell_time(self) -> float:
        """Average time spent in each state."""
        if len(self.durations) == 0:
            return 0.0
        return sum(self.durations) / len(self.durations)

    def __repr__(self) -> str:
        return f"ThoughtTrajectory({' -> '.join(map(str, self.states))})"
