"""
Spillover transmission (volume transmission) for synaptic weights.

Spillover/volume transmission is a form of non-synaptic neurotransmission where
neurotransmitters released at one synapse diffuse through extracellular space
and activate receptors at nearby synapses.

Key biological features:
- Weaker than direct synaptic transmission (~10-20% strength)
- Affects neurons with shared presynaptic inputs ("functional neighbors")
- No spatial coordinates needed - connectivity defines neighborhood
- Compatible with binary spike representation (ADR-004)

Implementation insight:
    Spillover = weak synapses from functionally proximal neighbors

This module augments existing weight matrices with spillover connections,
providing biologically realistic volume transmission with zero computational
overhead during forward passes (spillover weights computed once at init).

References:
    - Agnati et al. (2010): Volume transmission and wiring transmission
    - Vizi & Lendvai (1999): Nonsynaptic chemical transmission
    - Zoli et al. (1999): Volume transmission in the CNS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

SpilloverMode = Literal["connectivity", "similarity", "lateral"]


@dataclass
class SpilloverConfig:
    """Configuration for spillover transmission.

    Attributes:
        strength: Spillover weight strength relative to direct (0.1-0.2 biological)
        mode: Method for computing spillover neighborhoods
            - 'connectivity': Shared presynaptic inputs define neighbors
            - 'similarity': Weight pattern similarity defines neighbors
            - 'lateral': Simple index-based banded spillover
        lateral_radius: Neighborhood radius for lateral mode (default: 3)
        similarity_threshold: Minimum similarity for spillover in similarity mode (default: 0.5)
        normalize: Whether to normalize spillover weights to prevent runaway excitation
    """

    strength: float = 0.15  # Spillover ~15% of direct synaptic strength
    mode: SpilloverMode = "connectivity"
    lateral_radius: int = 3
    similarity_threshold: float = 0.5
    normalize: bool = True


class SpilloverTransmission:
    """
    Augment synaptic weight matrices with spillover transmission.

    This class computes spillover weights from existing direct synaptic weights,
    creating an effective weight matrix that includes both direct and volume
    transmission. The spillover weights are computed once at initialization
    and cached, so there is no performance penalty during forward passes.

    Biological interpretation:
        Direct synapse: w_ij (neuron i → neuron j)
        Spillover: w_ik * connectivity(k,j) * strength

    Where connectivity(k,j) measures how "close" neurons k and j are in
    functional space (shared inputs, similar weight patterns, or spatial proximity).
    """

    def __init__(
        self,
        base_weights: torch.Tensor,
        config: SpilloverConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize spillover transmission module.

        Args:
            base_weights: Direct synaptic weights [n_post, n_pre]
            config: Spillover configuration
            device: Device for computation (inferred from weights if None)
        """
        self.config = config
        self.device = device or base_weights.device

        # Store direct weights
        self.W_direct = base_weights.to(self.device)

        # Compute spillover weights
        self.W_spillover = self._build_spillover_weights()

        # Combine into effective weight matrix (cached for fast forward passes)
        self.W_effective = self.W_direct + self.W_spillover

    def _build_spillover_weights(self) -> torch.Tensor:
        """
        Compute spillover weight matrix from base connectivity.

        Returns:
            Spillover weight matrix [n_post, n_pre]
        """
        if self.config.mode == "connectivity":
            return self._connectivity_spillover()
        elif self.config.mode == "similarity":
            return self._similarity_spillover()
        elif self.config.mode == "lateral":
            return self._lateral_spillover()
        else:
            raise ValueError(f"Unknown spillover mode: {self.config.mode}")

    def _connectivity_spillover(self) -> torch.Tensor:
        """
        Connectivity-based spillover: neurons sharing presynaptic inputs are neighbors.

        Biological rationale:
            If neurons i and j both receive input from presynaptic neuron k,
            then neurotransmitter spillover from synapse k→i can affect j.

        Algorithm:
            1. Compute postsynaptic neuron similarity via shared inputs
            2. Use similarity to weight spillover from neighbors

        Returns:
            Spillover weight matrix [n_post, n_pre]
        """
        # Binarize weights to find connections
        W_binary = (self.W_direct.abs() > 0).float()

        # Compute postsynaptic similarity: neurons sharing inputs are "close"
        # post_similarity[i,j] = number of shared presynaptic neurons
        post_similarity = W_binary @ W_binary.T  # [n_post, n_post]

        # Normalize to [0, 1]
        max_similarity = post_similarity.max()
        if max_similarity > 0:
            post_similarity = post_similarity / max_similarity

        # Remove self-connections (no spillover to self)
        post_similarity.fill_diagonal_(0)

        # Spillover = weighted average of neighbor synapses
        # If neurons i and j share many inputs, spillover from i affects j
        W_spillover = post_similarity @ self.W_direct * self.config.strength

        # Optional normalization to prevent runaway excitation
        if self.config.normalize:
            W_spillover = self._normalize_spillover(W_spillover)

        return W_spillover

    def _similarity_spillover(self) -> torch.Tensor:
        """
        Similarity-based spillover: neurons with similar weight patterns are neighbors.

        Biological rationale:
            Neurons with similar receptive fields are often anatomically close,
            so they experience spillover from each other.

        Algorithm:
            1. Normalize weight vectors
            2. Compute cosine similarity
            3. Threshold to find similar neurons
            4. Weight spillover by similarity

        Returns:
            Spillover weight matrix [n_post, n_pre]
        """
        # Normalize weight vectors for cosine similarity
        W_norm = self.W_direct / (self.W_direct.norm(dim=1, keepdim=True) + 1e-8)

        # Compute pairwise cosine similarity
        similarity = W_norm @ W_norm.T  # [n_post, n_post]

        # Threshold: only spillover between sufficiently similar neurons
        similarity = torch.relu(similarity - self.config.similarity_threshold)

        # Remove self-connections
        similarity.fill_diagonal_(0)

        # Check if any neighbors exist after thresholding
        if similarity.sum() == 0:
            # No similar neurons found - return zero spillover
            return torch.zeros_like(self.W_direct)

        # Normalize similarity scores
        similarity_sum = similarity.sum(dim=1, keepdim=True)
        similarity = similarity / (similarity_sum + 1e-8)

        # Spillover = similarity-weighted average of neighbor synapses
        W_spillover = similarity @ self.W_direct * self.config.strength

        if self.config.normalize:
            W_spillover = self._normalize_spillover(W_spillover)

        return W_spillover

    def _lateral_spillover(self) -> torch.Tensor:
        """
        Lateral spillover: simple index-based neighborhood.

        Biological rationale:
            Assumes neuron index ordering reflects spatial proximity
            (e.g., cortical columns, hippocampal layers).

        Algorithm:
            1. Define neighborhood radius
            2. Create banded connectivity matrix
            3. Weight spillover by distance (closer = stronger)

        Returns:
            Spillover weight matrix [n_post, n_pre]
        """
        n_post = self.W_direct.shape[0]

        # Compute pairwise distances in index space
        indices = torch.arange(n_post, device=self.device)
        dist = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()

        # Define neighborhood: within radius, excluding self
        in_radius = (dist > 0) & (dist <= self.config.lateral_radius)

        # Weight by distance (closer = stronger spillover)
        # Linear decay: spillover = strength * (1 - dist/radius)
        lateral_weights = torch.zeros_like(dist, dtype=torch.float32)
        lateral_weights[in_radius] = self.config.strength * (
            1.0 - dist[in_radius].float() / self.config.lateral_radius
        )

        # Normalize each row (total spillover from all neighbors)
        row_sums = lateral_weights.sum(dim=1, keepdim=True)
        lateral_weights = lateral_weights / (row_sums + 1e-8)

        # Apply lateral weights to direct synapses
        W_spillover = lateral_weights @ self.W_direct

        if self.config.normalize:
            W_spillover = self._normalize_spillover(W_spillover)

        return W_spillover

    def _normalize_spillover(self, W_spillover: torch.Tensor) -> torch.Tensor:
        """
        Normalize spillover weights to prevent runaway excitation.

        Ensures that spillover doesn't dramatically increase total synaptic drive.
        Scales spillover so that max(W_effective) ≈ max(W_direct).

        Args:
            W_spillover: Unnormalized spillover weights

        Returns:
            Normalized spillover weights
        """
        # Find maximum effective weight per postsynaptic neuron
        W_effective_max = (self.W_direct.abs() + W_spillover.abs()).max(dim=1, keepdim=True)[0]
        W_direct_max = self.W_direct.abs().max(dim=1, keepdim=True)[0]

        # Scale spillover to keep effective weights in reasonable range
        scale_factor = W_direct_max / (W_effective_max + 1e-8)
        scale_factor = torch.clamp(scale_factor, min=0.5, max=1.0)  # Limit scaling

        return W_spillover * scale_factor

    def get_effective_weights(self) -> torch.Tensor:
        """
        Get effective weight matrix (direct + spillover).

        Returns:
            Effective weights for forward pass [n_post, n_pre]
        """
        return self.W_effective

    def get_spillover_weights(self) -> torch.Tensor:
        """
        Get spillover weight matrix only.

        Returns:
            Spillover weights [n_post, n_pre]
        """
        return self.W_spillover

    def get_spillover_fraction(self) -> float:
        """
        Compute fraction of total weight due to spillover.

        Returns:
            Spillover fraction (0 = no spillover, 1 = all spillover)
        """
        direct_norm = self.W_direct.abs().sum().item()
        spillover_norm = self.W_spillover.abs().sum().item()

        if direct_norm + spillover_norm == 0:
            return 0.0

        return float(spillover_norm / (direct_norm + spillover_norm))

    def update_direct_weights(self, new_weights: torch.Tensor) -> None:
        """
        Update direct weights and recompute spillover.

        Call this after learning updates that modify W_direct.

        Args:
            new_weights: Updated direct synaptic weights [n_post, n_pre]
        """
        self.W_direct = new_weights.to(self.device)

        self.W_spillover = self._build_spillover_weights()
        self.W_effective = self.W_direct + self.W_spillover


def apply_spillover_to_weights(
    weights: torch.Tensor,
    config: SpilloverConfig,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convenience function to apply spillover augmentation to weight matrix.

    Args:
        weights: Direct synaptic weights [n_post, n_pre]
        config: Spillover configuration
        device: Device for computation

    Returns:
        Effective weights (direct + spillover) [n_post, n_pre]
    """
    spillover = SpilloverTransmission(weights, config, device)
    return spillover.get_effective_weights()
