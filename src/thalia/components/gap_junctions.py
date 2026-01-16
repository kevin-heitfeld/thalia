"""
Gap junction electrical coupling for neuron populations.

Gap junctions are direct electrical connections between neurons via connexin
protein channels. They enable ultra-fast bidirectional current flow (<0.1ms)
and are critical for neural synchronization, especially in interneuron networks.

Key biological features:
- Ultra-fast synchronization (<1ms vs 1-2ms for chemical synapses)
- Bidirectional voltage coupling (current flows both ways)
- Dense in interneuron populations (~80% of gap junctions)
- Critical for gamma oscillations (40-80 Hz) and attention

Implementation approach:
    Since we don't have explicit spatial coordinates, we use FUNCTIONAL
    connectivity to define neighborhoods. Neurons sharing presynaptic inputs
    are assumed to be anatomically close (biologically valid for most circuits).

References:
    - Bennett & Zukin (2004): Electrical coupling and neuronal synchronization
    - Galarreta & Hestrin (2001): Gap junctions in cortical interneurons
    - Connors & Long (2004): Electrical synapses in the mammalian brain
    - Landisman et al. (2002): Gap junctions in thalamic reticular nucleus
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class GapJunctionConfig:
    """Configuration for gap junction coupling.

    Attributes:
        enabled: Whether to apply gap junction coupling
        coupling_strength: Gap junction conductance relative to leak (g_gap/g_L)
            Biological range: 0.05-0.3 (Galarreta & Hestrin 2001)
        connectivity_threshold: Minimum shared input for coupling (0-1)
            Higher = sparser networks, lower = denser coupling
        max_neighbors: Maximum coupled neighbors per neuron (computational limit)
            Biological: 4-12 for interneurons (Galarreta & Hestrin 1999)
        interneuron_only: Only couple inhibitory neurons (biological default)
    """

    enabled: bool = True
    coupling_strength: float = 0.1  # g_gap = 10% of leak conductance
    connectivity_threshold: float = 0.3  # Share ≥30% of inputs for coupling
    max_neighbors: int = 8  # Limit to 8 neighbors for efficiency
    interneuron_only: bool = True


class GapJunctionCoupling(nn.Module):
    """
    Gap junction electrical coupling for neuron populations.

    Implements bidirectional voltage coupling via gap junctions, enabling
    ultra-fast synchronization. Uses functional connectivity (shared inputs)
    to infer spatial neighborhoods without explicit coordinates.

    The coupling adds a term to neuron dynamics:
        I_gap = Σ g_gap * (V_neighbor - V_self)

    This creates attractive dynamics where coupled neurons' voltages converge,
    leading to synchronous firing patterns.

    Usage:
        >>> # Create gap junction network from afferent connectivity
        >>> config = GapJunctionConfig(enabled=True, coupling_strength=0.1)
        >>> gap_junctions = GapJunctionCoupling(
        ...     n_neurons=100,
        ...     afferent_weights=input_weights,  # [n_neurons, n_input]
        ...     config=config,
        ...     device="cpu"
        ... )
        >>> # During neuron forward pass
        >>> coupling_current = gap_junctions(voltages)  # Add to I_total
    """

    def __init__(
        self,
        n_neurons: int,
        afferent_weights: torch.Tensor,
        config: GapJunctionConfig,
        interneuron_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize gap junction coupling network.

        Args:
            n_neurons: Number of neurons in population
            afferent_weights: Input weights [n_neurons, n_input] for computing
                functional connectivity neighborhoods
            config: Gap junction configuration
            interneuron_mask: Boolean mask [n_neurons] for inhibitory neurons
                (if None and interneuron_only=True, assumes all are interneurons)
            device: Device for computation
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config
        self.device = device or afferent_weights.device

        if not config.enabled:
            # Disabled: no coupling
            self.register_buffer(
                "coupling_matrix", torch.zeros(n_neurons, n_neurons, device=self.device)
            )
            return

        # Determine which neurons get gap junctions
        if config.interneuron_only:
            if interneuron_mask is not None:
                self.interneuron_mask = interneuron_mask.to(self.device)
            else:
                # If not provided, assume ALL neurons are interneurons
                self.interneuron_mask = torch.ones(n_neurons, dtype=torch.bool, device=self.device)
        else:
            # Couple all neurons
            self.interneuron_mask = torch.ones(n_neurons, dtype=torch.bool, device=self.device)

        # Build coupling matrix from functional connectivity
        coupling_matrix = self._build_coupling_matrix(afferent_weights)

        # Store as buffer (not a parameter - doesn't get gradient updates)
        self.register_buffer("coupling_matrix", coupling_matrix)

    def _build_coupling_matrix(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Build gap junction coupling matrix from functional connectivity.

        Uses shared presynaptic inputs as proxy for spatial proximity.
        Neurons that receive input from similar sources are likely anatomically
        close and thus candidates for gap junction coupling.

        Args:
            weights: Afferent weights [n_neurons, n_input]

        Returns:
            Coupling matrix [n_neurons, n_neurons] where entry (i,j) is the
            coupling conductance from neuron j to neuron i
        """
        # Ensure weights are on the correct device
        weights = weights.to(self.device)

        # Binarize weights to find active connections
        W_binary = (weights.abs() > 1e-6).float()

        # Compute shared input matrix: how many inputs do neurons i and j share?
        # shared[i,j] = number of common presynaptic partners
        shared_inputs = W_binary @ W_binary.T  # [n_neurons, n_neurons]

        # Normalize to [0, 1]: shared[i,j] / max_possible_shared
        max_inputs = W_binary.sum(dim=1)  # [n_neurons]
        normalizer = torch.minimum(
            max_inputs.unsqueeze(1), max_inputs.unsqueeze(0)
        )  # [n_neurons, n_neurons]
        normalizer = torch.clamp(normalizer, min=1.0)  # Avoid divide-by-zero

        similarity = shared_inputs / normalizer  # [n_neurons, n_neurons]

        # Threshold: only couple neurons with sufficient shared inputs
        coupling_mask = similarity >= self.config.connectivity_threshold

        # Remove self-connections (no gap junctions to self)
        coupling_mask.fill_diagonal_(False)

        # Only couple interneurons
        interneuron_pairs = self.interneuron_mask.unsqueeze(1) & self.interneuron_mask.unsqueeze(0)
        coupling_mask = coupling_mask & interneuron_pairs

        # Limit neighbors per neuron (for computational efficiency)
        # Keep only top-k strongest connections per neuron
        if self.config.max_neighbors is not None:
            coupling_matrix = torch.zeros_like(similarity)
            for i in range(self.n_neurons):
                if not self.interneuron_mask[i]:
                    continue

                # Get similarity scores for neuron i
                scores = similarity[i].clone()
                scores[~coupling_mask[i]] = -1.0  # Mask out non-coupled neurons

                if scores.max() < 0:
                    continue  # No valid neighbors

                # Select top-k neighbors
                k = min(self.config.max_neighbors, (scores >= 0).sum().item())
                if k == 0:
                    continue

                _, top_indices = torch.topk(scores, k)

                # Set coupling strength proportional to similarity
                coupling_matrix[i, top_indices] = (
                    self.config.coupling_strength * scores[top_indices]
                )
        else:
            # No limit: couple all above-threshold pairs
            coupling_matrix = torch.where(
                coupling_mask,
                self.config.coupling_strength * similarity,
                torch.zeros_like(similarity),
            )

        return coupling_matrix

    def forward(self, voltages: torch.Tensor) -> torch.Tensor:
        """
        Compute gap junction coupling current.

        Implements bidirectional electrical coupling:
            I_gap[i] = Σ_j g_gap[i,j] * (V[j] - V[i])

        This current is added to the total input current in neuron dynamics,
        creating attractive voltage dynamics that synchronize coupled neurons.

        Args:
            voltages: Membrane voltages [n_neurons]

        Returns:
            Coupling current [n_neurons] to add to neuron input
        """
        if not self.config.enabled:
            return torch.zeros_like(voltages)

        # Compute voltage differences: V[j] - V[i] for all coupled pairs
        # coupling_matrix[i,j] * voltages[j] gives contribution from j to i
        neighbor_contribution = self.coupling_matrix @ voltages  # [n_neurons]

        # Subtract own voltage scaled by total coupling
        # This ensures I_gap = Σ g[i,j] * (V[j] - V[i])
        total_coupling_per_neuron = self.coupling_matrix.sum(dim=1)  # [n_neurons]
        self_contribution = total_coupling_per_neuron * voltages

        coupling_current = neighbor_contribution - self_contribution

        return coupling_current

    def get_coupling_stats(self) -> dict:
        """
        Get statistics about gap junction network structure.

        Returns:
            Dictionary with network statistics:
                - n_coupled_neurons: Number of neurons with gap junctions
                - n_connections: Total number of gap junction pairs
                - avg_neighbors: Average number of neighbors per coupled neuron
                - coupling_density: Fraction of possible connections present
        """
        if not self.config.enabled:
            return {
                "n_coupled_neurons": 0,
                "n_connections": 0,
                "avg_neighbors": 0.0,
                "coupling_density": 0.0,
            }

        # Count neurons with at least one gap junction
        has_coupling = (self.coupling_matrix.sum(dim=1) > 0) | (
            self.coupling_matrix.sum(dim=0) > 0
        )
        n_coupled = has_coupling.sum().item()

        # Count total connections (undirected, so divide by 2)
        n_connections = (self.coupling_matrix > 0).sum().item() // 2

        # Average neighbors per coupled neuron
        neighbors_per_neuron = (self.coupling_matrix > 0).sum(dim=1).float()
        avg_neighbors = (
            neighbors_per_neuron[has_coupling].mean().item() if n_coupled > 0 else 0.0
        )

        # Coupling density among interneurons
        n_interneurons = self.interneuron_mask.sum().item()
        max_connections = n_interneurons * (n_interneurons - 1) / 2
        coupling_density = n_connections / max_connections if max_connections > 0 else 0.0

        return {
            "n_coupled_neurons": n_coupled,
            "n_connections": n_connections,
            "avg_neighbors": avg_neighbors,
            "coupling_density": coupling_density,
        }

    def __repr__(self) -> str:
        """String representation of gap junction network."""
        if not self.config.enabled:
            return f"GapJunctionCoupling(disabled, {self.n_neurons} neurons)"

        stats = self.get_coupling_stats()
        return (
            f"GapJunctionCoupling("
            f"{stats['n_coupled_neurons']}/{self.n_neurons} coupled, "
            f"{stats['n_connections']} connections, "
            f"g_gap={self.config.coupling_strength:.3f}, "
            f"density={stats['coupling_density']:.2%})"
        )


def create_gap_junction_coupling(
    n_neurons: int,
    afferent_weights: torch.Tensor,
    coupling_strength: float = 0.1,
    connectivity_threshold: float = 0.3,
    max_neighbors: int = 8,
    interneuron_only: bool = True,
    interneuron_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> GapJunctionCoupling:
    """
    Factory function for creating gap junction coupling.

    Convenience function that creates a GapJunctionCoupling module with
    sensible defaults for common use cases.

    Args:
        n_neurons: Number of neurons in population
        afferent_weights: Input weights [n_neurons, n_input]
        coupling_strength: Gap junction conductance (0.05-0.3 biological)
        connectivity_threshold: Minimum shared input for coupling (0-1)
        max_neighbors: Maximum coupled neighbors per neuron
        interneuron_only: Only couple inhibitory neurons
        interneuron_mask: Boolean mask for inhibitory neurons
        device: Device for computation

    Returns:
        Configured GapJunctionCoupling module

    Example:
        >>> # For TRN interneurons
        >>> gap_junctions = create_gap_junction_coupling(
        ...     n_neurons=100,
        ...     afferent_weights=relay_to_trn_weights,
        ...     coupling_strength=0.15,  # Strong coupling for TRN
        ...     connectivity_threshold=0.2,  # Liberal coupling
        ...     max_neighbors=10,
        ...     interneuron_only=True,
        ... )
    """
    config = GapJunctionConfig(
        enabled=True,
        coupling_strength=coupling_strength,
        connectivity_threshold=connectivity_threshold,
        max_neighbors=max_neighbors,
        interneuron_only=interneuron_only,
    )

    return GapJunctionCoupling(
        n_neurons=n_neurons,
        afferent_weights=afferent_weights,
        config=config,
        interneuron_mask=interneuron_mask,
        device=device,
    )


__all__ = [
    "GapJunctionCoupling",
    "GapJunctionConfig",
    "create_gap_junction_coupling",
]
