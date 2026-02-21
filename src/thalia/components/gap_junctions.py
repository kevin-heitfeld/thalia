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

from thalia.units import GapJunctionConductance, GapJunctionReversal, VoltageTensor


@dataclass
class GapJunctionConfig:
    """Configuration for gap junction coupling.

    Attributes:
        coupling_strength: Gap junction conductance relative to leak (g_gap/g_L)
            Biological range: 0.05-0.3 (Galarreta & Hestrin 2001)
        connectivity_threshold: Minimum shared input for coupling (0-1)
            Higher = sparser networks, lower = denser coupling
        max_neighbors: Maximum coupled neighbors per neuron (computational limit)
            Biological: 4-12 for interneurons (Galarreta & Hestrin 1999)
        interneuron_only: Only couple inhibitory neurons (biological default)
    """

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
        # Ensure weights are positive conductances for biological realism
        weights = weights.abs().to(self.device)

        # Binarize weights to find active connections
        W_binary = (weights > 1e-6).float()

        # Compute shared input matrix: how many inputs do neurons i and j share?
        # shared[i,j] = number of common presynaptic partners
        shared_inputs = W_binary @ W_binary.T  # [n_neurons, n_neurons]

        # Normalize to [0, 1]: shared[i,j] / max_possible_shared
        max_inputs = W_binary.sum(dim=1)  # [n_neurons]
        normalizer = torch.minimum(max_inputs.unsqueeze(1), max_inputs.unsqueeze(0))  # [n_neurons, n_neurons]
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
                k_value = min(self.config.max_neighbors, (scores >= 0).sum().item())
                k = int(k_value)
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

    def __call__(self, *args, **kwds):
        assert False, f"{self.__class__.__name__} instances should not be called directly. Use forward() instead."
        return super().__call__(*args, **kwds)

    def forward(self, voltages: VoltageTensor) -> tuple[GapJunctionConductance, GapJunctionReversal]:
        """
        Compute gap junction coupling as (conductance, effective_reversal).

        Implements bidirectional electrical coupling using dynamic reversal:
            I_gap[i] = Σ_j g_gap[i,j] * (V[j] - V[i])
                     = g_total[i] * (E_eff[i] - V[i])

        Where:
            g_total[i] = Σ_j g_gap[i,j]  (total gap conductance)
            E_eff[i] = Σ_j [g_gap[i,j] * V[j]] / g_total[i]  (weighted avg of neighbor voltages)

        This formulation allows gap junctions to integrate cleanly into
        conductance-based neuron dynamics without mixing currents and conductances.

        **Physics**: The effective reversal potential equals the weighted average
        of neighbor voltages, making gap junctions behave like a conductance
        with a dynamic (time-varying) reversal potential.

        Args:
            voltages: Membrane voltages [n_neurons]

        Returns:
            g_gap_total: Total gap junction conductance per neuron [n_neurons]
            E_gap_effective: Effective reversal potential per neuron [n_neurons]
                (weighted average of neighbor voltages)
        """
        coupling_matrix_tensor: torch.Tensor = self.coupling_matrix

        # Total gap junction conductance per neuron
        g_gap_total = coupling_matrix_tensor.sum(dim=1)  # [n_neurons]

        # Weighted sum of neighbor voltages
        neighbor_weighted_v = coupling_matrix_tensor @ voltages  # [n_neurons]

        # Effective reversal potential = weighted average of neighbor voltages
        # Initialize with zeros for neurons with no gap junctions
        E_gap_effective = torch.zeros_like(g_gap_total)

        # Only compute where g_gap_total > 0 (avoid division by zero)
        mask = g_gap_total > 1e-6
        if mask.any():
            E_gap_effective[mask] = neighbor_weighted_v[mask] / g_gap_total[mask]

        return GapJunctionConductance(g_gap_total), GapJunctionReversal(E_gap_effective)
