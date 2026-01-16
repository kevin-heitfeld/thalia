"""
Granule Cell Layer - Sparse Expansion and Temporal Delay Lines.

The granule cell layer is the input stage of the cerebellar cortex, providing:
1. **Sparse expansion**: 4-5× more granule cells than mossy fiber inputs
2. **Sparse coding**: Only 2-5% of granule cells active at any time
3. **Pattern separation**: Similar inputs activate different granule patterns
4. **Temporal delays**: Parallel fibers at different distances = delay lines

Biological facts:
- Most numerous neurons in the entire brain (~50 billion in humans!)
- Each granule receives input from 4-5 mossy fibers
- Produces parallel fibers that run perpendicular to Purkinje dendrites
- Sparse firing (2-5% active) maximizes representational capacity

Author: Thalia Project
Date: December 17, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from thalia.components.synapses.weight_init import WeightInitializer
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class GranuleLayerState:
    """State for granule cell layer.

    Attributes:
        mossy_to_granule: Synaptic weights from mossy fibers to granule cells
        granule_neurons: State dict from granule neuron model (ConductanceLIF)
    """
    mossy_to_granule: torch.Tensor
    granule_neurons: Dict[str, Any]  # State from ConductanceLIF.get_state()


class GranuleCellLayer(nn.Module):
    """Granule cell layer - Sparse coding and expansion.

    Key properties:
    - Most numerous neurons in the entire brain (~50 billion!)
    - Sparse firing (2-5% active at any time)
    - Expansion layer: 4-5× more granule cells than mossy fibers
    - Parallel fibers: Long axons that contact many Purkinje cells

    Biological function:
    - Pattern separation (like hippocampus DG)
    - Temporal delay lines (parallel fibers at different distances)
    - Combinatorial expansion (increase representational capacity)
    """

    def __init__(
        self,
        n_mossy_fibers: int,
        expansion_factor: float = 4.0,
        sparsity: float = 0.03,  # 3% active (biological)
        device: str = "cpu",
        dt_ms: float = 1.0,
    ):
        """Initialize granule cell layer.

        Args:
            n_mossy_fibers: Number of mossy fiber inputs
            expansion_factor: Ratio of granule cells to mossy fibers (typically 4-5)
            sparsity: Fraction of granule cells active (0.02-0.05 biological)
            device: Torch device
            dt_ms: Simulation timestep in milliseconds
        """
        super().__init__()
        self.n_input = n_mossy_fibers
        self.n_granule = int(n_mossy_fibers * expansion_factor)
        self.sparsity = sparsity
        self.device = device
        self.dt_ms = dt_ms

        # Mossy fiber → Granule cell synapses
        # Sparse random connectivity (each granule receives from 4-5 mossy fibers)
        self.mossy_to_granule = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_granule,
                n_input=n_mossy_fibers,
                sparsity=0.05,  # 5% connectivity (each granule gets ~4-5 inputs)
                device=device,
            )
        )

        # Granule cell neurons (simple LIF, small time constant)
        # Granule cells are small, fast-spiking neurons
        granule_config = ConductanceLIFConfig(
            v_threshold=-50.0,  # mV, more excitable than pyramidal
            v_reset=-65.0,      # mV
            tau_mem=5.0,        # ms, faster than pyramidal (5ms vs 10-30ms)
            tau_E=1.0,          # ms, fast excitatory conductance decay
            tau_I=2.0,          # ms, fast inhibitory conductance decay
            dt_ms=dt_ms,        # Timestep in milliseconds
        )
        self.granule_neurons = ConductanceLIF(
            n_neurons=self.n_granule,
            config=granule_config,
        )
        self.granule_neurons.to(device)

    @property
    def weights(self) -> nn.Parameter:
        """Alias for mossy_to_granule weights (test compatibility)."""
        return self.mossy_to_granule

    @property
    def neurons(self):
        """Alias for granule_neurons (test compatibility)."""
        return self.granule_neurons

    def forward(
        self,
        mossy_fiber_spikes: torch.Tensor,
        mf_efficacy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process mossy fiber input through granule cells.

        Args:
            mossy_fiber_spikes: Mossy fiber activity [n_mossy]
            mf_efficacy: Optional STP efficacy modulation [n_mossy, n_granule]
                        Applied to mossy_to_granule weights (per-synapse modulation)

        Returns:
            Parallel fiber spikes [n_granule] (sparse, ~3% active)
        """
        # Mossy fiber → Granule cell
        # Apply STP efficacy to weights if provided (per-synapse modulation)
        if mf_efficacy is not None:
            # mf_efficacy is [n_mossy, n_granule], weights are [n_granule, n_mossy]
            # Modulate: W_eff = W * efficacy.T
            effective_weights = self.mossy_to_granule * mf_efficacy.T
            g_exc = torch.mv(effective_weights, mossy_fiber_spikes.float())
        else:
            g_exc = torch.mv(self.mossy_to_granule, mossy_fiber_spikes.float())

        # Granule cell spiking (minimal inhibition - granule layer is excitatory)
        parallel_fiber_spikes, _ = self.granule_neurons(g_exc, None)

        # Enforce sparsity (top-k activation based on excitation)
        # Always select the k most excited neurons that spiked
        k = int(self.n_granule * self.sparsity)
        n_spiking = parallel_fiber_spikes.sum().item()

        if n_spiking > k:
            # More neurons spiked than target, select top-k by excitation
            # Only consider neurons that actually spiked
            spiking_mask = parallel_fiber_spikes.bool()
            g_exc_spiking = g_exc.clone()
            g_exc_spiking[~spiking_mask] = -float('inf')  # Exclude non-spiking
            _, top_k_idx = torch.topk(g_exc_spiking, k)
            sparse_spikes = torch.zeros_like(parallel_fiber_spikes, dtype=torch.bool)
            sparse_spikes[top_k_idx] = True
            parallel_fiber_spikes = sparse_spikes
        # else: Fewer than k neurons spiked naturally, keep them all

        return parallel_fiber_spikes

    def reset_state(self) -> None:
        """Reset granule cell states."""
        self.granule_neurons.reset_state()

    def get_state(self) -> GranuleLayerState:
        """Get granule layer state for checkpointing."""
        return GranuleLayerState(
            mossy_to_granule=self.mossy_to_granule.data.clone(),
            granule_neurons=self.granule_neurons.get_state(),
        )

    def load_state(self, state: GranuleLayerState) -> None:
        """Load granule layer state from checkpoint."""
        # Detect target device - prefer self.device if it's set, else infer from state
        if hasattr(self, 'mossy_to_granule') and self.mossy_to_granule is not None:
            # Module already exists, use its current device
            target_device = self.mossy_to_granule.device
        else:
            # Module doesn't exist yet, use state's device
            target_device = state.mossy_to_granule.device

        # Update device attribute
        self.device = str(target_device)

        # Move module to target device (this moves parameters and buffers)
        self.to(target_device)

        # Now copy state (ensure source is moved to target device)
        self.mossy_to_granule.data.copy_(state.mossy_to_granule.to(target_device))
        self.granule_neurons.load_state(state.granule_neurons)

    def get_full_state(self) -> GranuleLayerState:
        """Get full granule layer state (alias for get_state)."""
        return self.get_state()

    def load_full_state(self, state: GranuleLayerState) -> None:
        """Load granule layer state from checkpoint (alias for load_state)."""
        self.load_state(state)

    def grow(self, n_new: int) -> None:
        """Grow granule cell population.

        Args:
            n_new: Number of new granule cells to add
        """
        old_n_granule = self.n_granule
        self.n_granule = old_n_granule + n_new

        # Expand mossy→granule weights
        new_weights = WeightInitializer.sparse_random(
            n_output=n_new,
            n_input=self.n_input,
            sparsity=0.05,
            device=self.device,
        )
        self.mossy_to_granule = nn.Parameter(
            torch.cat([self.mossy_to_granule.data, new_weights], dim=0)
        )

        # Expand granule neurons
        self.granule_neurons.grow(n_new)

    def grow_input(self, n_new: int) -> None:
        """Grow input dimension (mossy fibers).

        Args:
            n_new: Number of new mossy fiber inputs to add
        """
        old_n_input = self.n_input
        self.n_input = old_n_input + n_new

        # Add columns to mossy→granule weights
        new_weights = WeightInitializer.sparse_random(
            n_output=self.n_granule,
            n_input=n_new,
            sparsity=0.05,
            device=self.device,
        )
        self.mossy_to_granule = nn.Parameter(
            torch.cat([self.mossy_to_granule.data, new_weights], dim=1)
        )


__all__ = ["GranuleCellLayer"]
