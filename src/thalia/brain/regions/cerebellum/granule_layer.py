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
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import WeightInitializer
from thalia.constants import DEFAULT_DT_MS
from thalia.units import ConductanceTensor


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
        dt_ms: float = DEFAULT_DT_MS,
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
            v_reset=-65.0,  # mV
            tau_mem=5.0,  # ms, faster than pyramidal (5ms vs 10-30ms)
            tau_E=2.5,  # ms, fast AMPA-like (biological minimum ~2-3ms)
            tau_I=6.0,  # ms, fast GABA_A (biological range 5-10ms)
        )
        self.granule_neurons = ConductanceLIF(
            n_neurons=self.n_granule,
            config=granule_config,
            device=device,
        )

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
        # TODO: Future enhancement - add feedforward inhibition from Golgi cells for more realistic dynamics
        parallel_fiber_spikes, _ = self.granule_neurons(
            g_exc_input=ConductanceTensor(g_exc),
            g_inh_input=None,
        )

        # Enforce sparsity (top-k activation based on excitation)
        # Always select the k most excited neurons that spiked
        k = int(self.n_granule * self.sparsity)
        n_spiking = parallel_fiber_spikes.sum().item()

        if n_spiking > k:
            # More neurons spiked than target, select top-k by excitation
            # Only consider neurons that actually spiked
            spiking_mask = parallel_fiber_spikes.bool()
            g_exc_spiking = g_exc.clone()
            g_exc_spiking[~spiking_mask] = -float("inf")  # Exclude non-spiking
            _, top_k_idx = torch.topk(g_exc_spiking, k)
            sparse_spikes = torch.zeros_like(parallel_fiber_spikes, dtype=torch.bool)
            sparse_spikes[top_k_idx] = True
            parallel_fiber_spikes = sparse_spikes
        # else: Fewer than k neurons spiked naturally, keep them all

        return parallel_fiber_spikes
