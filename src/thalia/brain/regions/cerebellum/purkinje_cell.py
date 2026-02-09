"""
Enhanced Purkinje Cell Model with Dendritic Computation.

Purkinje cells are the sole output of the cerebellar cortex, featuring:
1. **Complex spikes**: Calcium events triggered by climbing fibers (1-10 Hz)
2. **Simple spikes**: Regular sodium spikes from parallel fiber input (40-100 Hz)
3. **Massive dendritic tree**: ~200,000 parallel fiber inputs per Purkinje cell
4. **Dendritic calcium**: Gates plasticity and spike generation

Biological significance:
- Complex spike = "error detected" (teaching signal)
- Simple spike = regular output (motor command)
- Dendritic calcium compartments enable local learning rules
- Inhibitory output to deep cerebellar nuclei (GABA)

Author: Thalia Project
Date: December 17, 2025
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import WeightInitializer
from thalia.constants import DEFAULT_DT_MS


class EnhancedPurkinjeCell(nn.Module):
    """Enhanced Purkinje cell with dendritic computation.

    Key biological features:
    - Complex spikes (from climbing fiber, 1-10 Hz, calcium events)
    - Simple spikes (from parallel fibers, 40-100 Hz, regular firing)
    - Massive dendritic tree (~200,000 parallel fiber inputs!)
    - Dendritic calcium signals gate plasticity
    - Inhibitory output to deep nuclei
    """

    def __init__(
        self,
        n_parallel_fibers: int,
        n_dendrites: int = 100,
        device: str = "cpu",
        dt_ms: float = DEFAULT_DT_MS,
    ):
        """Initialize enhanced Purkinje cell.

        Args:
            n_parallel_fibers: Number of parallel fiber inputs (typically granule layer size)
            n_dendrites: Number of dendritic compartments (simplified model)
            device: Torch device
            dt_ms: Simulation timestep
        """
        super().__init__()
        self.n_parallel_fibers = n_parallel_fibers
        self.n_dendrites = n_dendrites
        self.device = device
        self.dt_ms = dt_ms

        # Dendritic compartments (simplified 2-compartment model)
        self.dendrite_voltage = torch.zeros(n_dendrites, device=device)
        self.dendrite_calcium = torch.zeros(n_dendrites, device=device)

        # Soma (main cell body) - Purkinje cells are large, complex
        soma_config = ConductanceLIFConfig(
            v_threshold=-55.0,  # mV
            v_reset=-70.0,  # mV
            tau_mem=15.0,  # ms, slower integration for complex computation
            tau_E=2.0,  # ms, excitatory conductance decay
            tau_I=5.0,  # ms, inhibitory conductance decay (strong inhibition)
        )
        self.soma_neurons = ConductanceLIF(
            n_neurons=1,
            config=soma_config,
            device=device,
        )

        # Complex spike state
        self.last_complex_spike_time: int = -1000
        self.complex_spike_refractory_ms: float = 100.0  # ~10 Hz max
        self.timestep: int = 0

        # Dendritic weights (parallel fiber → dendrites)
        # EAGER INITIALIZATION: Weights created immediately
        self.dendritic_weights = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=1,  # Single Purkinje cell
                n_input=n_parallel_fibers,
                sparsity=0.2,  # 20% connectivity (sparse but not ultra-sparse)
                device=device,
            )
        )

    def forward(
        self,
        parallel_fiber_input: torch.Tensor,  # [n_parallel_fibers]
        climbing_fiber_active: bool,  # Binary: error detected or not
    ) -> Tuple[torch.Tensor, bool]:
        """Process inputs and generate simple/complex spikes.

        Args:
            parallel_fiber_input: Parallel fiber spikes [n_parallel_fibers]
            climbing_fiber_active: Whether climbing fiber is active (error signal)

        Returns:
            simple_spikes: Regular output spikes [bool]
            complex_spike: Whether complex spike occurred [bool]
        """
        # Dendritic processing (parallel fiber input to dendrites)
        dendrite_input = torch.mv(self.dendritic_weights, parallel_fiber_input.float())

        # Dendritic voltage integration (leaky)
        self.dendrite_voltage = 0.9 * self.dendrite_voltage + dendrite_input

        # Soma input (sum of dendritic voltages)
        soma_input = self.dendrite_voltage.sum()

        # Complex spike detection
        complex_spike = False
        if climbing_fiber_active:
            time_since_last = self.timestep - self.last_complex_spike_time
            if time_since_last > self.complex_spike_refractory_ms:
                complex_spike = True
                self.last_complex_spike_time = self.timestep
                # Complex spike triggers calcium influx in dendrites
                self.dendrite_calcium += 1.0

        # Calcium decay (tau ~50ms)
        self.dendrite_calcium *= 0.95

        # Simple spikes (regular output)
        # Calcium modulates excitability (high calcium = more excitable)
        calcium_modulation = 1.0 + 0.2 * self.dendrite_calcium.mean()
        simple_spikes, _ = self.soma_neurons(soma_input.unsqueeze(0) * calcium_modulation, None)

        self.timestep += 1

        return simple_spikes.squeeze(), complex_spike

    @property
    def calcium(self) -> torch.Tensor:
        """Get dendritic calcium levels (test compatibility)."""
        return self.dendrite_calcium

    @property
    def pf_synaptic_weights(self) -> torch.Tensor:
        """Get parallel fiber synaptic weights (alias for dendritic_weights)."""
        return self.dendritic_weights
