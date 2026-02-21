"""Tonic Dopamine Neuron: Specialized Neuron Type with Autonomous Firing and Burst/Pause Dynamics.

**UPDATED**: Now uses Izhikevich model instead of conductance-based LIF.
**Reason**: Conductance-based LIF has fundamental shunting issues that prevent
tonic inhibition from coexisting with intrinsic pacemaker currents. The Izhikevich
model uses current-based (not conductance-based) inputs, eliminating this problem.

Dopamine neurons in the VTA exhibit characteristic firing patterns that encode
reward prediction errors (RPE) through two distinct modes:

1. **Tonic Firing** (4-5 Hz baseline):
   - Intrinsic/autonomous oscillation driven by tonic depolarizing current
   - Represents background motivation/mood state
   - Provides baseline dopamine tone

2. **Phasic Modulation**:
   - **Burst** (15-20 Hz): Positive RPE (unexpected reward)
   - **Pause** (<1 Hz): Negative RPE (expected reward omitted)
   - Duration: 100-200 ms

This specialized neuron type is used exclusively by the VTA region to encode
reward prediction errors through burst and pause dynamics.
"""

from __future__ import annotations

import torch

from .izhikevich_neuron import IzhikevichNeuronConfig, IzhikevichNeuron


class TonicDopamineNeuron(IzhikevichNeuron):
    """Specialized Izhikevich neuron tuned for tonic dopamine firing dynamics.

    Configured for:
    - Tonic (autonomous) firing at 4-5 Hz (via i_tonic)
    - Robust to tonic inhibition (no shunting issues!)
    - Burst mode on positive RPE (15-20 Hz)
    - Pause mode on negative RPE (<1 Hz)
    - Spike-frequency adaptation (via recovery variable)
    """

    def __init__(self, n_neurons: int, device: torch.device = torch.device("cpu")):
        """Initialize tonic dopamine neurons.

        Args:
            n_neurons: Number of dopamine neurons
            device: Torch device for computation
        """
        # Configuration optimized for DA neuron pacemaking
        config = IzhikevichNeuronConfig(
            device=device,
            a=0.02,  # Slow recovery for adaptation
            b=0.25,  # Moderate voltage sensitivity
            c=-55.0,  # Depolarized reset for easy re-excitation
            d=0.05,  # Small recovery increment for regular firing
            v_threshold=30.0,
            v_rest=-65.0,
            excitatory_current_scale=100.0,
            inhibitory_current_scale=80.0,
            i_tonic=5.0,  # Tuned for ~4-5 Hz baseline
            noise_std=1.5,  # Moderate noise for stochastic firing
        )

        super().__init__(n_neurons, config)
