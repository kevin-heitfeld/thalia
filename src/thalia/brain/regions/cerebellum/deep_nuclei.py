"""
Deep Cerebellar Nuclei (DCN) - Final Output Stage.

The DCN are the sole output of the cerebellum, integrating:
1. **Purkinje inhibition**: Sculpts tonic excitation
2. **Mossy fiber excitation**: Direct excitatory drive (collaterals)
3. **Climbing fiber excitation**: Error signal reinforcement (collaterals)

Biological organization:
- Fastigial nucleus: Midline, vestibular and axial control
- Interposed nuclei: Intermediate, limb movements
- Dentate nucleus: Lateral, cognitive and planning

Function:
- Provides tonic excitation (Purkinje sculpts this)
- Timing and amplitude control of motor output
- Projects to thalamus, brainstem, spinal cord
"""

from __future__ import annotations

import torch
import torch.nn as nn

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import WeightInitializer
from thalia.constants import DEFAULT_DT_MS
from thalia.units import ConductanceTensor


class DeepCerebellarNuclei(nn.Module):
    """Deep cerebellar nuclei (DCN) - Final cerebellar output stage.

    Architecture:
    - Receives inhibition from Purkinje cells
    - Receives excitation from mossy fiber collaterals
    - Receives excitation from climbing fiber collaterals
    - Generates motor commands (output to thalamus, brainstem)

    Function:
    - Integrates Purkinje inhibition with excitatory drive
    - Provides tonic excitation (Purkinje sculpts this)
    - Timing and amplitude control of motor output
    """

    def __init__(
        self,
        n_output: int,
        n_purkinje: int,
        n_mossy: int,
        device: str = "cpu",
        dt_ms: float = DEFAULT_DT_MS,
    ):
        """Initialize deep cerebellar nuclei.

        Args:
            n_output: Number of DCN output neurons
            n_purkinje: Number of Purkinje cells providing inhibition
            n_mossy: Number of mossy fibers providing excitation
            device: Torch device
            dt_ms: Simulation timestep
        """
        super().__init__()
        self.n_output = n_output
        self.n_purkinje = n_purkinje
        self.n_mossy = n_mossy
        self.device = device
        self.dt_ms = dt_ms

        # Purkinje → DCN (inhibitory, convergent)
        # Many Purkinje cells converge onto each DCN neuron
        self.purkinje_to_dcn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=n_output,
                n_input=n_purkinje,
                sparsity=0.2,  # 20% connectivity
                weight_scale=1.5,  # Strong inhibition
                device=device,
            ).abs()  # Make positive (inhibitory conductance)
        )

        # Mossy fiber → DCN (excitatory collaterals)
        # Mossy fibers send collaterals to DCN before reaching granule cells
        self.mossy_to_dcn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=n_output,
                n_input=n_mossy,
                sparsity=0.1,  # 10% connectivity
                weight_scale=0.8,  # Moderate excitation
                device=device,
            ).abs()  # Make positive (excitatory)
        )

        # DCN neurons (tonic firing, modulated by inhibition)
        # DCN neurons are spontaneously active (tonic firing ~40-60 Hz)
        # Use NORMALIZED units (threshold=1.0 scale) NOT absolute millivolts
        dcn_config = ConductanceLIFConfig(
            v_threshold=1.0,  # Standard normalized threshold
            v_reset=0.0,  # Reset to rest
            v_rest=0.0,  # Resting potential (normalized)
            E_L=0.0,  # Leak reversal (normalized)
            E_E=3.0,  # Excitatory reversal (normalized, above threshold)
            E_I=-0.5,  # Inhibitory reversal (normalized, hyperpolarizing)
            g_L=0.10,  # Moderate leak conductance
            tau_mem=20.0,  # ms, moderate integration
            tau_E=4.0,  # ms, AMPA kinetics (biological range 2-5ms)
            tau_I=10.0,  # ms, GABA_A kinetics (biological range 5-10ms)
            tau_ref=12.0,  # ms, refractory period (max ~83 Hz ceiling, allows biological 40-60 Hz range)
            noise_std=0.15,  # Higher membrane noise to break synchrony (normalized units)
            device=device,
        )
        self.dcn_neurons = ConductanceLIF(
            n_neurons=n_output,
            config=dcn_config,
            device=device,
        )

        # HETEROGENEOUS tonic excitation to break pathological synchrony
        # Real DCN neurons have different baseline excitabilities
        # Use Gaussian distribution for moderate spontaneous firing (~40-60 Hz)
        # DCN neurons are tonically active, but Purkinje inhibition sculpts output
        self.tonic_excitation = torch.normal(
            mean=0.005,  # Ultra-minimal baseline drive (rely more on mossy excitation + noise)
            std=0.003,  # Small heterogeneity
            size=(n_output,),
            device=device,
        ).clamp(
            min=0.002, max=0.010
        )  # Very low range - DCN activity primarily from mossy collaterals, not tonic

        # Initialize neurons with heterogeneous membrane potentials
        # Prevents synchronized oscillations from identical initial conditions
        v_init = torch.normal(
            mean=0.0,  # Around rest (normalized units)
            std=0.1,  # Moderate spread
            size=(n_output,),
            device=device,
        ).clamp(
            min=-0.2, max=0.5
        )  # Subthreshold range (normalized units)

        # Set initial membrane potentials
        self._initial_v = v_init

        # Initialize DCN neuron membrane potentials with heterogeneous values
        # This prevents all neurons starting at same potential and synchronizing
        self.dcn_neurons.membrane = self._initial_v.clone()

    def forward(
        self,
        purkinje_spikes: torch.Tensor,  # [n_purkinje]
        mossy_spikes: torch.Tensor,  # [n_mossy]
    ) -> torch.Tensor:
        """Generate motor output from Purkinje inhibition and mossy excitation.

        Args:
            purkinje_spikes: Purkinje cell output [n_purkinje]
            mossy_spikes: Mossy fiber activity [n_mossy]

        Returns:
            Motor command spikes [n_output]
        """
        # Purkinje inhibition (conductance, positive)
        purkinje_inh = torch.mv(self.purkinje_to_dcn, purkinje_spikes.float())

        # Mossy fiber excitation (collateral)
        mossy_exc = torch.mv(self.mossy_to_dcn, mossy_spikes.float())

        # Add heterogeneous tonic excitation (DCN neurons are intrinsically active)
        # Note: ConductanceLIF now has built-in noise_std, so no manual noise needed
        total_exc = mossy_exc + self.tonic_excitation

        # DCN spiking (excitation vs inhibition)
        # Note: g_inh_input uses conductance-based shunting inhibition
        # Purkinje cells sculpt DCN output through strong inhibition
        # With low tonic drive (0.02), Purkinje inhibition has proper control
        dcn_spikes, _ = self.dcn_neurons(
            g_exc_input=ConductanceTensor(total_exc),
            g_inh_input=ConductanceTensor(purkinje_inh),  # Full strength - Purkinje sculpts output
        )

        return dcn_spikes
