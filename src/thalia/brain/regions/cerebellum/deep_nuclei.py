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
        # DCN neurons are spontaneously active (tonic firing)
        # Key: Use biologically realistic reversal potentials to prevent pathological oscillations
        dcn_config = ConductanceLIFConfig(
            v_threshold=-50.0,  # mV, relatively low threshold for spontaneous activity
            v_reset=-60.0,  # mV, moderate hyperpolarization (not too deep)
            E_L=-60.0,  # mV, leak/resting potential (match reset for stability)
            E_E=-45.0,  # mV, excitatory reversal (just above threshold, not 0!)
            E_I=-80.0,  # mV, inhibitory reversal (hyperpolarizing)
            g_L=0.05,  # Leak conductance (moderate)
            tau_mem=20.0,  # ms, longer integration for irregular firing
            tau_E=3.0,  # ms, excitatory conductance decay
            tau_I=8.0,  # ms, inhibitory conductance decay (slower for sustained effect)
            noise_std=0.5,  # mV, membrane noise to break synchrony
        )
        self.dcn_neurons = ConductanceLIF(
            n_neurons=n_output,
            config=dcn_config,
            device=device,
        )

        # HETEROGENEOUS tonic excitation to break pathological synchrony
        # Real DCN neurons have different baseline excitabilities
        # Use Gaussian distribution with moderate drive
        self.tonic_excitation = torch.normal(
            mean=0.08,  # Reduced from 0.12 for more stable dynamics
            std=0.02,  # Reduced from 0.03 for tighter distribution
            size=(n_output,),
            device=device,
        ).clamp(
            min=0.03, max=0.15
        )  # Narrower biological range

        # Initialize neurons with heterogeneous membrane potentials
        # Prevents synchronized oscillations from identical initial conditions
        v_init = torch.normal(
            mean=-60.0,  # Around reset potential
            std=5.0,  # Moderate spread
            size=(n_output,),
            device=device,
        ).clamp(
            min=-70.0, max=-50.0
        )  # Keep subthreshold

        # Set initial membrane potentials
        self._initial_v = v_init

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
        # With E_I = -80 mV and proper reversal potentials, inhibition now has clear effect
        dcn_spikes, _ = self.dcn_neurons(
            g_exc_input=total_exc,
            g_inh_input=purkinje_inh * 2.0,  # Moderate scaling for biological effect
        )

        return dcn_spikes
