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

Author: Thalia Project
Date: December 17, 2025
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from thalia.components.synapses.weight_init import WeightInitializer
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


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
        dt_ms: float = 1.0,
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
                scale=1.5,     # Strong inhibition
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
                scale=0.8,     # Moderate excitation
                device=device,
            ).abs()  # Make positive (excitatory)
        )

        # DCN neurons (tonic firing, modulated by inhibition)
        # DCN neurons are spontaneously active (tonic firing)
        dcn_config = ConductanceLIFConfig(
            v_threshold=-50.0,  # mV, relatively low threshold
            v_reset=-60.0,      # mV
            tau_mem=10.0,       # ms, moderate integration
            tau_E=2.0,          # ms, excitatory conductance decay
            tau_I=5.0,          # ms, inhibitory conductance decay (slower)
            dt_ms=dt_ms,        # Timestep in milliseconds
        )
        self.dcn_neurons = ConductanceLIF(
            n_neurons=n_output,
            config=dcn_config,
        )
        self.dcn_neurons.to(device)

        # Tonic excitation (DCN neurons have intrinsic activity)
        # Reduced to allow inhibition to have visible effect
        self.tonic_excitation = 0.2  # Baseline drive (lower for sensitivity to inhibition)

    def forward(
        self,
        purkinje_spikes: torch.Tensor,  # [n_purkinje]
        mossy_spikes: torch.Tensor,     # [n_mossy]
    ) -> torch.Tensor:
        """Generate motor output from Purkinje inhibition and mossy excitation.

        Args:
            purkinje_spikes: Purkinje cell output [n_purkinje]
            mossy_spikes: Mossy fiber activity [n_mossy]

        Returns:
            Motor command spikes [n_output]
        """
        # Purkinje inhibition (conductance, positive)
        purkinje_inh = torch.mv(
            self.purkinje_to_dcn,
            purkinje_spikes.float()
        )

        # Mossy fiber excitation (collateral)
        mossy_exc = torch.mv(
            self.mossy_to_dcn,
            mossy_spikes.float()
        )

        # Add tonic excitation (DCN neurons are intrinsically active)
        total_exc = mossy_exc + self.tonic_excitation

        # DCN spiking (excitation vs inhibition)
        # Note: g_inh_input uses conductance-based shunting inhibition
        dcn_spikes, _ = self.dcn_neurons(
            g_exc_input=total_exc,
            g_inh_input=purkinje_inh * 2.0,  # Scale up inhibition for visible effect
        )

        return dcn_spikes

    def reset_state(self) -> None:
        """Reset DCN state."""
        self.dcn_neurons.reset_state()

    def get_state(self) -> dict:
        """Get DCN state for checkpointing."""
        return {
            "purkinje_to_dcn": self.purkinje_to_dcn.data.clone(),
            "mossy_to_dcn": self.mossy_to_dcn.data.clone(),
            "dcn_neurons": self.dcn_neurons.get_state(),
        }

    def load_state(self, state: dict) -> None:
        """Load DCN state from checkpoint."""
        self.purkinje_to_dcn.data.copy_(state["purkinje_to_dcn"])
        self.mossy_to_dcn.data.copy_(state["mossy_to_dcn"])
        self.dcn_neurons.load_state(state["dcn_neurons"])

    def get_full_state(self) -> dict:
        """Get full DCN state (alias for get_state)."""
        return self.get_state()

    def grow_output(self, n_new: int) -> None:
        """Grow DCN output neurons (alias for grow).

        Args:
            n_new: Number of new output neurons to add
        """
        return self.grow(n_new)

    def grow(self, n_new: int) -> None:
        """Grow DCN neuron population.

        Args:
            n_new: Number of new DCN neurons to add
        """
        old_n_output = self.n_output
        self.n_output = old_n_output + n_new

        # Expand Purkinje→DCN weights
        new_purkinje_weights = WeightInitializer.sparse_random(
            n_output=n_new,
            n_input=self.n_purkinje,
            sparsity=0.2,
            scale=1.5,
            device=self.device,
        ).abs()
        self.purkinje_to_dcn = nn.Parameter(
            torch.cat([self.purkinje_to_dcn.data, new_purkinje_weights], dim=0)
        )

        # Expand mossy→DCN weights
        new_mossy_weights = WeightInitializer.sparse_random(
            n_output=n_new,
            n_input=self.n_mossy,
            sparsity=0.1,
            scale=0.8,
            device=self.device,
        ).abs()
        self.mossy_to_dcn = nn.Parameter(
            torch.cat([self.mossy_to_dcn.data, new_mossy_weights], dim=0)
        )

        # Expand DCN neurons by recreating with new size
        # ConductanceLIF doesn't have grow() - recreate with expanded size
        old_config = self.dcn_neurons.config
        self.dcn_neurons = ConductanceLIF(
            n_neurons=self.n_output,
            config=old_config,
        )
        self.dcn_neurons.to(self.device)


__all__ = ["DeepCerebellarNuclei"]
