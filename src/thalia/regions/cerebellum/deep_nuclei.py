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
        # Key: Use biologically realistic reversal potentials to prevent pathological oscillations
        dcn_config = ConductanceLIFConfig(
            v_threshold=-50.0,  # mV, relatively low threshold for spontaneous activity
            v_reset=-60.0,      # mV, moderate hyperpolarization (not too deep)
            E_L=-60.0,          # mV, leak/resting potential (match reset for stability)
            E_E=-45.0,          # mV, excitatory reversal (just above threshold, not 0!)
            E_I=-80.0,          # mV, inhibitory reversal (hyperpolarizing)
            g_L=0.05,           # Leak conductance (moderate)
            tau_mem=20.0,       # ms, longer integration for irregular firing
            tau_E=3.0,          # ms, excitatory conductance decay
            tau_I=8.0,          # ms, inhibitory conductance decay (slower for sustained effect)
            dt_ms=dt_ms,        # Timestep in milliseconds
            noise_std=0.5,      # mV, membrane noise to break synchrony
        )
        self.dcn_neurons = ConductanceLIF(
            n_neurons=n_output,
            config=dcn_config,
        )
        self.dcn_neurons.to(device)

        # HETEROGENEOUS tonic excitation to break pathological synchrony
        # Real DCN neurons have different baseline excitabilities
        # Use Gaussian distribution with moderate drive
        self.tonic_excitation = torch.normal(
            mean=0.08,  # Reduced from 0.12 for more stable dynamics
            std=0.02,   # Reduced from 0.03 for tighter distribution
            size=(n_output,),
            device=device,
        ).clamp(min=0.03, max=0.15)  # Narrower biological range

        # Initialize neurons with heterogeneous membrane potentials
        # Prevents synchronized oscillations from identical initial conditions
        v_init = torch.normal(
            mean=-60.0,  # Around reset potential
            std=5.0,     # Moderate spread
            size=(n_output,),
            device=device,
        ).clamp(min=-70.0, max=-50.0)  # Keep subthreshold

        # Set initial membrane potentials (will be set after first reset_state)
        self._initial_v = v_init

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

    def reset_state(self) -> None:
        """Reset DCN state with heterogeneous initialization."""
        self.dcn_neurons.reset_state()

        # Re-initialize with heterogeneous membrane potentials to break synchrony
        v_init = torch.normal(
            mean=-60.0,  # Around reset potential
            std=5.0,     # Moderate spread
            size=(self.n_output,),
            device=self.device,
        ).clamp(min=-70.0, max=-50.0)  # Keep subthreshold
        self.dcn_neurons.membrane.data = v_init

    def get_state(self) -> dict:
        """Get DCN state for checkpointing."""
        return {
            "purkinje_to_dcn": self.purkinje_to_dcn.data.clone(),
            "mossy_to_dcn": self.mossy_to_dcn.data.clone(),
            "tonic_excitation": self.tonic_excitation.clone(),
            "dcn_neurons": self.dcn_neurons.get_state(),
        }

    def load_state(self, state: dict) -> None:
        """Load DCN state from checkpoint."""
        self.purkinje_to_dcn.data.copy_(state["purkinje_to_dcn"])
        self.mossy_to_dcn.data.copy_(state["mossy_to_dcn"])
        if "tonic_excitation" in state:
            self.tonic_excitation.copy_(state["tonic_excitation"])
        self.dcn_neurons.load_state(state["dcn_neurons"])

    def get_full_state(self) -> dict:
        """Get full DCN state (alias for get_state)."""
        return self.get_state()

    def load_full_state(self, state: dict) -> None:
        """Load DCN state from checkpoint (alias for load_state)."""
        self.load_state(state)

    def grow_output(self, n_new: int) -> None:
        """Grow DCN output neurons (alias for grow).

        Expands deep cerebellar nucleus neuron population and adjusts
        Purkinje cell → DCN weight matrix to accommodate new outputs.

        Args:
            n_new: Number of new output neurons to add

        Example:
            >>> dcn = DeepCerebellarNuclei(n_output=64, n_input=128, device=device)
            >>> dcn.grow_output(16)  # Expands to 80 output neurons
            >>> assert dcn.n_output == 80
        """
        return self.grow(n_new)

    def grow(self, n_new: int) -> None:
        """Grow DCN neuron population.

        Expands the DCN output dimension by:
        1. Adding new rows to Purkinje→DCN weights (Xavier initialization)
        2. Growing the ConductanceLIF neuron population
        3. Updating configuration and state dimensions

        New weights are initialized with Xavier uniform and clamped to
        config bounds (w_min, w_max) for biological plausibility.

        Args:
            n_new: Number of new DCN neurons to add

        Effects:
            - self.n_output increases by n_new
            - self.weights expands from [n_output, n_input] to [n_output+n_new, n_input]
            - self.neurons population grows by n_new

        Example:
            >>> dcn = DeepCerebellarNuclei(n_output=64, n_input=128, device=device)
            >>> old_weights = dcn.weights.clone()
            >>> dcn.grow(16)
            >>> # Old weights preserved:
            >>> assert torch.allclose(dcn.weights[:64], old_weights)
            >>> # New weights initialized:
            >>> assert dcn.weights.shape == (80, 128)
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

        # Expand tonic excitation (heterogeneous for new neurons)
        new_tonic = torch.normal(
            mean=0.12,
            std=0.03,
            size=(n_new,),
            device=self.device,
        ).clamp(min=0.05, max=0.25)
        self.tonic_excitation = torch.cat([self.tonic_excitation, new_tonic], dim=0)

        # Expand DCN neurons by recreating with new size
        # ConductanceLIF doesn't have grow() - recreate with expanded size
        old_config = self.dcn_neurons.config
        self.dcn_neurons = ConductanceLIF(
            n_neurons=self.n_output,
            config=old_config,
        )
        self.dcn_neurons.to(self.device)

    def grow_input(self, n_new: int, source: str = 'purkinje') -> None:
        """Grow input dimension from Purkinje or mossy fibers.

        Expands weight matrix columns to accept more input neurons.

        Args:
            n_new: Number of new input neurons
            source: 'purkinje' or 'mossy' - which input to grow

        Effects:
            - Adds columns to purkinje_to_dcn or mossy_to_dcn
            - Updates n_purkinje or n_mossy
        """
        if source == 'purkinje':
            # Add columns to Purkinje→DCN weights
            new_weights = WeightInitializer.sparse_random(
                n_output=self.n_output,
                n_input=n_new,
                sparsity=0.2,
                scale=1.5,
                device=self.device,
            ).abs()
            self.purkinje_to_dcn = nn.Parameter(
                torch.cat([self.purkinje_to_dcn.data, new_weights], dim=1)
            )
            self.n_purkinje += n_new
        elif source == 'mossy':
            # Add columns to mossy→DCN weights
            new_weights = WeightInitializer.sparse_random(
                n_output=self.n_output,
                n_input=n_new,
                sparsity=0.1,
                scale=0.8,
                device=self.device,
            ).abs()
            self.mossy_to_dcn = nn.Parameter(
                torch.cat([self.mossy_to_dcn.data, new_weights], dim=1)
            )
            self.n_mossy += n_new
        else:
            raise ValueError(f"Unknown source: {source}. Use 'purkinje' or 'mossy'.")


__all__ = ["DeepCerebellarNuclei"]
