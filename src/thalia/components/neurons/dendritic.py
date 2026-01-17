"""
Dendritic nonlinearities for biologically realistic computation.

This module implements dendritic branches with NMDA-mediated nonlinearities,
enabling neurons to perform local computations before signals reach the soma.

Key features:
- NMDA spikes: Local regenerative events when clustered inputs exceed threshold
- Plateau potentials: Sustained depolarization from NMDA activation
- Branch-specific integration: Each branch is a computational subunit
- Supralinear summation: Co-active inputs on same branch are amplified

Biological basis:
- Dendrites are not passive cables - they perform nonlinear computation
- NMDA receptors provide voltage-dependent amplification
- Clustered synaptic inputs trigger local dendritic spikes
- This enables feature binding and increases memory capacity

References:
- Larkum et al. (2009): A cellular mechanism for cortical associations
- Major et al. (2013): Active properties of neocortical pyramidal neuron dendrites
- Poirazi et al. (2003): Pyramidal neuron as two-layer neural network
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class DendriticBranchConfig:
    """Configuration for a dendritic branch with NMDA nonlinearity.

    Attributes:
        nmda_threshold: Local threshold for NMDA spike initiation (normalized units)
            When summed input on a branch exceeds this, NMDA amplification occurs.
            Typical value: 0.3-0.5 of somatic threshold.

        nmda_gain: Supralinear amplification factor for NMDA spikes
            Inputs exceeding threshold are multiplied by this factor.
            Experimental data suggests 3-5x amplification.

        plateau_tau_ms: Time constant for NMDA plateau decay (ms)
            NMDA plateaus can last 50-200ms, enabling temporal integration.

        tau_syn_ms: Synaptic conductance time constant (ms)
            Controls how spikes accumulate over time on the branch.
            ~10-20ms allows temporal summation of sequential inputs.
            This is the key mechanism for NMDA clustering with sequential patterns.

        saturation_level: Maximum branch output (prevents runaway excitation)
            Branch output is soft-clamped to this level.

        subthreshold_attenuation: Attenuation factor for weak inputs
            Inputs below threshold are slightly attenuated (passive cable loss).
            Set to 1.0 for no attenuation, <1.0 for realistic cable filtering.

        branch_coupling: How strongly branch output couples to soma
            1.0 = full coupling, <1.0 = attenuated coupling (distal branches)
    """

    nmda_threshold: float = 0.3  # Local spike threshold
    nmda_gain: float = 3.0  # Supralinear amplification
    plateau_tau_ms: float = 80.0  # NMDA plateau duration
    tau_syn_ms: float = 15.0  # Synaptic conductance decay
    saturation_level: float = 2.0  # Maximum branch output
    subthreshold_attenuation: float = 0.8  # Cable filtering for weak inputs
    branch_coupling: float = 1.0  # Soma coupling strength
    dt_ms: float = 0.1  # Simulation timestep (ms)

    @property
    def plateau_decay(self) -> float:
        """Decay factor for NMDA plateau per timestep."""
        return float(torch.exp(torch.tensor(-self.dt_ms / self.plateau_tau_ms)).item())

    @property
    def syn_decay(self) -> float:
        """Decay factor for synaptic conductance per timestep."""
        return float(torch.exp(torch.tensor(-self.dt_ms / self.tau_syn_ms)).item())


@dataclass
class DendriticNeuronConfig:
    """Configuration for a neuron with dendritic branches.

    Attributes:
        n_branches: Number of dendritic branches per neuron
            Pyramidal neurons typically have 5-20 major branches.
            More branches = more computational power but more parameters.

        inputs_per_branch: Number of synaptic inputs per branch
            Each branch receives a subset of total inputs.
            Total inputs = n_branches × inputs_per_branch

        branch_config: Configuration for individual branches

        soma_config: Configuration for somatic compartment (ConductanceLIF)

        input_routing: How inputs are routed to branches
            "fixed": Predetermined mapping (inputs i*k to (i+1)*k go to branch i)
            "random": Random assignment at initialization
            "learned": Weights determine routing (soft assignment)
    """

    n_branches: int = 4
    inputs_per_branch: int = 50
    branch_config: Optional[DendriticBranchConfig] = None
    soma_config: Optional[ConductanceLIFConfig] = None
    input_routing: str = "fixed"  # "fixed", "random", or "learned"

    def __post_init__(self):
        if self.branch_config is None:
            self.branch_config = DendriticBranchConfig()
        if self.soma_config is None:
            self.soma_config = ConductanceLIFConfig()

    @property
    def total_inputs(self) -> int:
        """Total number of inputs to the neuron."""
        return self.n_branches * self.inputs_per_branch


class DendriticBranch(nn.Module):
    """A single dendritic branch with NMDA-mediated nonlinearity.

    Each branch integrates its inputs locally and applies a nonlinear
    transformation based on NMDA receptor dynamics:

    - Subthreshold: Linear or slightly attenuated integration
    - Suprathreshold: NMDA spike with supralinear amplification
    - Plateau: Sustained activity due to NMDA kinetics

    The output represents the effective conductance contribution
    to the soma from this branch.

    Example:
        >>> branch = DendriticBranch(n_inputs=50)
        >>> branch.reset_state(batch_size=1)
        >>>
        >>> # Weak scattered inputs - sublinear
        >>> weak_output = branch(weak_inputs)
        >>>
        >>> # Strong clustered inputs - NMDA spike
        >>> strong_output = branch(strong_clustered_inputs)  # >> linear sum
    """

    def __init__(
        self,
        n_inputs: int,
        config: Optional[DendriticBranchConfig] = None,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.config = config or DendriticBranchConfig()

        # Synaptic weights for this branch
        # Initialize with small positive values (excitatory)
        self.weights = nn.Parameter(
            torch.rand(n_inputs, requires_grad=False) * 0.1 + 0.05, requires_grad=False
        )

        # Register constants
        self.register_buffer(
            "plateau_decay", torch.tensor(self.config.plateau_decay, dtype=torch.float32)
        )

        # State: NMDA plateau potential (persists across timesteps)
        self.plateau: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Reset branch state."""
        device = self.weights.device
        # Internal state is scalar (no batch dimension per ADR-005)
        self.plateau = torch.tensor(0.0, device=device)

    def forward(
        self,
        inputs: torch.Tensor,  # [n_inputs] (1D per ADR-005)
        membrane_potential: Optional[torch.Tensor] = None,  # Scalar, for voltage-dependent NMDA
    ) -> torch.Tensor:
        """Compute branch output with NMDA nonlinearity.

        ADR-005 compliant: accepts 1D input, returns scalar output.
        Uses internal batch dimension only for computation efficiency.

        Args:
            inputs: Input conductances to this branch, shape [n_inputs]
            membrane_potential: Somatic membrane potential (scalar) for NMDA gating
                If provided, NMDA gain is voltage-dependent (stronger when depolarized)

        Returns:
            Branch output conductance (scalar)
        """
        # Initialize state if needed
        if self.plateau is None:
            self.reset_state()

        # ADR-005: Expect 1D input
        assert (
            inputs.dim() == 1 and inputs.shape[0] == self.n_inputs
        ), f"DendriticBranch.forward: Expected 1D input [n_inputs={self.n_inputs}], got shape {inputs.shape}"

        # Compute weighted sum: [n_inputs] @ [n_inputs] → scalar
        linear_sum = torch.dot(inputs, self.weights.clamp(min=0))

        # === NMDA Nonlinearity ===
        threshold = self.config.nmda_threshold
        gain = self.config.nmda_gain
        saturation = self.config.saturation_level
        attenuation = self.config.subthreshold_attenuation

        # Voltage-dependent NMDA modulation (optional)
        if membrane_potential is not None:
            # NMDA is more effective when soma is depolarized
            # This creates cooperative integration: branch helps soma, soma helps branch
            # Use sigmoid for smooth transition
            voltage_gate = torch.sigmoid(membrane_potential * 2)  # 0.5 at rest, ~1 at threshold
            effective_gain = 1 + (gain - 1) * voltage_gate
        else:
            effective_gain = gain  # type: ignore[assignment]

        # Smooth nonlinearity using sigmoid for transition
        # This avoids discontinuities while preserving the key dynamics
        above_threshold = torch.sigmoid((linear_sum - threshold) * 10)

        # Subthreshold: attenuated, Suprathreshold: amplified
        subthreshold_output = linear_sum * attenuation
        suprathreshold_output = linear_sum * effective_gain

        # Blend based on how far above threshold
        instantaneous_output = (
            subthreshold_output * (1 - above_threshold) + suprathreshold_output * above_threshold
        )

        # Apply saturation (soft clamp)
        instantaneous_output = saturation * torch.tanh(instantaneous_output / saturation)

        # === NMDA Plateau Dynamics ===
        # Plateau decays but is boosted by suprathreshold activity
        self.plateau = self.plateau * self.plateau_decay  # type: ignore[operator]

        # Suprathreshold activity triggers/maintains plateau
        plateau_boost = above_threshold * instantaneous_output * 0.5
        self.plateau = torch.maximum(self.plateau, plateau_boost)

        # Output is max of instantaneous and plateau (plateau sustains activity)
        branch_output = torch.maximum(instantaneous_output, self.plateau)

        # Apply branch-soma coupling
        branch_output = branch_output * self.config.branch_coupling

        # ADR-005: Return scalar
        return branch_output

    def get_state(self) -> dict:
        """Get current branch state."""
        return {
            "plateau": self.plateau.clone() if self.plateau is not None else None,
        }


class DendriticNeuron(nn.Module):
    """A neuron with multiple dendritic branches and a somatic compartment.

    This implements a two-layer computational model:
    1. Dendritic layer: Multiple branches perform local nonlinear integration
    2. Somatic layer: Conductance-based LIF integrates branch outputs

    This architecture enables:
    - Feature binding: Clustered inputs on same branch → strong response
    - Increased capacity: Each branch can store different patterns
    - AND-OR computation: AND within branches, OR across branches
    - Noise robustness: Local thresholds reject random weak inputs

    Example:
        >>> config = DendriticNeuronConfig(
        ...     n_branches=4,
        ...     inputs_per_branch=50,
        ... )
        >>> neuron = DendriticNeuron(n_neurons=10, config=config)
        >>> neuron.reset_state(batch_size=1)
        >>>
        >>> # Process inputs through dendrites
        >>> inputs = torch.rand(1, 200)  # 4 branches × 50 inputs
        >>> spikes, membrane = neuron(inputs)
    """

    def __init__(
        self,
        n_neurons: int,
        config: Optional[DendriticNeuronConfig] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or DendriticNeuronConfig()

        # Create branches for each neuron
        # Shape: (n_neurons, n_branches) worth of branches
        self.n_branches = self.config.n_branches
        self.inputs_per_branch = self.config.inputs_per_branch

        # Branch weights: (n_neurons, n_branches, inputs_per_branch)
        # Each neuron has independent branch weights
        self.branch_weights = nn.Parameter(
            torch.rand(n_neurons, self.n_branches, self.inputs_per_branch) * 0.1 + 0.05
        )

        # Cached clamped weights (optimization: avoid recomputing every forward)
        # Updated via invalidate_weight_cache() after learning
        self._cached_weights: Optional[torch.Tensor] = None

        # Register branch config constants
        branch_cfg = self.config.branch_config
        self.register_buffer(
            "plateau_decay", torch.tensor(branch_cfg.plateau_decay, dtype=torch.float32)  # type: ignore[union-attr]
        )
        self.register_buffer(
            "nmda_threshold", torch.tensor(branch_cfg.nmda_threshold, dtype=torch.float32)  # type: ignore[union-attr]
        )
        self.register_buffer("nmda_gain", torch.tensor(branch_cfg.nmda_gain, dtype=torch.float32))  # type: ignore[union-attr]
        self.register_buffer(
            "saturation_level", torch.tensor(branch_cfg.saturation_level, dtype=torch.float32)  # type: ignore[union-attr]
        )
        self.register_buffer(
            "subthreshold_attenuation",
            torch.tensor(branch_cfg.subthreshold_attenuation, dtype=torch.float32),  # type: ignore[union-attr]
        )
        self.register_buffer(
            "branch_coupling", torch.tensor(branch_cfg.branch_coupling, dtype=torch.float32)  # type: ignore[union-attr]
        )

        # Pre-compute gain difference for NMDA blend (optimization)
        self.register_buffer(
            "_gain_minus_atten",
            torch.tensor(
                branch_cfg.nmda_gain - branch_cfg.subthreshold_attenuation, dtype=torch.float32  # type: ignore[union-attr]
            ),
        )

        # Synaptic conductance decay (for temporal integration)
        self.register_buffer("syn_decay", torch.tensor(branch_cfg.syn_decay, dtype=torch.float32))  # type: ignore[union-attr]

        # Somatic compartment (conductance-based LIF)
        self.soma = ConductanceLIF(n_neurons, self.config.soma_config)

        # Branch plateau state: (batch, n_neurons, n_branches)
        self.branch_plateaus: Optional[torch.Tensor] = None

        # Synaptic conductance state: (batch, n_neurons, n_branches)
        # Accumulates weighted spikes over time with tau_syn decay
        self.branch_g_syn: Optional[torch.Tensor] = None

        # Input routing for "random" mode
        if self.config.input_routing == "random":
            # Create random permutation for each neuron
            routing = torch.stack(
                [torch.randperm(self.config.total_inputs) for _ in range(n_neurons)]
            )
            self.register_buffer("input_routing_indices", routing)
        else:
            self.input_routing_indices = None

    def invalidate_weight_cache(self) -> None:
        """Invalidate the cached clamped weights. Call after modifying weights."""
        self._cached_weights = None

    def _get_clamped_weights(self) -> torch.Tensor:
        """Get clamped weights, using cache if available."""
        if self._cached_weights is None:
            self._cached_weights = self.branch_weights.clamp(min=0)
        return self._cached_weights

    def reset_state(self) -> None:
        """Reset all state (branches and soma).

        Internal implementation uses batch dimension for vectorization
        across neurons, but public API is ADR-005 compliant (1D).
        """
        device = self.branch_weights.device
        # Internal batch=1 for efficient vectorization across neurons
        batch_size = 1

        # Reset branch plateaus: [1, n_neurons, n_branches]
        self.branch_plateaus = torch.zeros(
            batch_size, self.n_neurons, self.n_branches, device=device
        )

        # Reset synaptic conductance (for temporal integration): [1, n_neurons, n_branches]
        self.branch_g_syn = torch.zeros(batch_size, self.n_neurons, self.n_branches, device=device)

        # Reset soma (ConductanceLIF already ADR-005 compliant)
        self.soma.reset_state()

    @property
    def membrane(self) -> Optional[torch.Tensor]:
        """Membrane potential (proxy to soma.membrane for compatibility)."""
        return self.soma.membrane

    @membrane.setter
    def membrane(self, value: torch.Tensor) -> None:
        """Set membrane potential (proxy to soma.membrane for compatibility)."""
        self.soma.membrane = value

    def _route_inputs_to_branches(
        self,
        inputs: torch.Tensor,  # [total_inputs] externally, [1, total_inputs] internally
    ) -> torch.Tensor:
        """Route inputs to branches based on routing mode.

        Internal method: uses batch dimension for vectorization.

        Args:
            inputs: Input tensor, shape [1, total_inputs] (internal batch representation)

        Returns:
            Routed inputs, shape [1, n_neurons, n_branches, inputs_per_branch]
        """
        batch_size = inputs.shape[0]

        if self.config.input_routing == "fixed":
            # Fixed routing: inputs 0:k go to branch 0, k:2k to branch 1, etc.
            # Reshape: (batch, total) → (batch, n_branches, inputs_per_branch)
            routed = inputs.view(batch_size, self.n_branches, self.inputs_per_branch)
            # Expand to all neurons (same routing for all neurons in fixed mode)
            routed = routed.unsqueeze(1).expand(
                batch_size, self.n_neurons, self.n_branches, self.inputs_per_branch
            )
        elif self.config.input_routing == "random":
            # Random routing: use pre-computed permutation indices
            # (batch, total) → gather per neuron → (batch, n_neurons, total)
            assert inputs is not None, "inputs cannot be None for random routing"
            assert (
                self.input_routing_indices is not None
            ), "input_routing_indices must be set for random mode"
            inputs_tensor = cast(torch.Tensor, inputs)
            expanded_inputs = inputs_tensor.unsqueeze(1).expand(batch_size, self.n_neurons, -1)
            indices = self.input_routing_indices.unsqueeze(0).expand(batch_size, -1, -1)
            routed = torch.gather(expanded_inputs, dim=2, index=indices)
            routed = routed.view(
                batch_size, self.n_neurons, self.n_branches, self.inputs_per_branch
            )
        else:  # "learned" - weights determine soft routing
            # For learned routing, inputs go to all branches weighted by attention
            # This is handled differently - all inputs available to all branches
            routed = (
                inputs.unsqueeze(1)
                .unsqueeze(2)
                .expand(batch_size, self.n_neurons, self.n_branches, -1)
            )
            # Truncate or pad to inputs_per_branch
            if routed.shape[-1] > self.inputs_per_branch:
                routed = routed[..., : self.inputs_per_branch]
            elif routed.shape[-1] < self.inputs_per_branch:
                padding = torch.zeros(
                    *routed.shape[:-1],
                    self.inputs_per_branch - routed.shape[-1],
                    device=inputs.device,
                )
                routed = torch.cat([routed, padding], dim=-1)

        return routed

    def _compute_branch_outputs(
        self,
        inputs: torch.Tensor,  # [1, total_inputs] (internal batch representation)
        membrane_potential: Optional[torch.Tensor] = None,  # [n_neurons] (1D per ADR-005)
    ) -> torch.Tensor:
        """Compute outputs from all branches with NMDA nonlinearity.

        Internal method: uses batch dimension for vectorization.

        This uses synaptic conductance dynamics for temporal integration:
        - Incoming spikes are weighted and added to g_syn
        - g_syn decays with tau_syn (~15ms)
        - NMDA threshold is compared against accumulated g_syn
        - This allows sequential spikes to sum and trigger NMDA

        Args:
            inputs: Input spike conductances, shape [1, total_inputs]
            membrane_potential: Somatic membrane for voltage-dependent NMDA, shape [n_neurons]

        Returns:
            Branch outputs, shape [1, n_neurons, n_branches]
        """
        # Initialize state if needed
        if self.branch_plateaus is None or self.branch_g_syn is None:
            self.reset_state()

        # Route inputs to branches if needed
        if inputs.dim() == 2:
            branch_inputs = self._route_inputs_to_branches(inputs)
        else:
            branch_inputs = inputs

        # Compute weighted sum of INSTANTANEOUS spikes per branch
        # branch_inputs: (batch, n_neurons, n_branches, inputs_per_branch)
        # Use cached clamped weights (updated only when invalidate_weight_cache() is called)
        weights_clamped = self._get_clamped_weights()
        instantaneous_input = (branch_inputs * weights_clamped).sum(dim=-1)
        # instantaneous_input: (batch, n_neurons, n_branches)

        # === Synaptic Conductance Dynamics (KEY for temporal integration) ===
        # g_syn decays and accumulates new spikes (in-place for efficiency)
        # This allows multiple spikes over time to sum and cross NMDA threshold
        self.branch_g_syn = self.branch_g_syn.mul(self.syn_decay).add_(instantaneous_input)  # type: ignore[union-attr, arg-type]

        # Use accumulated g_syn for NMDA threshold comparison
        g_syn = self.branch_g_syn

        # === NMDA Nonlinearity (Optimized) ===
        # Smooth threshold sigmoid
        above_threshold = torch.sigmoid((g_syn - self.nmda_threshold) * 10)  # type: ignore[operator]

        # Fused NMDA blend: linear_sum * (atten + (gain - atten) * above_threshold)
        # This combines subthreshold attenuation and suprathreshold gain in one operation
        if membrane_potential is not None:
            # Voltage-dependent NMDA: gain varies with membrane potential
            # membrane_potential is [n_neurons] (1D per ADR-005), need to add batch dim and branch dim
            # [n_neurons] → [1, n_neurons, 1] for broadcasting
            voltage_gate = torch.sigmoid(membrane_potential.unsqueeze(0).unsqueeze(-1) * 2)
            # effective_gain = 1 + (nmda_gain - 1) * voltage_gate
            # gain_diff = effective_gain - attenuation = (1 - atten) + (nmda_gain - 1) * voltage_gate
            effective_gain_diff = (1.0 - self.subthreshold_attenuation) + (  # type: ignore[operator]
                self.nmda_gain - 1.0  # type: ignore[operator]
            ) * voltage_gate
            blend_factor = self.subthreshold_attenuation + effective_gain_diff * above_threshold  # type: ignore[operator]
        else:
            # Static gain: use pre-computed (gain - attenuation)
            blend_factor = self.subthreshold_attenuation + self._gain_minus_atten * above_threshold  # type: ignore[operator]

        instantaneous_output = g_syn * blend_factor  # type: ignore[operator]

        # Saturation (soft clamp)
        instantaneous_output = self.saturation_level * torch.tanh(  # type: ignore[operator]
            instantaneous_output / self.saturation_level  # type: ignore[operator]
        )

        # === NMDA Plateau Dynamics (in-place where possible) ===
        self.branch_plateaus.mul_(self.plateau_decay)  # type: ignore[union-attr, arg-type]
        plateau_boost = above_threshold * instantaneous_output * 0.5
        self.branch_plateaus = torch.maximum(self.branch_plateaus, plateau_boost)  # type: ignore[arg-type]

        # Output is max of instantaneous and plateau, with coupling applied
        branch_output = torch.maximum(instantaneous_output, self.branch_plateaus)

        # Apply coupling (skip if 1.0)
        if self.branch_coupling != 1.0:
            branch_output = branch_output * self.branch_coupling  # type: ignore[operator]

        return branch_output

    def forward(
        self,
        inputs: torch.Tensor,  # [total_inputs] per ADR-005
        g_inh: Optional[torch.Tensor] = None,  # [n_neurons] per ADR-005
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process inputs through dendrites then soma.

        ADR-005 compliant: accepts 1D inputs, returns 1D outputs.
        Uses internal batch dimension only for vectorization efficiency.

        Args:
            inputs: Input conductances, shape [total_inputs]
            g_inh: Inhibitory conductance to soma, shape [n_neurons]

        Returns:
            spikes: Binary spike tensor, shape [n_neurons]
            membrane: Membrane potentials, shape [n_neurons]
        """
        # ADR-005: Validate 1D inputs
        assert (
            inputs.dim() == 1 and inputs.shape[0] == self.config.total_inputs
        ), f"DendriticNeuron.forward: Expected 1D input [total_inputs={self.config.total_inputs}], got shape {inputs.shape}"
        if g_inh is not None:
            assert (
                g_inh.dim() == 1 and g_inh.shape[0] == self.n_neurons
            ), f"DendriticNeuron.forward: Expected 1D g_inh [n_neurons={self.n_neurons}], got shape {g_inh.shape}"

        # Add internal batch dimension for vectorization: [total_inputs] → [1, total_inputs]
        inputs_batched = inputs.unsqueeze(0)
        # Note: g_inh could be batched if needed, but currently unused
        # Get current membrane potential for voltage-dependent NMDA
        # soma.membrane is already 1D [n_neurons] per ADR-005
        membrane_potential = self.soma.membrane

        # Compute branch outputs (internal batched computation)
        branch_outputs = self._compute_branch_outputs(inputs_batched, membrane_potential)
        # branch_outputs shape: [1, n_neurons, n_branches]

        # Sum across branches → somatic excitatory conductance
        g_exc_soma_batched = branch_outputs.sum(dim=-1)  # [1, n_neurons]
        g_exc_soma = g_exc_soma_batched.squeeze(0)  # [n_neurons] - ADR-005 compliant

        # Process through conductance-based soma (already ADR-005 compliant)
        spikes, membrane = self.soma(g_exc_soma, g_inh)

        # Return 1D tensors per ADR-005
        return spikes, membrane

    def forward_with_branch_info(
        self,
        inputs: torch.Tensor,  # [total_inputs] per ADR-005
        g_inh: Optional[torch.Tensor] = None,  # [n_neurons] per ADR-005
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass that also returns branch outputs for analysis.

        ADR-005 compliant: accepts 1D inputs, returns 1D spikes/membrane,
        and 2D branch outputs [n_neurons, n_branches].

        Returns:
            spikes: [n_neurons]
            membrane: [n_neurons]
            branch_outputs: [n_neurons, n_branches] (internal batch dim squeezed)
        """
        # Add batch dimension for internal computation
        inputs_batched = inputs.unsqueeze(0)
        # Note: g_inh unused in current implementation

        membrane_potential = self.soma.membrane
        branch_outputs_batched = self._compute_branch_outputs(inputs_batched, membrane_potential)
        # branch_outputs_batched: [1, n_neurons, n_branches]

        g_exc_soma_batched = branch_outputs_batched.sum(dim=-1)
        g_exc_soma = g_exc_soma_batched.squeeze(0)

        spikes, membrane = self.soma(g_exc_soma, g_inh)

        # Squeeze batch dimension from branch_outputs for return
        branch_outputs = branch_outputs_batched.squeeze(0)  # [n_neurons, n_branches]

        return spikes, membrane, branch_outputs

    def get_state(self) -> dict:
        """Get current neuron state."""
        soma_state = self.soma.get_state()
        return {
            "branch_plateaus": (
                self.branch_plateaus.clone() if self.branch_plateaus is not None else None
            ),
            **soma_state,
        }

    def __repr__(self) -> str:
        return (
            f"DendriticNeuron(n={self.n_neurons}, "
            f"branches={self.n_branches}, "
            f"inputs_per_branch={self.inputs_per_branch}, "
            f"total_inputs={self.config.total_inputs})"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_branch_selectivity(
    neuron: DendriticNeuron,
    input_patterns: torch.Tensor,  # [n_patterns, total_inputs]
) -> torch.Tensor:
    """Compute which branches respond to which input patterns.

    Useful for analyzing learned representations.

    Args:
        neuron: DendriticNeuron instance
        input_patterns: Test patterns, shape [n_patterns, total_inputs]

    Returns:
        Selectivity matrix, shape [n_patterns, n_neurons, n_branches]
    """
    neuron.reset_state()

    selectivity = []
    for pattern in input_patterns:
        # pattern is 1D [total_inputs] per ADR-005
        _, _, branch_outputs = neuron.forward_with_branch_info(pattern)
        # branch_outputs is [n_neurons, n_branches]
        selectivity.append(branch_outputs)

    return torch.stack(selectivity)  # [n_patterns, n_neurons, n_branches]


def create_clustered_input(
    n_inputs: int,
    n_active: int,
    cluster_branch: int,
    n_branches: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create an input pattern with activity clustered on one branch.

    Useful for testing dendritic nonlinearity: clustered inputs should
    produce stronger responses than scattered inputs with same total activity.

    Args:
        n_inputs: Total number of inputs
        n_active: Number of active inputs
        cluster_branch: Which branch to cluster activity on (0 to n_branches-1)
        n_branches: Number of branches
        device: Device for tensor creation

    Returns:
        Input pattern, shape (n_inputs,)
    """
    inputs_per_branch = n_inputs // n_branches
    pattern = torch.zeros(n_inputs, device=device)

    start_idx = cluster_branch * inputs_per_branch
    end_idx = min(start_idx + n_active, start_idx + inputs_per_branch)
    pattern[start_idx:end_idx] = 1.0

    return pattern


def create_scattered_input(
    n_inputs: int,
    n_active: int,
    n_branches: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create an input pattern with activity scattered across branches.

    Same total activity as clustered, but distributed across all branches.
    Should produce weaker response due to lack of NMDA spike.

    Args:
        n_inputs: Total number of inputs
        n_active: Number of active inputs per branch
        n_branches: Number of branches
        device: Device to create tensor on

    Returns:
        Input pattern, shape (n_inputs,)
    """
    inputs_per_branch = n_inputs // n_branches
    pattern = torch.zeros(n_inputs, device=device)

    for branch in range(n_branches):
        start_idx = branch * inputs_per_branch
        # Activate n_active / n_branches inputs per branch
        n_per_branch = max(1, n_active // n_branches)
        pattern[start_idx : start_idx + n_per_branch] = 1.0

    return pattern
