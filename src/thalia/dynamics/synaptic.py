"""Short-term synaptic plasticity mechanisms.

This module implements biologically realistic short-term plasticity:
- Short-term Depression (STD): vesicle depletion
- Short-term Facilitation (STF): calcium buildup
- NMDA voltage gating: Mg2+ block
- Dendritic saturation: limited integration capacity
- Neuromodulation: dopamine/acetylcholine effects

These operate on fast timescales (ms to seconds) and are distinct from
long-term plasticity (STDP, Hebbian learning) which modifies weights.

References:
- Tsodyks & Markram (1997): STD/STF model
- Jahr & Stevens (1990): NMDA voltage dependence
- Seamans & Yang (2004): Dopamine modulation
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class STPConfig:
    """Configuration for short-term plasticity.
    
    Attributes:
        depression_rate: Fraction of resources depleted per spike (0-1)
        facilitation_rate: Fractional increase in release probability per spike
        recovery_tau_ms: Time constant for resource recovery (ms)
        facilitation_tau_ms: Time constant for facilitation decay (ms)
        baseline_release_prob: Baseline vesicle release probability
    """
    # Depression (vesicle depletion)
    depression_rate: float = 0.2       # 20% depletion per spike
    recovery_tau_ms: float = 200.0     # 200ms recovery time constant
    
    # Facilitation (calcium buildup)
    facilitation_rate: float = 0.1     # 10% increase per spike
    facilitation_tau_ms: float = 50.0  # 50ms decay time constant
    baseline_release_prob: float = 0.5 # Baseline P(release)
    
    # Combined STP mode
    mode: str = "depression"  # "depression", "facilitation", or "both"


class ShortTermPlasticity:
    """Short-term synaptic plasticity (depression and/or facilitation).
    
    Models the Tsodyks-Markram STP dynamics:
    - Depression: Available vesicles (resources) deplete with each spike
    - Facilitation: Release probability increases with recent activity
    
    The effective synaptic strength is: w_eff = w * resources * release_prob
    
    Example:
        >>> stp = ShortTermPlasticity(n_pre=20, n_post=10, config=STPConfig())
        >>> stp.to(device)
        >>> 
        >>> # On each timestep with spikes:
        >>> effective_weights = stp.modulate_weights(weights, pre_spikes)
        >>> stp.update(pre_spikes, dt=0.1)
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[STPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config or STPConfig()
        self.device = device or torch.device("cpu")
        
        # Resources per synapse (1.0 = full, depletes with use)
        self.resources: torch.Tensor = torch.ones(n_post, n_pre, device=self.device)
        
        # Facilitation factor per synapse (1.0 = baseline, increases with use)
        self.facilitation: torch.Tensor = torch.ones(n_post, n_pre, device=self.device)
        
        # Precompute decay factors
        self._recovery_decay: Optional[float] = None
        self._facilitation_decay: Optional[float] = None
    
    def to(self, device: torch.device) -> "ShortTermPlasticity":
        """Move to device."""
        self.device = device
        self.resources = self.resources.to(device)
        self.facilitation = self.facilitation.to(device)
        return self
    
    def reset(self) -> None:
        """Reset to baseline state."""
        self.resources.fill_(1.0)
        self.facilitation.fill_(1.0)
    
    def _ensure_decay_factors(self, dt: float) -> None:
        """Precompute decay factors for given dt."""
        if self._recovery_decay is None:
            self._recovery_decay = 1.0 - dt / self.config.recovery_tau_ms
            self._facilitation_decay = 1.0 - dt / self.config.facilitation_tau_ms
    
    def modulate_weights(
        self,
        weights: torch.Tensor,
        pre_spikes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply STP modulation to weights.
        
        Args:
            weights: Base weight matrix (n_post, n_pre)
            pre_spikes: Optional presynaptic spikes for selective modulation
            
        Returns:
            Modulated weights incorporating STP effects
        """
        if self.config.mode == "depression":
            return weights * self.resources
        elif self.config.mode == "facilitation":
            release_prob = self.config.baseline_release_prob * self.facilitation
            return weights * release_prob.clamp(0, 1)
        else:  # both
            release_prob = self.config.baseline_release_prob * self.facilitation
            return weights * self.resources * release_prob.clamp(0, 1)
    
    def update(self, pre_spikes: torch.Tensor, dt: float = 0.1) -> None:
        """Update STP state based on presynaptic activity.
        
        Args:
            pre_spikes: Presynaptic spike tensor (batch, n_pre) or (n_pre,)
            dt: Timestep in ms
        """
        self._ensure_decay_factors(dt)
        
        # Ensure proper shape
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(0)
        
        # Broadcast spikes to all postsynaptic neurons: (batch, n_pre) -> (n_post, n_pre)
        spike_mask = pre_spikes.squeeze(0).unsqueeze(0).expand(self.n_post, -1)
        
        # Depression: deplete resources where presynaptic neuron spiked
        if self.config.mode in ("depression", "both"):
            depletion = spike_mask * self.config.depression_rate * self.resources
            self.resources = self.resources - depletion
            # Recovery toward 1.0
            self.resources = self.resources + (1.0 - self.resources) * (1 - self._recovery_decay)
        
        # Facilitation: increase release probability where presynaptic neuron spiked
        if self.config.mode in ("facilitation", "both"):
            increase = spike_mask * self.config.facilitation_rate
            self.facilitation = self.facilitation + increase
            # Decay toward 1.0
            self.facilitation = 1.0 + (self.facilitation - 1.0) * self._facilitation_decay


@dataclass 
class NMDAConfig:
    """Configuration for NMDA receptor dynamics.
    
    Attributes:
        mg_concentration: Extracellular Mg2+ concentration (mM)
        voltage_slope: Steepness of voltage dependence (mV)
        half_block_voltage: Voltage at which 50% block occurs (mV)
        nmda_fraction: Fraction of current through NMDA (vs AMPA)
    """
    mg_concentration: float = 1.0      # mM (physiological)
    voltage_slope: float = 0.062      # /mV (from Jahr & Stevens 1990)
    half_block_voltage: float = -40.0  # mV
    nmda_fraction: float = 0.3        # 30% NMDA, 70% AMPA


class NMDAGating:
    """NMDA receptor voltage-dependent gating.
    
    Models the Mg2+ block of NMDA receptors:
    - At resting potential (~-70mV): NMDA channels blocked
    - With depolarization: block relieved, current flows
    - Creates coincidence detection at the receptor level
    
    The gating factor g(V) = 1 / (1 + [Mg2+] * exp(-γV))
    
    Example:
        >>> nmda = NMDAGating(config=NMDAConfig())
        >>> 
        >>> # Get NMDA gating factor based on membrane potential
        >>> gate = nmda.compute_gate(membrane_potential)  # (batch, n_neurons)
        >>> 
        >>> # Apply to input current
        >>> effective_current = current * (1 - nmda_fraction + nmda_fraction * gate)
    """
    
    def __init__(self, config: Optional[NMDAConfig] = None):
        self.config = config or NMDAConfig()
    
    def compute_gate(self, membrane_potential: torch.Tensor) -> torch.Tensor:
        """Compute NMDA gating factor based on membrane potential.
        
        Args:
            membrane_potential: Membrane voltage tensor (any shape)
            
        Returns:
            Gating factor 0-1 (same shape as input)
        """
        # Jahr & Stevens (1990) NMDA voltage dependence
        # g(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))
        # We use normalized voltages, so scale appropriately
        
        # Convert normalized potential to mV-equivalent
        # Assuming v_threshold=1.0 corresponds to ~-55mV and rest is ~-70mV
        # So 0 → -70mV, 1.0 → -55mV (15mV range)
        v_mv = -70.0 + membrane_potential * 15.0
        
        exponent = -self.config.voltage_slope * v_mv
        block_factor = self.config.mg_concentration / 3.57 * torch.exp(exponent)
        gate = 1.0 / (1.0 + block_factor)
        
        return gate
    
    def apply_to_current(
        self,
        current: torch.Tensor,
        membrane_potential: torch.Tensor,
    ) -> torch.Tensor:
        """Apply NMDA gating to input current.
        
        Splits current into AMPA (always active) and NMDA (voltage-gated).
        
        Args:
            current: Total input current
            membrane_potential: Postsynaptic membrane potential
            
        Returns:
            Effective current after NMDA gating
        """
        gate = self.compute_gate(membrane_potential)
        
        # AMPA fraction always passes, NMDA fraction is gated
        ampa_fraction = 1.0 - self.config.nmda_fraction
        effective = current * (ampa_fraction + self.config.nmda_fraction * gate)
        
        return effective

    def apply_to_conductance(
        self,
        g_exc: torch.Tensor,
        membrane_potential: torch.Tensor,
    ) -> torch.Tensor:
        """Apply NMDA gating to excitatory conductance.
        
        For conductance-based neurons, the NMDA voltage-dependent gating
        modulates the excitatory conductance directly. The total g_exc is
        split into AMPA (always active) and NMDA (voltage-gated) components.
        
        This is biologically accurate: NMDA receptors are a subset of
        glutamatergic synapses with voltage-dependent Mg2+ block.
        
        Args:
            g_exc: Excitatory conductance (any shape)
            membrane_potential: Postsynaptic membrane potential
            
        Returns:
            Effective g_exc after NMDA voltage-dependent gating
        """
        gate = self.compute_gate(membrane_potential)
        
        # AMPA fraction always passes, NMDA fraction is voltage-gated
        ampa_fraction = 1.0 - self.config.nmda_fraction
        effective_g = g_exc * (ampa_fraction + self.config.nmda_fraction * gate)
        
        return effective_g


@dataclass
class DendriticConfig:
    """Configuration for dendritic saturation.
    
    Attributes:
        saturation_threshold: Input level at which saturation begins
        saturation_steepness: How sharply saturation occurs
        compartments: Number of dendritic compartments (1 = single compartment)
    """
    saturation_threshold: float = 2.0   # Normalized input level
    saturation_steepness: float = 1.0   # Steepness of saturation curve
    compartments: int = 1               # Single compartment for simplicity


class DendriticSaturation:
    """Dendritic saturation - limits total input integration.
    
    Models the limited capacity of dendrites to integrate inputs:
    - Low input: linear integration
    - High input: saturating (sublinear) response
    
    Uses a soft saturation function: y = x / (1 + |x|/threshold)
    
    This prevents any single input pattern from driving excessive activity
    and encourages competition between inputs.
    
    Example:
        >>> dendrite = DendriticSaturation(config=DendriticConfig())
        >>> 
        >>> # Apply saturation to total input current
        >>> saturated_current = dendrite.apply(total_input_current)
    """
    
    def __init__(self, config: Optional[DendriticConfig] = None):
        self.config = config or DendriticConfig()
    
    def apply(self, current: torch.Tensor) -> torch.Tensor:
        """Apply dendritic saturation to input current.
        
        Args:
            current: Total input current (any shape)
            
        Returns:
            Saturated current (same shape)
        """
        threshold = self.config.saturation_threshold
        steepness = self.config.saturation_steepness
        
        # Soft saturation: x / (1 + |x|^steepness / threshold)
        magnitude = torch.abs(current)
        saturation_factor = 1.0 / (1.0 + torch.pow(magnitude / threshold, steepness))
        
        return current * saturation_factor

    def apply_to_conductance(
        self,
        g_exc: torch.Tensor,
        E_E: float = 3.0,
        E_L: float = 0.0,
    ) -> torch.Tensor:
        """Apply dendritic saturation to excitatory conductance.
        
        For conductance-based neurons, saturation limits g_exc directly.
        The threshold is scaled by the driving force (E_E - E_L) to maintain
        consistent saturation behavior across current and conductance modes.
        
        This models the biophysical limitation that dendritic branches can
        only integrate a finite number of synaptic inputs before saturating.
        
        Args:
            g_exc: Excitatory conductance (any shape)
            E_E: Excitatory reversal potential (default: 3.0)
            E_L: Leak reversal potential (default: 0.0)
            
        Returns:
            Saturated g_exc (same shape)
        """
        # Scale threshold from current domain to conductance domain
        # I = g * (E_E - E_L), so g = I / (E_E - E_L)
        driving_force = E_E - E_L
        g_threshold = self.config.saturation_threshold / driving_force
        steepness = self.config.saturation_steepness
        
        # Soft saturation: g / (1 + g^steepness / threshold)
        saturation_factor = 1.0 / (1.0 + torch.pow(g_exc / g_threshold, steepness))
        
        return g_exc * saturation_factor


@dataclass
class NeuromodulationConfig:
    """Configuration for neuromodulation effects.
    
    Attributes:
        dopamine_baseline: Baseline dopamine level (0-1)
        dopamine_tau_ms: Dopamine decay time constant (ms)
        ach_baseline: Baseline acetylcholine level (0-1)
        ach_tau_ms: Acetylcholine decay time constant (ms)
        learning_rate_modulation: How much DA affects learning rate
        gain_modulation: How much DA affects neural gain
    """
    # Dopamine (reward/salience)
    dopamine_baseline: float = 0.5
    dopamine_tau_ms: float = 200.0
    learning_rate_modulation: float = 2.0  # 2x learning at max DA
    
    # Acetylcholine (attention/learning)
    ach_baseline: float = 0.5
    ach_tau_ms: float = 500.0
    gain_modulation: float = 1.5  # 1.5x gain at max ACh


class Neuromodulation:
    """Neuromodulation system (dopamine and acetylcholine).
    
    Models global neuromodulatory effects:
    - Dopamine (DA): Modulates learning rate based on reward prediction error
    - Acetylcholine (ACh): Modulates attention and neural gain
    
    These are GLOBAL signals that affect the entire network, unlike
    synaptic plasticity which is local.
    
    Example:
        >>> neuromod = Neuromodulation(config=NeuromodulationConfig())
        >>> 
        >>> # Update based on reward/attention signals
        >>> neuromod.update_dopamine(reward_signal, dt=0.1)
        >>> neuromod.update_ach(attention_signal, dt=0.1)
        >>> 
        >>> # Get modulated learning rate
        >>> lr = base_lr * neuromod.get_learning_rate_factor()
        >>> 
        >>> # Get modulated neural gain
        >>> gain = neuromod.get_gain_factor()
    """
    
    def __init__(
        self,
        config: Optional[NeuromodulationConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or NeuromodulationConfig()
        self.device = device or torch.device("cpu")
        
        # Current neuromodulator levels
        self.dopamine = torch.tensor(self.config.dopamine_baseline, device=self.device)
        self.acetylcholine = torch.tensor(self.config.ach_baseline, device=self.device)
        
        # Precomputed decay factors
        self._da_decay: Optional[float] = None
        self._ach_decay: Optional[float] = None
    
    def to(self, device: torch.device) -> "Neuromodulation":
        """Move to device."""
        self.device = device
        self.dopamine = self.dopamine.to(device)
        self.acetylcholine = self.acetylcholine.to(device)
        return self
    
    def reset(self) -> None:
        """Reset to baseline levels."""
        self.dopamine.fill_(self.config.dopamine_baseline)
        self.acetylcholine.fill_(self.config.ach_baseline)
    
    def _ensure_decay_factors(self, dt: float) -> None:
        """Precompute decay factors for given dt."""
        if self._da_decay is None:
            self._da_decay = 1.0 - dt / self.config.dopamine_tau_ms
            self._ach_decay = 1.0 - dt / self.config.ach_tau_ms
    
    def update_dopamine(self, signal: float, dt: float = 0.1) -> None:
        """Update dopamine level based on reward/salience signal.
        
        Args:
            signal: Reward prediction error or salience signal (can be negative)
            dt: Timestep in ms
        """
        self._ensure_decay_factors(dt)
        
        # Phasic dopamine response to signal
        phasic = signal * 0.1  # Scale factor
        
        # Update with decay toward baseline
        self.dopamine = (
            self.config.dopamine_baseline + 
            (self.dopamine - self.config.dopamine_baseline + phasic) * self._da_decay
        )
        self.dopamine = self.dopamine.clamp(0, 1)
    
    def update_ach(self, attention: float, dt: float = 0.1) -> None:
        """Update acetylcholine level based on attention signal.
        
        Args:
            attention: Attention/novelty signal (0-1)
            dt: Timestep in ms
        """
        self._ensure_decay_factors(dt)
        
        # ACh increases with attention/novelty
        target = attention
        
        # Update with decay toward target
        self.acetylcholine = (
            target + (self.acetylcholine - target) * self._ach_decay
        )
        self.acetylcholine = self.acetylcholine.clamp(0, 1)
    
    def get_learning_rate_factor(self) -> float:
        """Get learning rate modulation factor based on dopamine.
        
        Returns:
            Multiplier for learning rate (typically 0.5 to 2.0)
        """
        # Linear modulation: baseline DA → 1.0x, max DA → learning_rate_modulation
        da = self.dopamine.item()
        baseline = self.config.dopamine_baseline
        max_mod = self.config.learning_rate_modulation
        
        # Map [0, 1] DA to [1/max_mod, max_mod] learning rate factor
        # At baseline DA (0.5), factor = 1.0
        if da >= baseline:
            factor = 1.0 + (da - baseline) / (1.0 - baseline) * (max_mod - 1.0)
        else:
            factor = 1.0 - (baseline - da) / baseline * (1.0 - 1.0/max_mod)
        
        return factor
    
    def get_gain_factor(self) -> float:
        """Get neural gain modulation factor based on acetylcholine.
        
        Returns:
            Multiplier for neural gain (typically 0.7 to 1.5)
        """
        # ACh increases gain/responsiveness
        ach = self.acetylcholine.item()
        max_mod = self.config.gain_modulation
        
        # Map [0, 1] ACh to [1/max_mod, max_mod]
        factor = 1.0 / max_mod + ach * (max_mod - 1.0 / max_mod)
        
        return factor


# Convenience function to create all mechanisms
def create_synaptic_mechanisms(
    n_pre: int,
    n_post: int,
    device: torch.device,
    stp_config: Optional[STPConfig] = None,
    nmda_config: Optional[NMDAConfig] = None,
    dendritic_config: Optional[DendriticConfig] = None,
    neuromod_config: Optional[NeuromodulationConfig] = None,
) -> dict:
    """Create synaptic mechanisms from config objects.
    
    Each mechanism is created only if its config is provided (not None).
    This provides a clean way to enable/disable mechanisms and configure them.
    
    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        device: Torch device
        stp_config: STP config (enables STD/STF). Set mode="depression", 
                    "facilitation", or "both".
        nmda_config: NMDA config (enables voltage-dependent gating)
        dendritic_config: Dendritic config (enables input saturation)
        neuromod_config: Neuromodulation config (enables dopamine/ACh effects)
        
    Returns:
        Dictionary of mechanism instances. Keys present only if enabled:
        - "stp": ShortTermPlasticity
        - "nmda": NMDAGating  
        - "dendritic": DendriticSaturation
        - "neuromod": Neuromodulation
        
    Example:
        >>> # Enable only STD and NMDA with custom settings
        >>> mechanisms = create_synaptic_mechanisms(
        ...     n_pre=20, n_post=10, device=device,
        ...     stp_config=STPConfig(mode="depression", depression_rate=0.3),
        ...     nmda_config=NMDAConfig(nmda_fraction=0.4),
        ... )
    """
    mechanisms = {}
    
    if stp_config is not None:
        mechanisms["stp"] = ShortTermPlasticity(n_pre, n_post, stp_config, device)
    
    if nmda_config is not None:
        mechanisms["nmda"] = NMDAGating(nmda_config)
    
    if dendritic_config is not None:
        mechanisms["dendritic"] = DendriticSaturation(dendritic_config)
    
    if neuromod_config is not None:
        mechanisms["neuromod"] = Neuromodulation(neuromod_config, device)
    
    return mechanisms
