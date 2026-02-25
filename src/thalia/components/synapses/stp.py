"""
Short-Term Plasticity (STP) - Tsodyks-Markram Model.

Short-term plasticity modulates synaptic strength on fast timescales (10ms-1s)
based on recent presynaptic activity. Unlike long-term plasticity (STDP),
STP is transient and automatically recovers.

Two key phenomena:
1. SHORT-TERM DEPRESSION (STD): Repeated firing depletes vesicles
   - High-frequency bursts → progressively weaker responses
   - Implements a temporal high-pass filter
   - Dominant in cortical pyramidal → interneuron synapses

2. SHORT-TERM FACILITATION (STF): Residual calcium enhances release
   - High-frequency bursts → progressively stronger responses
   - Implements a temporal low-pass filter / coincidence detection
   - Dominant in some cortical pyramidal → pyramidal synapses

The balance between STD and STF determines synapse type:
- Depressing (D): High initial U, fast depression, slow recovery
- Facilitating (F): Low initial U, strong facilitation, fast recovery
- Mixed (E1/E2/E3): Various combinations

Biological basis:
- Vesicle depletion and recycling (depression)
- Residual calcium in presynaptic terminal (facilitation)
- Vesicle pool dynamics (readily releasable, recycling, reserve)

Computational benefits:
- Temporal filtering (high-pass or low-pass depending on synapse type)
- Gain control (automatic normalization of bursts)
- Working memory (facilitation can maintain activity)
- Novelty detection (depressing synapses respond to change)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import torch
import torch.nn as nn


class STPType(StrEnum):
    """Predefined synapse types based on Markram et al. (1998) classification."""

    # Depressing synapses (high U, dominant depression)
    DEPRESSING = "depressing"  # Strong initial, rapid fatigue (U=0.5)
    DEPRESSING_MODERATE = "depressing_moderate"  # Moderate depression (U=0.4, thalamic sensory)
    DEPRESSING_STRONG = "depressing_strong"  # Strong depression (U=0.7, thalamic L6 feedback)
    DEPRESSING_FAST = "depressing_fast"  # Very fast depression, quick recovery (U=0.8)

    # Facilitating synapses (low U, dominant facilitation)
    FACILITATING = "facilitating"  # Weak initial, builds up with activity
    FACILITATING_MODERATE = "facilitating_moderate"  # Moderate facilitation (U=0.15, thalamic L6 feedback)
    FACILITATING_STRONG = "facilitating_strong"  # Very strong facilitation

    # Mixed dynamics
    PSEUDOLINEAR = "pseudolinear"  # Balanced, roughly linear response

    # No STP (pass-through)
    NONE = "none"


@dataclass
class STPConfig:
    """Configuration for short-term plasticity.

    The Tsodyks-Markram model uses three variables:
    - u: Utilization (release probability), increases with facilitation
    - x: Available resources (vesicles), decreases with depression
    - Effective transmission = u * x * w

    Dynamics:
    - On pre-spike: u → u + U(1-u), then x → x - u*x
    - Between spikes: u decays to U with τ_f, x recovers to 1 with τ_d

    Attributes:
        U: Baseline release probability (0-1)
            High U (~0.5-0.9) → depressing synapse
            Low U (~0.05-0.2) → facilitating synapse

        tau_d: Depression recovery time constant (ms)
            Time for vesicle pool to refill after depletion.
            Typical: 200-800ms for cortical synapses

        tau_f: Facilitation decay time constant (ms)
            Time for residual calcium to clear.
            Typical: 50-200ms for cortical synapses
    """

    U: float = 0.5  # Baseline release probability
    tau_d: float = 200.0  # Depression recovery (ms)
    tau_f: float = 50.0  # Facilitation decay (ms)

    device: torch.device = torch.device("cpu")

    @classmethod
    def from_type(cls, stp_type: STPType) -> STPConfig:
        """Create config from predefined synapse type.

        Parameters based on Markram et al. (1998) fits to cortical data.
        """
        if stp_type == STPType.DEPRESSING:
            return cls(U=0.5, tau_d=800.0, tau_f=20.0)
        elif stp_type == STPType.DEPRESSING_MODERATE:
            return cls(U=0.4, tau_d=700.0, tau_f=30.0)
        elif stp_type == STPType.DEPRESSING_STRONG:
            return cls(U=0.7, tau_d=600.0, tau_f=15.0)
        elif stp_type == STPType.DEPRESSING_FAST:
            return cls(U=0.8, tau_d=200.0, tau_f=10.0)
        elif stp_type == STPType.FACILITATING:
            return cls(U=0.15, tau_d=200.0, tau_f=200.0)
        elif stp_type == STPType.FACILITATING_MODERATE:
            return cls(U=0.1, tau_d=300.0, tau_f=300.0)
        elif stp_type == STPType.FACILITATING_STRONG:
            return cls(U=0.05, tau_d=100.0, tau_f=500.0)
        elif stp_type == STPType.PSEUDOLINEAR:
            return cls(U=0.3, tau_d=400.0, tau_f=100.0)
        else:  # NONE
            return cls(U=1.0, tau_d=1e6, tau_f=1e-6)  # Effectively no STP


class ShortTermPlasticity(nn.Module):
    """Tsodyks-Markram short-term plasticity model.

    Modulates synaptic efficacy based on recent presynaptic activity.
    Returns a multiplicative factor (0-1) to apply to synaptic weights.

    The model tracks two state variables per synapse:
    - u: Release probability (facilitation variable)
    - x: Available resources (depression variable)

    On each presynaptic spike:
    1. u increases (facilitation): u → u + U(1-u)
    2. x decreases (depression): x → x - u*x
    3. Transmission efficacy = u * x (before the x update)

    Between spikes:
    - u decays toward U with time constant τ_f
    - x recovers toward 1 with time constant τ_d

    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons (for per-synapse STP)
            If None, uses per-presynaptic-neuron STP (shared across targets)
        config: STP configuration parameters
    """

    def __init__(self, n_pre: int, n_post: int, config: STPConfig):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config

        # Register constants
        self.register_buffer("U", torch.tensor(self.config.U, dtype=torch.float32))

        # Decay factors (computed from dt in update_temporal_parameters)
        # Initialize to dummy values - must call update_temporal_parameters() before use
        self._dt_ms: Optional[float] = None
        self.register_buffer("decay_d", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("decay_f", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("recovery_d", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("recovery_f", torch.tensor(0.0, dtype=torch.float32))

        # Initialize state variables
        shape = (self.n_pre, self.n_post)
        # Release probability (facilitation)
        self.u: torch.Tensor = torch.full(shape, self.U.item(), device=self.config.device, dtype=torch.float32)
        # Available resources (depression)
        self.x: torch.Tensor = torch.ones(shape, device=self.config.device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute STP efficacy for current timestep (ADR-005: 1D tensors).

        Args:
            pre_spikes: Presynaptic spikes [n_pre] (1D)

        Returns:
            Efficacy factor:
            - [n_pre, n_post] (2D matrix)

            Multiply this with synaptic weights to get effective transmission.
        """
        assert (
            pre_spikes.dim() == 1
        ), f"STP.forward: Expected 1D pre_spikes (ADR-005), got shape {pre_spikes.shape}"
        assert (
            pre_spikes.shape[0] == self.n_pre
        ), f"STP.forward: pre_spikes has {pre_spikes.shape[0]} neurons, expected {self.n_pre}"

        # [n_pre] → [n_pre, 1] for broadcasting to [n_pre, n_post]
        spikes = pre_spikes.unsqueeze(-1)

        # === Continuous dynamics (between spikes) ===
        # u decays toward U: u(t+dt) = u(t)*decay_f + U*(1-decay_f)
        # x recovers toward 1: x(t+dt) = x(t)*decay_d + 1*(1-decay_d)

        self.u = self.u * self.decay_f + self.recovery_f
        self.x = self.x * self.decay_d + self.recovery_d

        # === Spike-triggered dynamics ===
        # Compute efficacy BEFORE applying spike effects
        # This is the u*x at the moment of the spike
        efficacy = self.u * self.x

        # On spike: u jumps up (facilitation), x drops (depression)
        # u → u + U(1-u) = u(1-U) + U = lerp toward 1 with step U
        # x → x(1-u) = x - u*x (release fraction u of available pool)

        # IMPORTANT: The x update uses the OLD value of u (before facilitation)
        # This is the Tsodyks-Markram convention: depression is based on the
        # release probability at the time of the spike, not after facilitation

        # Apply spike effects (only where spikes occurred)
        # Depression first (uses current u): x decreases by u*x
        # OPTIMIZATION: Compute x_release directly without clone
        x_release = self.u * self.x  # u*x at spike time (before facilitation)
        self.x = self.x - spikes * x_release

        # Facilitation second: u increases toward 1
        # u → u + U(1-u) when spike occurs
        u_jump = self.U * (1.0 - self.u)
        self.u = self.u + spikes * u_jump

        # Clamp to valid range (numerical safety)
        # OPTIMIZATION: Use clamp_ (in-place) to avoid allocation
        self.u.clamp_(0.0, 1.0)
        self.x.clamp_(0.0, 1.0)

        # Return efficacy (not modulated by spikes - that's the synaptic response)
        # The caller uses this to scale the PSP/PSC
        return efficacy

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Recomputes cached decay/recovery factors based on new dt.
        Called by DynamicBrain.set_timestep().

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        device = self.U.device

        # Recompute depression decay: exp(-dt / tau_d)
        self.decay_d = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_d)).item()),
            device=device,
            dtype=torch.float32,
        )

        # Recompute facilitation decay: exp(-dt / tau_f)
        self.decay_f = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_f)).item()),
            device=device,
            dtype=torch.float32,
        )

        # Compute recovery rates
        self.recovery_d = torch.tensor(
            1.0 - self.decay_d.item(), device=device, dtype=torch.float32
        )
        self.recovery_f = torch.tensor(
            (1.0 - self.decay_f.item()) * self.config.U, device=device, dtype=torch.float32
        )


# =============================================================================
# STP presets and utilities
# =============================================================================

@dataclass(frozen=True)
class STPPreset:
    """Immutable preset for STP configuration with biological documentation.

    Attributes:
        name: Human-readable name of the preset
        U: Baseline release probability (0-1)
        tau_u: Facilitation time constant (ms) - same as tau_f
        tau_x: Depression recovery time constant (ms) - same as tau_d
        description: Biological context and use case
    """

    name: str
    U: float
    tau_u: float  # Facilitation decay (tau_f in STPConfig)
    tau_x: float  # Depression recovery (tau_d in STPConfig)
    description: str

    def configure(self) -> STPConfig:
        """Create STPConfig with this preset's parameters.

        Returns:
            Configured STPConfig instance
        """
        return STPConfig(
            U=self.U,
            tau_d=self.tau_x,  # tau_x → tau_d (depression recovery)
            tau_f=self.tau_u,  # tau_u → tau_f (facilitation decay)
        )


# =============================================================================
# HIPPOCAMPAL PATHWAY PRESETS
# =============================================================================

MOSSY_FIBER_PRESET = STPPreset(
    name="Mossy Fiber (DG→CA3)",
    U=0.01,  # Reduced from 0.03 to reduce facilitation buildup
    tau_u=400.0,  # Reduced from 800.0 to decay facilitation faster
    tau_x=200.0,
    description=(
        "Dentate gyrus mossy fiber to CA3 pyramidal cells. "
        "Very strong facilitation (low U, long tau_u). "
        "Critical for pattern separation and rapid encoding. "
        "Weak baseline, strong burst response."
    ),
)

SCHAFFER_COLLATERAL_PRESET = STPPreset(
    name="Schaffer Collateral (CA3→CA1)",
    U=0.5,
    tau_u=400.0,
    tau_x=700.0,
    description=(
        "CA3 Schaffer collateral to CA1 pyramidal cells. "
        "Moderate depression (medium U). "
        "Main output pathway from CA3 attractor to CA1 comparator. "
        "Balances reliability with dynamic range."
    ),
)

# =============================================================================
# STRIATAL PATHWAY PRESETS
# =============================================================================

CORTICOSTRIATAL_PRESET = STPPreset(
    name="Corticostriatal",
    U=0.4,
    tau_u=150.0,
    tau_x=250.0,
    description=(
        "Cortical to medium spiny neuron connections. "
        "Moderate depression. "
        "Main input pathway for action selection. "
        "Provides context for reinforcement learning."
    ),
)

THALAMO_STRIATAL_PRESET = STPPreset(
    name="Thalamostriatal (Thalamus→Striatum)",
    U=0.12,
    tau_u=500.0,
    tau_x=200.0,
    description=(
        "Thalamic intralaminar (CL/Pf) to striatal MSN connections. "
        "True short-term facilitation: low baseline release probability, "
        "very long facilitation time constant. "
        "Raju et al. (2008): thalamostriatal synapses from caudal "
        "intralaminar nuclei facilitate strongly with burst activity, "
        "unlike corticostriatal synapses. "
        "Provides direct sensory and motivational input; bursting thalamic "
        "responses recruit increasing striatal drive. "
        "Complements the depressing corticostriatal input."
    ),
)


# =============================================================================
# PERFORANT PATH / CORTICAL-HIPPOCAMPAL PRESETS
# =============================================================================

PERFORANT_PATH_PRESET = STPPreset(
    name="Perforant Path (EC→DG / EC→CA3)",
    U=0.35,
    tau_u=50.0,
    tau_x=600.0,
    description=(
        "Entorhinal cortex perforant path synapses onto dentate granule cells "
        "and CA3 pyramidal cells (stratum lacunosum-moleculare). "
        "Moderate short-term depression: medium-high utilisation with fast "
        "facilitation decay and slow vesicle recovery. "
        "McNaughton (1980); Bortolotto et al. (2003): perforant path EPSPs "
        "depress modestly at theta frequencies, unlike the strong facilitation "
        "of true mossy fiber (DG→CA3) synapses. "
        "Responds reliably to sparse entorhinal activity while attenuating "
        "prolonged high-frequency bursts."
    ),
)

TEMPOROAMMONIC_PRESET = STPPreset(
    name="Temporoammonic Path (EC_III→CA1)",
    U=0.45,
    tau_u=40.0,
    tau_x=650.0,
    description=(
        "Entorhinal cortex layer III direct projection to CA1 distal apical "
        "dendrites (temporoammonic / perforant path to CA1). "
        "Moderate-strong short-term depression. "
        "Empfindk & Bhatt 2000; Otmakhova et al. 2002: EC_III→CA1 EPSPs "
        "depress faster than EC_II→DG, acting as a high-pass novelty filter. "
        "Strong initial response to new context patterns that fades rapidly "
        "with sustained input, emphasising mismatch detection."
    ),
)


# =============================================================================
# THALAMOCORTICAL PRESET
# =============================================================================

THALAMOCORTICAL_PRESET = STPPreset(
    name="Thalamocortical (Thalamus→Cortex L4)",
    U=0.45,
    tau_u=20.0,
    tau_x=700.0,
    description=(
        "Thalamic relay neuron synapses onto cortical layer 4 spiny stellate "
        "and pyramidal cells. "
        "Strong short-term depression: high initial release probability, very "
        "fast facilitation decay, slow vesicle recovery. "
        "Gil et al. (1997); Stratford et al. (1996); Bruno & Sakmann (2006): "
        "thalamocortical EPSPs depress by ~50% after the 2nd spike at 10 Hz. "
        "Acts as a temporal high-pass filter that privileges the first volley "
        "of thalamic activity — critical for novelty detection and the "
        "transient burst that gates L4 responses to stimulus onset."
    ),
)


# =============================================================================
# INTRA-CORTICAL PRESETS
# =============================================================================

CORTICAL_FF_PRESET = STPPreset(
    name="Cortical Feedforward (L4→L2/3, L2/3→L5)",
    U=0.50,
    tau_u=25.0,
    tau_x=600.0,
    description=(
        "Cortical feedforward connections between excitatory populations: "
        "L4 spiny stellate → L2/3 pyramidal, and L2/3 → L5 pyramidal. "
        "Moderate short-term depression. "
        "Reyes & Sakmann (1999); Thomson et al. (2002): L4→L2/3 connections "
        "show reliable but depressing EPSPs. "
        "Limits sustained runaway activation in the feedforward cascade and "
        "implements gain normalisation across cortical layers."
    ),
)

CORTICAL_RECURRENT_PRESET = STPPreset(
    name="Cortical Recurrent (L2/3→L2/3)",
    U=0.12,
    tau_u=600.0,
    tau_x=150.0,
    description=(
        "Layer 2/3 pyramidal-to-pyramidal recurrent connections. "
        "Strong short-term facilitation: very low baseline release probability, "
        "very long facilitation time constant, fast recovery. "
        "Markram et al. (1998): the canonical facilitating synapse — the "
        "prototypical EPSP-E type. Burst-coding: weak baseline, builds up "
        "strongly during sustained activity. "
        "Implements attractor dynamics, working memory maintenance, and "
        "pattern completion in supragranular cortex."
    ),
)


# =============================================================================
# INTRA-HIPPOCAMPAL PRESET
# =============================================================================

CA3_RECURRENT_PRESET = STPPreset(
    name="CA3 Recurrent Collateral (CA3→CA3)",
    U=0.50,
    tau_u=30.0,
    tau_x=500.0,
    description=(
        "CA3 recurrent Schaffer-like collateral connections. "
        "Moderate-strong short-term depression. "
        "Dobrunz & Stevens (1999); Fioravante & Regehr (2011): CA3 recurrent "
        "collaterals depress strongly at theta frequencies (4-10 Hz), acting "
        "as a high-pass filter that limits runaway excitation during pattern "
        "completion. "
        "Depression prevents attractor lock-up and ensures CA3 responds "
        "transiently to each new pattern rather than maintaining spurious "
        "sustained activity."
    ),
)


# =============================================================================
# STRIATAL INTERNEURON PRESET
# =============================================================================

FSI_MSN_PRESET = STPPreset(
    name="FSI→MSN (Striatal Fast-Spiking Interneuron→Medium Spiny Neuron)",
    U=0.65,
    tau_u=15.0,
    tau_x=550.0,
    description=(
        "Parvalbumin-positive fast-spiking interneuron (FSI) to medium spiny "
        "neuron (MSN / D1 or D2) GABAergic connections in striatum. "
        "Strong short-term depression: high initial release probability, very "
        "fast facilitation decay, slow vesicle recovery. "
        "Planert et al. (2010): FSI→MSN synapses are strongly depressing — "
        "a large initial IPSP is followed by rapid attenuation. "
        "This limits the duration of FSI-mediated inhibition, ensuring that "
        "feedforward inhibition is transient and does not permanently suppress "
        "MSN responses during sustained FSI firing."
    ),
)


# =============================================================================
# PV BASKET CELL PRESET (hippocampus / cortex interneuron inhibition)
# =============================================================================

PV_BASKET_PRESET = STPPreset(
    name="PV Basket Cell (interneuron→pyramidal GABA_A)",
    U=0.55,
    tau_u=15.0,
    tau_x=500.0,
    description=(
        "Parvalbumin-positive basket cell GABAergic synapses onto pyramidal "
        "and granule cell somata in hippocampus and cortex. "
        "Strong short-term depression: high release probability, very fast "
        "facilitation decay, slow vesicle recovery. "
        "Hefft & Jonas (2005): CA3 PV basket cell → granule cell IPSPs "
        "depress strongly with repetitive firing at theta frequencies. "
        "Also applies to CA1 PV basket cells and cortical PV axonal inhibition. "
        "Transient somatic shunting on the first IPSP; rapid fatigue ensures "
        "that sustained PV activity does not chronically silence targets."
    ),
)


# =============================================================================
# MSN LATERAL INHIBITION PRESET (striatum MSN→MSN)
# =============================================================================

MSN_LATERAL_PRESET = STPPreset(
    name="MSN Lateral Inhibition (MSN→MSN GABAergic collateral)",
    U=0.35,
    tau_u=20.0,
    tau_x=600.0,
    description=(
        "Medium spiny neuron (MSN) GABAergic axon collateral inhibition onto "
        "other MSNs (D1→D1, D2→D2, and cross-pathway D1↔D2). "
        "Moderate short-term depression: medium release probability, fast "
        "facilitation decay, slow vesicle recovery. "
        "Venance et al. (2004); Taverna et al. (2008): MSN→MSN IPSPs depress "
        "with repetitive stimulation at striatal firing rates (1-20 Hz). "
        "Depression limits the duration of lateral competition, preventing "
        "winner-take-all dynamics from permanently locking out alternatives "
        "and enabling action switching on sub-second timescales."
    ),
)


# =============================================================================
# PONTOCEREBELLAR MOSSY FIBER PRESET (cortex/pons→granule cell)
# =============================================================================

PONTOCEREBELLAR_PRESET = STPPreset(
    name="Pontocerebellar Mossy Fiber (Pons/Cortex→Granule Cell)",
    U=0.10,
    tau_u=500.0,
    tau_x=100.0,
    description=(
        "Pontocerebellar mossy fiber synapses onto cerebellar granule cells "
        "at the glomerulus (excitatory driving input from neocortex via "
        "pontine nuclei). "
        "Strong short-term facilitation: very low baseline release probability, "
        "long facilitation time constant, very fast recovery. "
        "Silver et al. (1998): mossy fiber → granule cell EPSCs facilitate "
        "strongly with burst activity, consistent with low initial Pr. "
        "Sola et al. (2004): pontocerebellar pathway facilitates reliably "
        "at 50-100 Hz, enabling motor-command bursts to recruit granule cells "
        "that are otherwise silenced by tonic Golgi inhibition."
    ),
)


# =============================================================================
# CORTICOTHALAMIC TYPE-II PRESET (L6b/CT→relay, facilitating)
# =============================================================================

CORTICOTHALAMIC_L6B_PRESET = STPPreset(
    name="Corticothalamic Type-II (L6b→Thalamic Relay, facilitating)",
    U=0.08,
    tau_u=800.0,
    tau_x=150.0,
    description=(
        "Type-II corticothalamic (CT) feedback synapses from cortical layer 6b "
        "pyramidal cells onto thalamic relay neurons (driver-like CT feedback; "
        "terminates in proximal dendrites). "
        "Very strong facilitation: very low baseline Pr, extremely long "
        "facilitation time constant, fast recovery. "
        "Jurgens et al. (2012); Reichova & Sherman (2004): type-II CT→TC "
        "synapses are among the most strongly facilitating in the brain "
        "(>10-fold increase with 40 Hz trains), in contrast to the "
        "depressing type-I CT (L6a→TRN) pathway. "
        "Selective gain amplifier: only sustained, high-frequency L6b bursts "
        "meaningfully drive relay neurons, ensuring that only confident "
        "top-down predictions override sensory-driven activity."
    ),
)


# =============================================================================
# STRIATOPALLIDAL / STRIATONIGRAL PATHWAY PRESET
# =============================================================================

STRIATOPALLIDAL_PRESET = STPPreset(
    name="Striatopallidal / Striatonigral (MSN→GPe / MSN→SNr)",
    U=0.45,
    tau_u=25.0,
    tau_x=500.0,
    description=(
        "GABAergic axon collateral synapses from striatal medium spiny neurons "
        "onto basal ganglia output nuclei: D2-MSN→GPe (indirect pathway) and "
        "D1-MSN→SNr (direct pathway). "
        "Moderate-strong short-term depression: medium-high release probability, "
        "fast facilitation decay, slow vesicle recovery. "
        "Connelly et al. (2010): striatopallidal IPSCs at GPe neurons depress "
        "reliably at physiologically relevant MSN firing rates (2-20 Hz). "
        "Gage et al. (2010): striatonigral IPSCs at SNr show similar depression. "
        "Depression prevents tonic high-frequency MSN barrages from permanently "
        "silencing GPe/SNr; actions can be re-gated after brief depolarization."
    ),
)


# =============================================================================
# CORTICOFUGAL PRESET (L5 / L6 → subcortical targets)
# =============================================================================

CORTICOFUGAL_PRESET = STPPreset(
    name="Corticofugal (L5/L6→Subcortical Targets)",
    U=0.50,
    tau_u=20.0,
    tau_x=700.0,
    description=(
        "Corticofugal synapses from cortical layer 5 or 6 pyramidal cells onto "
        "subcortical targets: amygdala, brainstem nuclei (LC, RMTg, DRN), "
        "entorhinal cortex, subthalamic nucleus (hyperdirect), and neuromodulatory "
        "nuclei (NB, LC). "
        "Moderate-to-strong short-term depression: medium release probability, "
        "very fast facilitation decay (short residual Ca2+ at distal axon "
        "terminals), slow vesicle recovery. "
        "Bhattacharyya et al. (2009); Shu et al. (2006): corticofugal terminals "
        "from L5 thick-tufted pyramidal cells depress rapidly at physiological "
        "firing rates.  The fast tau_f (20ms) ensures each spike is effectively "
        "independent — appropriate for the sparse, irregular firing of deep-layer "
        "output neurons. "
        "Biologically distinct from CORTICAL_FF (which targets cortical "
        "superficial layers with slightly slower recovery), but shares strong "
        "STD as the dominant dynamic."
    ),
)


# =============================================================================
# LHb → RMTg FACILITATING PRESET
# =============================================================================

LHB_RMTG_PRESET = STPPreset(
    name="LHb→RMTg (Lateral Habenula → Rostromedial Tegmentum, facilitating)",
    U=0.15,
    tau_u=250.0,
    tau_x=200.0,
    description=(
        "Glutamatergic synapses from lateral habenula principal neurons onto "
        "rostromedial tegmentum (RMTg / tail of VTA) GABAergic neurons. "
        "Short-term facilitation: low baseline release probability, long "
        "facilitation time constant, fast depression recovery. "
        "Hong et al. (2011, Nature): LHb principal neurons burst at "
        "20-100 Hz during negative prediction error events; these bursts "
        "recruit RMTg with progressively growing EPSP amplitude — a clear "
        "facilitating synapse.  The facilitation ensures that brief LHb "
        "volleys (1-2 spikes) barely reach RMTg, while sustained bursts "
        "reliably drive the GABAergic DA-pause.  This provides a biologically "
        "tuned temporal window: only genuine aversive-PE bursts (>5 spikes) "
        "produce a significant dopamine pause."
    ),
)
