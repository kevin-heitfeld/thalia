"""
Standard STP Configurations for Common Pathway Types.

This module provides biologically-validated presets for short-term plasticity
configurations across different pathway types. Each preset is based on
experimental measurements from neuroscience literature.

Usage:
    from thalia.core.stp_presets import STP_PRESETS, get_stp_config

    # Use a preset directly
    config = STP_PRESETS["mossy_fiber"].configure(dt=1.0)
    
    # Or use the helper function
    config = get_stp_config("mossy_fiber", dt=1.0)

References:
    - Salin et al. (1996): Distinct short-term plasticity at two excitatory pathways
      in the hippocampus. PNAS 93: 13304-13309.
    - Tsodyks & Markram (1997): The neural code between neocortical pyramidal neurons
      depends on neurotransmitter release probability. PNAS 94: 719-723.
    - Markram et al. (1998): Differential signaling via the same axon of neocortical
      pyramidal neurons. PNAS 95: 5323-5328.
"""

from dataclasses import dataclass
from typing import Dict

from thalia.components.synapses.stp import STPConfig


@dataclass(frozen=True)
class STPPreset:
    """Immutable preset for STP configuration with biological documentation.
    
    Attributes:
        name: Human-readable name of the preset
        U: Baseline release probability (0-1)
        tau_u: Facilitation time constant (ms) - same as tau_f
        tau_x: Depression recovery time constant (ms) - same as tau_d
        description: Biological context and use case
        references: Key papers validating these parameters
    """
    name: str
    U: float
    tau_u: float  # Facilitation decay (tau_f in STPConfig)
    tau_x: float  # Depression recovery (tau_d in STPConfig)
    description: str
    references: str
    
    def configure(self, dt: float = 1.0) -> STPConfig:
        """Create STPConfig with this preset's parameters.
        
        Args:
            dt: Simulation timestep in milliseconds
            
        Returns:
            Configured STPConfig instance
        """
        return STPConfig(
            U=self.U,
            tau_d=self.tau_x,  # tau_x → tau_d (depression recovery)
            tau_f=self.tau_u,  # tau_u → tau_f (facilitation decay)
            dt=dt
        )


# =============================================================================
# HIPPOCAMPAL PATHWAY PRESETS
# =============================================================================

MOSSY_FIBER_PRESET = STPPreset(
    name="Mossy Fiber (DG→CA3)",
    U=0.03,
    tau_u=800.0,
    tau_x=200.0,
    description=(
        "Dentate gyrus mossy fiber to CA3 pyramidal cells. "
        "Very strong facilitation (low U, long tau_u). "
        "Critical for pattern separation and rapid encoding. "
        "Weak baseline, strong burst response."
    ),
    references="Salin et al. (1996); Nicoll & Schmitz (2005)"
)

SCHAFFER_COLLATERAL_PRESET = STPPreset(
    name="Schaffer Collateral (CA3→CA1)",
    U=0.5,
    tau_u=400.0,
    tau_x=500.0,
    description=(
        "CA3 Schaffer collateral to CA1 pyramidal cells. "
        "Moderate depression (medium U). "
        "Main output pathway from CA3 attractor to CA1 comparator. "
        "Balances reliability with dynamic range."
    ),
    references="Salin et al. (1996); Dobrunz & Stevens (1997)"
)

EC_CA1_PRESET = STPPreset(
    name="Perforant Path (EC→CA1)",
    U=0.35,
    tau_u=300.0,
    tau_x=400.0,
    description=(
        "Entorhinal cortex direct pathway to CA1 pyramidal cells. "
        "Mild depression. "
        "Provides baseline input for match/mismatch comparison. "
        "Less plastic than CA3→CA1 pathway."
    ),
    references="Bartesaghi & Gessi (2004)"
)

CA3_RECURRENT_PRESET = STPPreset(
    name="CA3 Recurrent (CA3→CA3)",
    U=0.4,
    tau_u=200.0,
    tau_x=300.0,
    description=(
        "CA3 recurrent collaterals (auto-associative connections). "
        "Moderate depression with fast recovery. "
        "Supports pattern completion while preventing runaway excitation. "
        "Short time constants for rapid attractor dynamics."
    ),
    references="Miles & Wong (1986); Debanne et al. (1996)"
)

# =============================================================================
# CORTICAL PATHWAY PRESETS
# =============================================================================

CORTICAL_FF_PRESET = STPPreset(
    name="Cortical Feedforward (L4→L2/3)",
    U=0.2,
    tau_u=200.0,
    tau_x=300.0,
    description=(
        "Thalamocortical and layer 4 to layer 2/3 connections. "
        "Weak facilitation. "
        "Transmits sensory information with temporal filtering. "
        "Responds preferentially to stimulus changes."
    ),
    references="Tsodyks & Markram (1997); Reyes et al. (1998)"
)

CORTICAL_RECURRENT_PRESET = STPPreset(
    name="Cortical Recurrent (L2/3→L2/3)",
    U=0.6,
    tau_u=100.0,
    tau_x=200.0,
    description=(
        "Recurrent excitatory connections within cortical layers. "
        "Strong depression with fast dynamics. "
        "Provides gain control and prevents runaway activity. "
        "Implements divisive normalization."
    ),
    references="Markram et al. (1998); Tsodyks et al. (2000)"
)

CORTICAL_FB_PRESET = STPPreset(
    name="Cortical Feedback (L5→L2/3)",
    U=0.3,
    tau_u=250.0,
    tau_x=350.0,
    description=(
        "Feedback connections from deep to superficial layers. "
        "Mild depression with moderate recovery. "
        "Carries top-down predictions and attentional modulation. "
        "Balanced for sustained modulation."
    ),
    references="Thomson & Bannister (2003)"
)

CORTICAL_PYRAMIDAL_INTERNEURON_PRESET = STPPreset(
    name="Pyramidal→Interneuron",
    U=0.7,
    tau_u=50.0,
    tau_x=150.0,
    description=(
        "Pyramidal cell to inhibitory interneuron. "
        "Strong depression with very fast dynamics. "
        "Provides feedforward inhibition for gain control. "
        "Quick response, rapid fatigue."
    ),
    references="Gupta et al. (2000); Galarreta & Hestrin (1998)"
)

# =============================================================================
# STRIATAL PATHWAY PRESETS
# =============================================================================

CORTICOSTRIATAL_PRESET = STPPreset(
    name="Corticostriatal (Cortex→Striatum)",
    U=0.4,
    tau_u=150.0,
    tau_x=250.0,
    description=(
        "Cortical to medium spiny neuron connections. "
        "Moderate depression. "
        "Main input pathway for action selection. "
        "Provides context for reinforcement learning."
    ),
    references="Charpier et al. (1999); Partridge et al. (2000)"
)

THALAMO_STRIATAL_PRESET = STPPreset(
    name="Thalamostriatal (Thalamus→Striatum)",
    U=0.25,
    tau_u=300.0,
    tau_x=400.0,
    description=(
        "Thalamic to striatal connections. "
        "Weak facilitation. "
        "Provides direct sensory and motivational input. "
        "Complements cortical input."
    ),
    references="Ding et al. (2008)"
)

# =============================================================================
# PREFRONTAL PATHWAY PRESETS
# =============================================================================

PFC_RECURRENT_PRESET = STPPreset(
    name="PFC Recurrent",
    U=0.15,
    tau_u=400.0,
    tau_x=300.0,
    description=(
        "Prefrontal cortex recurrent connections. "
        "Weak facilitation for working memory. "
        "Long time constants support persistent activity. "
        "Critical for delay period maintenance."
    ),
    references="Wang et al. (2013); Compte et al. (2000)"
)

PFC_TO_STRIATUM_PRESET = STPPreset(
    name="PFC→Striatum",
    U=0.35,
    tau_u=200.0,
    tau_x=300.0,
    description=(
        "Prefrontal to striatal top-down modulation. "
        "Mild depression. "
        "Gates action selection based on goals. "
        "Provides contextual biasing signal."
    ),
    references="Haber et al. (2000)"
)

# =============================================================================
# PRESET REGISTRY
# =============================================================================

STP_PRESETS: Dict[str, STPPreset] = {
    # Hippocampal pathways
    "mossy_fiber": MOSSY_FIBER_PRESET,
    "schaffer_collateral": SCHAFFER_COLLATERAL_PRESET,
    "ec_ca1": EC_CA1_PRESET,
    "ca3_recurrent": CA3_RECURRENT_PRESET,
    
    # Cortical pathways
    "cortical_ff": CORTICAL_FF_PRESET,
    "cortical_recurrent": CORTICAL_RECURRENT_PRESET,
    "cortical_fb": CORTICAL_FB_PRESET,
    "cortical_pyr_int": CORTICAL_PYRAMIDAL_INTERNEURON_PRESET,
    
    # Striatal pathways
    "corticostriatal": CORTICOSTRIATAL_PRESET,
    "thalamostriatal": THALAMO_STRIATAL_PRESET,
    
    # Prefrontal pathways
    "pfc_recurrent": PFC_RECURRENT_PRESET,
    "pfc_striatum": PFC_TO_STRIATUM_PRESET,
}


def get_stp_config(pathway_type: str, dt: float = 1.0) -> STPConfig:
    """Get standard STP configuration for a pathway type.
    
    Args:
        pathway_type: Name of the pathway (see STP_PRESETS keys)
        dt: Simulation timestep in milliseconds
        
    Returns:
        Configured STPConfig instance
        
    Raises:
        KeyError: If pathway_type is not recognized
        
    Example:
        >>> config = get_stp_config("mossy_fiber", dt=1.0)
        >>> print(f"U={config.U}, tau_d={config.tau_d}, tau_f={config.tau_f}")
        U=0.03, tau_d=200.0, tau_f=800.0
    """
    if pathway_type not in STP_PRESETS:
        available = ", ".join(STP_PRESETS.keys())
        raise KeyError(
            f"Unknown pathway type: {pathway_type}. "
            f"Available presets: {available}"
        )
    
    return STP_PRESETS[pathway_type].configure(dt=dt)


def list_presets() -> Dict[str, str]:
    """List all available STP presets with descriptions.
    
    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {
        name: preset.description
        for name, preset in STP_PRESETS.items()
    }
