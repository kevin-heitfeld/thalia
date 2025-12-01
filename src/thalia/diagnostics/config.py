"""
Configuration classes for diagnostics and mechanism ablation.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class DiagnosticLevel(Enum):
    """Controls verbosity of diagnostic output."""
    NONE = auto()      # Minimal output - just final results
    SUMMARY = auto()   # End-of-cycle summary stats (default)
    VERBOSE = auto()   # Per-phase details, early cycle deep-dives
    DEBUG = auto()     # Everything, including per-timestep data


@dataclass
class DiagnosticConfig:
    """
    Configuration for what diagnostics to collect and report.

    Attributes:
        level: Overall verbosity level
        collect_spike_timing: Track when neurons fire relative to expected phases
        collect_weight_changes: Track LTP/LTD contributions per mechanism
        collect_mechanism_states: Track homeostatic mechanism states (saturation)
        collect_winner_consistency: Track which neurons win which phases
        collect_eligibility: Track eligibility trace dynamics (for three-factor learning)
        collect_dopamine: Track dopamine signal dynamics (for reward-modulated learning)
        report_every_n_cycles: How often to print summary stats
        early_cycle_details: Number of early cycles to show detailed diagnostics
        track_weight_snapshots: Store weight snapshots for evolution analysis
        weight_snapshot_interval: Cycles between weight snapshots
    """
    level: DiagnosticLevel = DiagnosticLevel.SUMMARY

    # What to collect
    collect_spike_timing: bool = True
    collect_weight_changes: bool = True
    collect_mechanism_states: bool = True
    collect_winner_consistency: bool = True
    collect_eligibility: bool = False  # Off by default (only for supervised learning)
    collect_dopamine: bool = False      # Off by default (only for supervised learning)

    # Reporting frequency
    report_every_n_cycles: int = 30
    early_cycle_details: int = 5  # Show detailed output for first N cycles

    # Weight evolution tracking
    track_weight_snapshots: bool = True
    weight_snapshot_interval: int = 10

    @classmethod
    def from_level(cls, level: str) -> "DiagnosticConfig":
        """Create config from string level name."""
        level_map = {
            "none": DiagnosticLevel.NONE,
            "summary": DiagnosticLevel.SUMMARY,
            "verbose": DiagnosticLevel.VERBOSE,
            "debug": DiagnosticLevel.DEBUG,
        }
        diag_level = level_map.get(level.lower(), DiagnosticLevel.SUMMARY)

        if diag_level == DiagnosticLevel.NONE:
            return cls(
                level=diag_level,
                collect_spike_timing=False,
                collect_weight_changes=False,
                collect_mechanism_states=False,
                collect_winner_consistency=False,
                track_weight_snapshots=False,
            )
        elif diag_level == DiagnosticLevel.SUMMARY:
            return cls(level=diag_level)
        elif diag_level == DiagnosticLevel.VERBOSE:
            return cls(
                level=diag_level,
                report_every_n_cycles=10,
                early_cycle_details=10,
            )
        else:  # DEBUG
            return cls(
                level=diag_level,
                report_every_n_cycles=1,
                early_cycle_details=20,
                weight_snapshot_interval=5,
            )


@dataclass
class MechanismConfig:
    """
    Enable/disable specific neural mechanisms for ablation studies.

    All mechanisms enabled by default. Disable specific ones to isolate
    their contribution to behavior.

    Attributes:
        # Synaptic mechanisms
        enable_stp: Short-term plasticity (facilitation/depression)
        enable_nmda: NMDA receptor dynamics (voltage-gated, slow)

        # Inhibition mechanisms
        enable_som_inhibition: SOM+ interneuron-like divisive inhibition
        enable_lateral_inhibition: Distance-based lateral inhibition
        enable_shunting_inhibition: Shunting (divisive) inhibition

        # Adaptation mechanisms
        enable_sfa: Spike-frequency adaptation
        enable_som_adaptation: SOM-like slow adaptation current

        # Homeostatic mechanisms
        enable_homeostasis: Firing rate homeostasis (g_tonic/excitability)
        enable_bcm: BCM threshold dynamics
        enable_synaptic_scaling: Global synaptic scaling

        # Learning mechanisms
        enable_feedforward_learning: Hebbian learning on input weights
        enable_recurrent_learning: Hebbian learning on recurrent weights
        enable_heterosynaptic_ltd: Heterosynaptic LTD (weight competition)
        enable_stp_gated_learning: STP modulation of learning rate

        # Other
        enable_theta_modulation: Theta rhythm phase-based modulation
        enable_neuromodulation: Dopamine/other neuromodulators
    """
    # Synaptic mechanisms
    enable_stp: bool = True
    enable_nmda: bool = True

    # Inhibition mechanisms
    enable_som_inhibition: bool = True
    enable_lateral_inhibition: bool = True
    enable_shunting_inhibition: bool = True

    # Adaptation mechanisms
    enable_sfa: bool = True
    enable_som_adaptation: bool = True

    # Homeostatic mechanisms
    enable_homeostasis: bool = True
    enable_bcm: bool = True
    enable_synaptic_scaling: bool = True

    # Learning mechanisms
    enable_feedforward_learning: bool = True
    enable_recurrent_learning: bool = True
    enable_heterosynaptic_ltd: bool = True
    enable_stp_gated_learning: bool = True

    # Other
    enable_theta_modulation: bool = True
    enable_neuromodulation: bool = True

    def get_disabled_mechanisms(self) -> List[str]:
        """Return list of disabled mechanism names."""
        disabled = []
        for name, value in self.__dict__.items():
            if name.startswith("enable_") and not value:
                # Convert enable_stp -> STP
                mechanism_name = name[7:].upper().replace("_", " ")
                disabled.append(mechanism_name)
        return disabled

    def summary(self) -> str:
        """Return human-readable summary of config."""
        disabled = self.get_disabled_mechanisms()
        if not disabled:
            return "All mechanisms enabled"
        return f"Disabled: {', '.join(disabled)}"

    @classmethod
    def all_disabled(cls) -> "MechanismConfig":
        """Create config with all mechanisms disabled (baseline)."""
        return cls(
            enable_stp=False,
            enable_nmda=False,
            enable_som_inhibition=False,
            enable_lateral_inhibition=False,
            enable_shunting_inhibition=False,
            enable_sfa=False,
            enable_som_adaptation=False,
            enable_homeostasis=False,
            enable_bcm=False,
            enable_synaptic_scaling=False,
            enable_feedforward_learning=False,
            enable_recurrent_learning=False,
            enable_heterosynaptic_ltd=False,
            enable_stp_gated_learning=False,
            enable_theta_modulation=False,
            enable_neuromodulation=False,
        )

    @classmethod
    def minimal_learning(cls) -> "MechanismConfig":
        """Config with just basic Hebbian learning, no bells and whistles."""
        config = cls.all_disabled()
        config.enable_feedforward_learning = True
        config.enable_lateral_inhibition = True  # Need some competition
        return config
