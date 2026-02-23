"""Configurations for amygdala regions: BasolateralAmygdala (BLA) and CentralAmygdala (CeA)."""

from __future__ import annotations

from dataclasses import dataclass

from thalia.errors import ConfigurationError

from .neural_region import NeuralRegionConfig


@dataclass
class BasolateralAmygdalaConfig(NeuralRegionConfig):
    """Configuration for BLA (basolateral amygdala) region.

    The BLA is the primary site for fear conditioning and extinction learning.
    It contains glutamatergic principal neurons that form CS–US associations via
    Hebbian/STDP-like plasticity, gated by local inhibitory networks.

    Key features:
    - Glutamatergic principal neurons with strong recurrent excitation
    - PV interneurons for feedforward inhibition (fear gating, gamma oscillations)
    - SOM interneurons for dendritic inhibition (extinction, behavioural flexibility)
    - Receives convergent cortical, thalamic, and hippocampal input
    - Projects to CeA (fear output), striatum (approach/avoidance), PFC (cognition)

    Biological parameters (Pape & Paré 2010; Herry & Johansen 2014):
    - ~50% principal / 20% PV / 10% SOM neurons
    - STDP window: ±25ms for CS–US association
    - Theta oscillations (~8 Hz) gate consolidation
    """

    # Tau (ms)
    tau_mem_principal: float = 25.0
    """Membrane time constant for principal neurons (slow, enables temporal integration)."""

    tau_mem_pv: float = 8.0
    """Membrane time constant for PV interneurons (fast, feedforward gating)."""

    tau_mem_som: float = 20.0
    """Membrane time constant for SOM interneurons (moderate, dendritic inhibition)."""

    # Excitability
    v_threshold_principal: float = 0.9
    """Firing threshold for principal neurons (slightly below standard)."""

    v_threshold_pv: float = 1.5
    """High threshold for PV interneurons to prevent hyperactivity."""

    tau_ref: float = 4.0
    """Refractory period in ms."""

    # Synaptic conductances
    baseline_drive: float = 0.0005
    """Tonic excitatory drive (very low - BLA is mostly silent at baseline)."""

    # Learning
    learning_rate: float = 0.002
    """STDP learning rate for CS–US association (slow for stable fear memories)."""

    w_min: float = 0.0
    """Minimum synaptic weight."""

    w_max: float = 5.0
    """Maximum synaptic weight."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau_mem_principal <= 0:
            raise ConfigurationError(f"tau_mem_principal must be > 0, got {self.tau_mem_principal}")
        if self.tau_mem_pv <= 0:
            raise ConfigurationError(f"tau_mem_pv must be > 0, got {self.tau_mem_pv}")
        if self.tau_mem_som <= 0:
            raise ConfigurationError(f"tau_mem_som must be > 0, got {self.tau_mem_som}")
        if self.tau_ref <= 0:
            raise ConfigurationError(f"tau_ref must be > 0, got {self.tau_ref}")
        if self.baseline_drive < 0:
            raise ConfigurationError(f"baseline_drive must be >= 0, got {self.baseline_drive}")


@dataclass
class CentralAmygdalaConfig(NeuralRegionConfig):
    """Configuration for CeA (central amygdala) region.

    The CeA is the output nucleus of the amygdala, receiving BLA input and
    projecting to hypothalamus, LC, LHb, and brainstem. It contains two
    main populations: lateral CeA (CeL, integrative) and medial CeA (CeM, output).

    Key features:
    - CeL ('lateral') contains fear-ON and fear-OFF neurons (PKCδ+ / PKCδ- cells)
    - CeM ('medial') is the main output: drives autonomic/behavioural fear responses
    - CeL → CeM lateral inhibition enables opponent-process fear regulation
    - Receives BLA principal neuron input as main driver
    - CeM → LC drives NE-mediated arousal during fear
    - CeM → LHb drives aversive RPE signal (negative DA)

    Biological parameters (Ciocchi et al. 2010; Haubensak et al. 2010):
    - CeL ≈ 60% / CeM ≈ 40% population split (approx)
    - ~90% GABAergic interneurons (CeA is almost entirely inhibitory)
    - ~10% glutamatergic projection neurons (CeM output to brainstem)
    """

    tau_mem: float = 20.0
    """Membrane time constant (ms)."""

    v_threshold: float = 1.0
    """Firing threshold."""

    tau_ref: float = 4.0
    """Refractory period (ms)."""

    baseline_drive_lateral: float = 0.0003
    """Tonic drive to CeL (mostly silent)."""

    baseline_drive_medial: float = 0.0005
    """Tonic drive to CeM (slightly more active, drives background autonomic tone)."""
