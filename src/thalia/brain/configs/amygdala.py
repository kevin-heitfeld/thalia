"""Configurations for amygdala regions: BasolateralAmygdala (BLA) and CentralAmygdala (CeA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from thalia.brain.regions.population_names import BLAPopulation, CeAPopulation
from thalia.brain.synapses import NMReceptorType
from thalia.errors import ConfigurationError
from thalia.typing import NeuromodulatorChannel, PopulationPolarity

from .neural_region import NMReceptorConfig, NeuralPopulationConfig, NeuralRegionConfig, SynapticScalingConfig


# ---------------------------------------------------------------------------
# Per-population config classes
# ---------------------------------------------------------------------------

@dataclass
class BLAPopulationConfig(NeuralPopulationConfig):
    """Per-population biophysical parameters for BLA populations."""

    polarity: PopulationPolarity = PopulationPolarity.EXCITATORY
    """Dale's law polarity for this population."""


@dataclass
class CeAPopulationConfig(NeuralPopulationConfig):
    """Per-population biophysical parameters for CeA populations."""

    polarity: PopulationPolarity = PopulationPolarity.INHIBITORY
    """Dale's law polarity for this population."""

    baseline_drive: float = 0.0
    """Tonic excitatory drive for this population."""


@dataclass
class AmygdalaNucleusConfig(NeuralRegionConfig):
    """Shared configuration base for amygdala nuclei (BLA, CeA).

    Captures biophysical parameters common across amygdala sub-regions.
    """

    tau_ref: float = 4.0
    """Refractory period in ms (shared by BLA principal/SOM neurons and CeA neurons)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau_ref <= 0:
            raise ConfigurationError(f"tau_ref must be > 0, got {self.tau_ref}")


@dataclass
class BasolateralAmygdalaConfig(AmygdalaNucleusConfig):
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

    # Synaptic conductances
    baseline_drive: float = 0.0005
    """Tonic excitatory drive (very low - BLA is mostly silent at baseline)."""

    # Learning
    learning_rate: float = 0.002
    """STDP learning rate for CS–US association (slow for stable fear memories)."""

    synaptic_scaling: SynapticScalingConfig = field(default_factory=lambda: SynapticScalingConfig(
        w_min=0.0,
        w_max=5.0,
    ))
    """Synaptic scaling with wider weight range for BLA fear conditioning."""

    homeostatic_target_rates: dict[str, float] = field(default_factory=lambda: {
        BLAPopulation.PRINCIPAL: 0.003,
    })

    neuromodulator_receptors: list[NMReceptorConfig] = field(default_factory=lambda: [
        NMReceptorConfig(NMReceptorType.DA_D1, NeuromodulatorChannel.DA_MESOLIMBIC, "_da_concentration", (BLAPopulation.PRINCIPAL,)),
        NMReceptorConfig(NMReceptorType.NE_BETA, NeuromodulatorChannel.NE, "_ne_concentration", (BLAPopulation.PRINCIPAL,)),
        NMReceptorConfig(NMReceptorType.SHT_1A, NeuromodulatorChannel.SHT, "_sht_concentration", (BLAPopulation.PRINCIPAL,)),
        NMReceptorConfig(NMReceptorType.ACH_MUSCARINIC_M1, NeuromodulatorChannel.ACH, "_ach_concentration", (BLAPopulation.PRINCIPAL,)),
    ])

    population_overrides: Dict[str, BLAPopulationConfig] = field(
        default_factory=lambda: {
            BLAPopulation.PRINCIPAL: BLAPopulationConfig(
                tau_mem_ms=25.0,
                v_threshold=1.2,
                v_reset=-0.10,
                adapt_increment=0.30,
                tau_adapt_ms=200.0,
                noise_std=0.06,
                polarity=PopulationPolarity.EXCITATORY,
            ),
            BLAPopulation.PV: BLAPopulationConfig(
                tau_mem_ms=8.0,
                v_threshold=1.0,
                v_reset=0.0,
                adapt_increment=0.0,
                tau_adapt_ms=100.0,
                noise_std=0.08,
                polarity=PopulationPolarity.INHIBITORY,
            ),
            BLAPopulation.SOM: BLAPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=1.1,
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=200.0,
                noise_std=0.03,
                polarity=PopulationPolarity.INHIBITORY,
            ),
        }
    )
    """Per-population biophysical overrides."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.baseline_drive < 0:
            raise ConfigurationError(f"baseline_drive must be >= 0, got {self.baseline_drive}")


@dataclass
class CentralAmygdalaConfig(AmygdalaNucleusConfig):
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
    neuromodulator_receptors: list[NMReceptorConfig] = field(default_factory=lambda: [
        NMReceptorConfig(NMReceptorType.SHT_1A, NeuromodulatorChannel.SHT, "_sht_concentration", (CeAPopulation.MEDIAL,)),
    ])
    population_overrides: Dict[str, CeAPopulationConfig] = field(
        default_factory=lambda: {
            CeAPopulation.LATERAL: CeAPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=1.35,
                v_reset=0.0,
                adapt_increment=0.15,
                tau_adapt_ms=120.0,
                noise_std=0.08,
                polarity=PopulationPolarity.INHIBITORY,
                baseline_drive=0.0003,
            ),
            CeAPopulation.MEDIAL: CeAPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=1.215,  # 1.35 * 0.9: CeM is slightly easier to activate
                v_reset=0.0,
                adapt_increment=0.12,
                tau_adapt_ms=150.0,
                noise_std=0.08,
                polarity=PopulationPolarity.INHIBITORY,
                baseline_drive=0.0005,
            ),
        }
    )
    """Per-population biophysical overrides."""
