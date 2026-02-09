"""
Single source of truth for layer size calculations.

This module provides the LayerSizeCalculator class which handles all
size computations for brain regions using biologically-motivated ratios.

Replaces inconsistent calculation functions spread across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from thalia.typing import RegionLayerSizes


@dataclass
class BiologicalRatios:
    """Biologically-motivated ratios from neuroscience literature."""

    # =========================================================================
    # HIPPOCAMPUS RATIOS
    # =========================================================================

    dg_to_ec: float = 4.0
    """Dentate Gyrus to Entorhinal Cortex expansion ratio.

    Pattern separation via expansion: DG has ~4x more neurons than EC input.
    This expansion enables similar inputs to map to distinct representations.

    Biological range: 3-5x depending on species.
    """

    ca3_to_dg: float = 0.5
    """CA3 to Dentate Gyrus compression ratio.

    Pattern completion via recurrence: CA3 has ~50% as many neurons as DG.
    Smaller size enables dense recurrent connectivity (all-to-all feasible).

    Biological range: 0.3-0.5 (rat: ~300k CA3 vs ~1M DG).
    """

    ca2_to_dg: float = 0.25
    """CA2 to Dentate Gyrus compression ratio.

    Social memory hub: CA2 has ~25% as many neurons as DG (~50% of CA3 size).
    Small but crucial region for social memory and temporal context.
    """

    ca1_to_ca3: float = 1.0
    """CA1 to CA3 ratio (approximately equal).

    Output comparison: CA1 roughly matches CA3 in size.
    Compares direct EC input with CA3 output for novelty detection.

    Biological range: 0.8-1.2 depending on region measured.
    """

    # =========================================================================
    # CORTEX RATIOS
    # =========================================================================

    l4_to_input: float = 1.5
    """Layer 4 to input size expansion ratio.

    Input expansion: L4 (granular layer) expands thalamic input by ~1.5x.
    Increases representational capacity while maintaining specificity.
    """

    l23_to_l4: float = 2.0
    """Layer 2/3 to Layer 4 expansion ratio.

    Processing expansion: L2/3 are the largest cortical layers, ~2x L4 size.
    Enable extensive lateral integration and recurrent processing.

    Biology: L2/3 comprise ~40% of cortical neurons vs L4 ~20%.
    """

    l5_to_l23: float = 0.5
    """Layer 5 to Layer 2/3 compression ratio.

    Output compression: L5 is ~50% of L2/3 size.
    Projects to subcortical structures (striatum, thalamus, brainstem).

    Biology: L5 ~15% of cortical neurons vs L2/3 ~40%.
    """

    l6a_to_l23: float = 0.2
    """Layer 6a to Layer 2/3 ratio.

    Corticothalamic type I: L6a projects to TRN (inhibitory modulation).
    Smaller feedback layer supporting low gamma oscillations (25-35 Hz).
    """

    l6b_to_l23: float = 0.13
    """Layer 6b to Layer 2/3 ratio.

    Corticothalamic type II: L6b projects to relay (excitatory modulation).
    Supports high gamma oscillations (60-80 Hz).
    """

    # =========================================================================
    # STRIATUM RATIOS
    # =========================================================================

    msn_to_cortex: float = 0.5
    """Medium Spiny Neurons to cortical input ratio.

    Dimensionality reduction: Striatum has roughly 50% as many MSNs as
    cortical input neurons. Convergent input enables winner-take-all
    action selection.
    """

    d1_to_total: float = 0.5
    """D1 pathway to total MSN ratio (D1/D2 balance).

    Opponent pathways: D1 (Go) and D2 (NoGo) pathways are approximately
    equal in size, providing balanced action facilitation and suppression.

    Biological range: 0.45-0.55 (roughly equal populations).
    """

    # =========================================================================
    # CEREBELLUM RATIOS
    # =========================================================================

    granule_to_purkinje: float = 4.0
    """Granule cells to Purkinje cells expansion ratio.

    Expansion coding: Granule layer expands mossy fiber input.
    Biological ratio is much higher (~1000:1), but we use modest ratio
    for computational feasibility.
    """

    # =========================================================================
    # THALAMUS RATIOS
    # =========================================================================

    trn_to_relay: float = 0.3
    """Thalamic Reticular Nucleus to relay neuron ratio.

    Inhibitory modulation: TRN provides gating and attentional filtering.
    Smaller inhibitory population controls larger relay population.
    """


class LayerSizeCalculator:
    """Single source of truth for layer size calculations.

    All calculations based on documented biological ratios.
    Supports multiple specification patterns for flexibility.
    """

    def __init__(self, ratios: Optional[BiologicalRatios] = None):
        """Initialize calculator with biological ratios.

        Args:
            ratios: Custom biological ratios. If None, uses defaults.
        """
        self.ratios = ratios or BiologicalRatios()

    # =========================================================================
    # CEREBELLUM CALCULATIONS
    # =========================================================================

    def cerebellum_from_purkinje(self, purkinje_size: int) -> RegionLayerSizes:
        """Calculate cerebellum sizes from Purkinje cell count.

        Args:
            purkinje_size: Number of Purkinje cells (= output size)

        Returns:
            Dictionary with keys:
            - purkinje_size: Purkinje cell count (same as parameter)
            - granule_size: Granule cell count
        """
        granule_size = int(purkinje_size * self.ratios.granule_to_purkinje)

        return {
            "purkinje_size": purkinje_size,
            "granule_size": granule_size,
        }

    # =========================================================================
    # CORTEX CALCULATIONS
    # =========================================================================

    def cortex_from_output(self, target_output_size: int) -> RegionLayerSizes:
        """Calculate cortex layer sizes from target OUTPUT size.

        Pattern: Work backwards from desired output.
        Use when you know what you want cortex to OUTPUT.

        Args:
            target_output_size: Desired L2/3 + L5 output size

        Returns:
            Layer sizes that produce approximately target_output_size

        Note:
            Output = L2/3 + L5. With default ratios (L2/3:L5 = 2:1),
            L2/3 is ~67% and L5 is ~33% of output.
        """
        # With L2/3:L5 = 2:1, we have L2/3 + L5 = target
        # So L2/3 = target * 2/3, L5 = target * 1/3
        l23_size = int(target_output_size * 2 / 3)
        l5_size = int(target_output_size * 1 / 3)

        # Work backwards to L4 (L2/3 = L4 * 2.0, so L4 = L2/3 / 2.0)
        l4_size = int(l23_size / self.ratios.l23_to_l4)

        # Compute L6 from L2/3
        l6a_size = int(l23_size * self.ratios.l6a_to_l23)
        l6b_size = int(l23_size * self.ratios.l6b_to_l23)

        return {
            "l23_size": l23_size,
            "l4_size": l4_size,
            "l5_size": l5_size,
            "l6a_size": l6a_size,
            "l6b_size": l6b_size,
        }

    # =========================================================================
    # HIPPOCAMPUS CALCULATIONS
    # =========================================================================

    def hippocampus_from_input(self, ec_input_size: int) -> RegionLayerSizes:
        """Calculate hippocampus layer sizes from entorhinal cortex input.

        Pattern: EC → DG → CA3 → CA2 → CA1

        Args:
            ec_input_size: Size of entorhinal cortex input

        Returns:
            Dictionary with keys:
            - dg_size: Dentate Gyrus size
            - ca3_size: CA3 size
            - ca2_size: CA2 size
            - ca1_size: CA1 size (output layer)
            - input_size: EC input size (same as parameter)
        """
        dg_size = int(ec_input_size * self.ratios.dg_to_ec)
        ca3_size = int(dg_size * self.ratios.ca3_to_dg)
        ca2_size = int(dg_size * self.ratios.ca2_to_dg)
        ca1_size = int(ca3_size * self.ratios.ca1_to_ca3)

        return {
            "dg_size": dg_size,
            "ca3_size": ca3_size,
            "ca2_size": ca2_size,
            "ca1_size": ca1_size,
        }

    # =========================================================================
    # PREFRONTAL CORTEX CALCULATIONS
    # =========================================================================

    def pfc_from_executive(self, executive_size: int) -> RegionLayerSizes:
        """Calculate prefrontal cortex sizes from executive layer size.

        Pattern: Executive layer size determines overall PFC size.
        Use when you want to specify the size of the executive layer directly.

        Args:
            executive_size: Number of neurons in the executive layer
        """
        return {
            "executive_size": executive_size,
        }

    # =========================================================================
    # STRIATUM CALCULATIONS
    # =========================================================================

    def striatum_from_actions(self, n_actions: int, neurons_per_action: int) -> RegionLayerSizes:
        """Calculate striatum sizes from number of actions.

        Pattern: Population coding with D1/D2 opponent pathways.

        **ARCHITECTURE**: Each action has neurons in BOTH D1 and D2 pathways.
        - D1 pathway: n_actions × neurons_per_action (Go signals)
        - D2 pathway: n_actions × neurons_per_action (NoGo signals)
        - Total neurons: 2 × n_actions × neurons_per_action

        Args:
            n_actions: Number of discrete actions
            neurons_per_action: Neurons per action per pathway (default: 10)

        Returns:
            Dictionary with keys:
            - d1_size: D1 (Go pathway) size = n_actions × neurons_per_action
            - d2_size: D2 (NoGo pathway) size = n_actions × neurons_per_action
            - n_actions: Number of actions (same as parameter)
            - neurons_per_action: Neurons per action per pathway
        """
        # Each pathway gets full neuron allocation per action
        # This allows each action to have independent Go/NoGo populations
        neurons_per_action_per_pathway = n_actions * neurons_per_action

        return {
            "d1_size": neurons_per_action_per_pathway,
            "d2_size": neurons_per_action_per_pathway,
            "n_actions": n_actions,
            "neurons_per_action": neurons_per_action,
        }

    # =========================================================================
    # THALAMUS CALCULATIONS
    # =========================================================================

    def thalamus_from_relay(self, relay_size: int) -> RegionLayerSizes:
        """Calculate thalamus sizes from relay neuron count.

        Args:
            relay_size: Number of relay neurons (= output size)

        Returns:
            Dictionary with keys:
            - relay_size: Relay neuron count (same as parameter)
            - trn_size: TRN neuron count
            - input_size: Relay size (for 1:1 relay pattern)
        """
        trn_size = int(relay_size * self.ratios.trn_to_relay)

        return {
            "relay_size": relay_size,
            "trn_size": trn_size,
        }
