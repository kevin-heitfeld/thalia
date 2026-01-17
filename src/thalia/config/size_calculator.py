"""
Single source of truth for layer size calculations.

This module provides the LayerSizeCalculator class which handles all
size computations for brain regions using biologically-motivated ratios.

Replaces inconsistent calculation functions spread across the codebase.

Author: Thalia Project
Date: January 10, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class BiologicalRatios:
    """Biologically-motivated ratios from neuroscience literature.

    All ratios documented with references.

    References:
        Hippocampus: Amaral & Witter (1989)
        Cortex: Douglas & Martin (2004)
        Striatum: Gerfen & Surmeier (2011)
        Cerebellum: Ito (2006)
        Thalamus: Sherman & Guillery (2006)
    """

    # =========================================================================
    # HIPPOCAMPUS RATIOS (Amaral & Witter, 1989)
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

    Reference: Hitti & Siegelbaum (2014)
    """

    ca1_to_ca3: float = 1.0
    """CA1 to CA3 ratio (approximately equal).

    Output comparison: CA1 roughly matches CA3 in size.
    Compares direct EC input with CA3 output for novelty detection.

    Biological range: 0.8-1.2 depending on region measured.
    """

    # =========================================================================
    # CORTEX RATIOS (Douglas & Martin, 2004)
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

    Reference: Sherman & Guillery (2002)
    """

    l6b_to_l23: float = 0.13
    """Layer 6b to Layer 2/3 ratio.

    Corticothalamic type II: L6b projects to relay (excitatory modulation).
    Supports high gamma oscillations (60-80 Hz).

    Reference: Sherman & Guillery (2002)
    """

    # =========================================================================
    # STRIATUM RATIOS (Gerfen & Surmeier, 2011)
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
    # CEREBELLUM RATIOS (Ito, 2006)
    # =========================================================================

    granule_to_purkinje: float = 4.0
    """Granule cells to Purkinje cells expansion ratio.

    Expansion coding: Granule layer expands mossy fiber input.
    Biological ratio is much higher (~1000:1), but we use modest ratio
    for computational feasibility.
    """

    # =========================================================================
    # THALAMUS RATIOS (Sherman & Guillery, 2006)
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

    Usage:
        >>> calc = LayerSizeCalculator()
        >>> sizes = calc.cortex_from_input(input_size=192)
        >>> config = LayeredCortexConfig(**sizes)

        >>> # Or with custom ratios:
        >>> custom_ratios = BiologicalRatios(l23_to_l4=3.0)
        >>> calc = LayerSizeCalculator(ratios=custom_ratios)
        >>> sizes = calc.cortex_from_scale(scale_factor=128)
    """

    def __init__(self, ratios: BiologicalRatios | None = None):
        """Initialize calculator with biological ratios.

        Args:
            ratios: Custom biological ratios. If None, uses defaults.
        """
        self.ratios = ratios or BiologicalRatios()

    # =========================================================================
    # CORTEX CALCULATIONS
    # =========================================================================

    def cortex_from_input(self, input_size: int) -> Dict[str, int]:
        """Calculate cortex layer sizes from INPUT size.

        Pattern: Input → L4 → L2/3 → L5 → L6a/L6b
        Use when you know what's connecting TO cortex.

        Args:
            input_size: Total input from all sources (thalamus + other cortex + hippocampus)

        Returns:
            Dictionary with keys:
            - l4_size: Layer 4 (input) size
            - l23_size: Layer 2/3 (processing) size
            - l5_size: Layer 5 (subcortical output) size
            - l6a_size: Layer 6a (TRN feedback) size
            - l6b_size: Layer 6b (relay feedback) size
            - input_size: Total input size (same as parameter)
            - output_size: L2/3 + L5 (dual output pathways)
            - total_neurons: Sum of all layers

        Example:
            >>> calc = LayerSizeCalculator()
            >>> # Cortex receives from thalamus (64) + hippocampus (128) = 192
            >>> sizes = calc.cortex_from_input(input_size=192)
            >>> sizes['l4_size']  # 288 (192 * 1.5)
            >>> sizes['output_size']  # 864 (576 + 288)
        """
        l4_size = int(input_size * self.ratios.l4_to_input)
        l23_size = int(l4_size * self.ratios.l23_to_l4)
        l5_size = int(l23_size * self.ratios.l5_to_l23)
        l6a_size = int(l23_size * self.ratios.l6a_to_l23)
        l6b_size = int(l23_size * self.ratios.l6b_to_l23)

        return {
            "l4_size": l4_size,
            "l23_size": l23_size,
            "l5_size": l5_size,
            "l6a_size": l6a_size,
            "l6b_size": l6b_size,
            "input_size": input_size,
            "output_size": l23_size + l5_size,  # Dual output pathways
            "total_neurons": l4_size + l23_size + l5_size + l6a_size + l6b_size,
        }

    def cortex_from_output(self, target_output_size: int) -> Dict[str, int]:
        """Calculate cortex layer sizes from target OUTPUT size.

        Pattern: Work backwards from desired output.
        Use when you know what you want cortex to OUTPUT.

        Args:
            target_output_size: Desired L2/3 + L5 output size

        Returns:
            Layer sizes that produce approximately target_output_size
            (same keys as cortex_from_input)

        Note:
            Output = L2/3 + L5. With default ratios (L2/3:L5 = 2:1),
            L2/3 is ~67% and L5 is ~33% of output.

        Example:
            >>> calc = LayerSizeCalculator()
            >>> sizes = calc.cortex_from_output(target_output_size=300)
            >>> sizes['l23_size']  # ~200 (300 * 2/3)
            >>> sizes['l5_size']   # ~100 (300 * 1/3)
            >>> sizes['output_size']  # ~300
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

        # Infer input size from L4
        input_size = int(l4_size / self.ratios.l4_to_input)

        return {
            "l4_size": l4_size,
            "l23_size": l23_size,
            "l5_size": l5_size,
            "l6a_size": l6a_size,
            "l6b_size": l6b_size,
            "input_size": input_size,
            "output_size": l23_size + l5_size,
            "total_neurons": l4_size + l23_size + l5_size + l6a_size + l6b_size,
        }

    def cortex_from_scale(self, scale_factor: int) -> Dict[str, int]:
        """Calculate cortex layer sizes from SCALE factor.

        Pattern: Scale all layers proportionally.
        Use when you want "small/medium/large" cortex without caring about specifics.

        Args:
            scale_factor: Base multiplier for L4 size (e.g., 32, 64, 128, 256)

        Returns:
            Proportionally scaled layers (same keys as cortex_from_input)

        Example:
            >>> calc = LayerSizeCalculator()
            >>> small = calc.cortex_from_scale(scale_factor=64)
            >>> medium = calc.cortex_from_scale(scale_factor=128)
            >>> large = calc.cortex_from_scale(scale_factor=256)
        """
        # Use scale_factor as L4 base, apply standard ratios
        l4_size = scale_factor
        l23_size = int(scale_factor * self.ratios.l23_to_l4)
        l5_size = int(l23_size * self.ratios.l5_to_l23)
        l6a_size = int(l23_size * self.ratios.l6a_to_l23)
        l6b_size = int(l23_size * self.ratios.l6b_to_l23)

        # Infer input size from L4
        input_size = int(l4_size / self.ratios.l4_to_input)

        return {
            "l4_size": l4_size,
            "l23_size": l23_size,
            "l5_size": l5_size,
            "l6a_size": l6a_size,
            "l6b_size": l6b_size,
            "input_size": input_size,
            "output_size": l23_size + l5_size,
            "total_neurons": l4_size + l23_size + l5_size + l6a_size + l6b_size,
        }

    # =========================================================================
    # HIPPOCAMPUS CALCULATIONS
    # =========================================================================

    def hippocampus_from_input(self, ec_input_size: int) -> Dict[str, int]:
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
            - output_size: CA1 size (output)
            - total_neurons: Sum of all layers

        Example:
            >>> calc = LayerSizeCalculator()
            >>> sizes = calc.hippocampus_from_input(ec_input_size=100)
            >>> sizes['dg_size']  # 400 (100 * 4.0, pattern separation)
            >>> sizes['ca1_size']  # 200 (output layer)
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
            "input_size": ec_input_size,
            "output_size": ca1_size,
            "total_neurons": dg_size + ca3_size + ca2_size + ca1_size,
        }

    # =========================================================================
    # STRIATUM CALCULATIONS
    # =========================================================================

    def striatum_from_actions(self, n_actions: int, neurons_per_action: int = 10) -> Dict[str, int]:
        """Calculate striatum sizes from number of actions.

        Pattern: Population coding with D1/D2 opponent pathways.

        **IMPORTANT**: D1 and D2 pathways always get equal neuron counts.
        If neurons_per_action is odd, it will be rounded down to nearest even number.

        Args:
            n_actions: Number of discrete actions
            neurons_per_action: Neurons per action (default: 10 for noise reduction)
                Must be even for symmetric D1/D2 splits.

        Returns:
            Dictionary with keys:
            - d1_size: D1 (Go pathway) size
            - d2_size: D2 (NoGo pathway) size
            - n_actions: Number of actions (same as parameter)
            - neurons_per_action: Adjusted neurons per action (even number)
            - output_size: D1 + D2 (both pathways output spikes)
            - total_neurons: D1 + D2

        Example:
            >>> calc = LayerSizeCalculator()
            >>> sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=10)
            >>> sizes['d1_size']  # 20 (4 actions * 5 neurons)
            >>> sizes['d2_size']  # 20 (4 actions * 5 neurons)
            >>> sizes['output_size']  # 40 (20 + 20)
        """
        # Ensure even number for symmetric D1/D2 split
        if neurons_per_action < 2:
            neurons_per_action = 2
        elif neurons_per_action % 2 != 0:
            neurons_per_action = neurons_per_action + 1  # Round up to even

        neurons_per_pathway = neurons_per_action // 2
        d1_size = n_actions * neurons_per_pathway
        d2_size = n_actions * neurons_per_pathway

        return {
            "d1_size": d1_size,
            "d2_size": d2_size,
            "n_actions": n_actions,
            "neurons_per_action": neurons_per_action,
            "output_size": d1_size + d2_size,  # Both pathways output spikes
            "total_neurons": d1_size + d2_size,
        }

    # =========================================================================
    # CEREBELLUM CALCULATIONS
    # =========================================================================

    def cerebellum_from_output(self, purkinje_size: int) -> Dict[str, int]:
        """Calculate cerebellum sizes from Purkinje cell count.

        Args:
            purkinje_size: Number of Purkinje cells (= output size)

        Returns:
            Dictionary with keys:
            - purkinje_size: Purkinje cell count (same as parameter)
            - granule_size: Granule cell count
            - basket_size: Basket cell count (~10% of Purkinje)
            - stellate_size: Stellate cell count (~5% of Purkinje)
            - output_size: Purkinje size (output)
            - total_neurons: Sum of all cell types

        Example:
            >>> calc = LayerSizeCalculator()
            >>> sizes = calc.cerebellum_from_output(purkinje_size=100)
            >>> sizes['granule_size']  # 400 (100 * 4.0, expansion coding)
        """
        granule_size = int(purkinje_size * self.ratios.granule_to_purkinje)
        basket_size = int(purkinje_size * 0.1)  # ~10% basket cells
        stellate_size = int(purkinje_size * 0.05)  # ~5% stellate cells

        return {
            "purkinje_size": purkinje_size,
            "granule_size": granule_size,
            "basket_size": basket_size,
            "stellate_size": stellate_size,
            "output_size": purkinje_size,
            "total_neurons": purkinje_size + granule_size + basket_size + stellate_size,
        }

    # =========================================================================
    # THALAMUS CALCULATIONS
    # =========================================================================

    def thalamus_from_relay(self, relay_size: int) -> Dict[str, int]:
        """Calculate thalamus sizes from relay neuron count.

        Args:
            relay_size: Number of relay neurons (= output size)

        Returns:
            Dictionary with keys:
            - relay_size: Relay neuron count (same as parameter)
            - trn_size: TRN neuron count
            - input_size: Relay size (for 1:1 relay pattern)
            - output_size: Relay size
            - total_neurons: Relay + TRN

        Example:
            >>> calc = LayerSizeCalculator()
            >>> sizes = calc.thalamus_from_relay(relay_size=128)
            >>> sizes['trn_size']  # 38 (128 * 0.3, inhibitory modulation)
        """
        trn_size = int(relay_size * self.ratios.trn_to_relay)

        return {
            "relay_size": relay_size,
            "trn_size": trn_size,
            "input_size": relay_size,  # Typically 1:1 relay
            "output_size": relay_size,
            "total_neurons": relay_size + trn_size,
        }
