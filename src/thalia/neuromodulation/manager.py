"""
Neuromodulator Manager for DynamicBrain.

Centralizes management of VTA dopamine, LC norepinephrine, and NB acetylcholine systems.
"""

from typing import Dict, Any
import torch

from thalia.components.coding.spike_utils import compute_firing_rate
from thalia.neuromodulation.systems.vta import VTADopamineSystem, VTAConfig
from thalia.neuromodulation.systems.locus_coeruleus import LocusCoeruleusSystem, LocusCoeruleusConfig
from thalia.neuromodulation.systems.nucleus_basalis import NucleusBasalisSystem, NucleusBasalisConfig
from thalia.neuromodulation.homeostasis import NeuromodulatorCoordination


class NeuromodulatorManager:
    """Manages centralized neuromodulator systems and broadcasts to regions.

    Responsibilities:
    - Coordinate VTA (dopamine), LC (norepinephrine), NB (acetylcholine)
    - Handle inter-system interactions (DA-ACh, NE-ACh, DA-NE)
    - Broadcast neuromodulator levels to all brain regions
    - Compute signals: uncertainty, intrinsic reward, prediction error
    """

    def __init__(self):
        """Initialize neuromodulator systems."""
        # VTA DOPAMINE SYSTEM (reward prediction error)
        # Manages tonic + phasic dopamine, broadcasts to all regions
        self.vta = VTADopamineSystem(VTAConfig())

        # LOCUS COERULEUS (norepinephrine arousal)
        # Manages arousal/uncertainty, broadcasts NE to all regions
        self.locus_coeruleus = LocusCoeruleusSystem(LocusCoeruleusConfig())

        # NUCLEUS BASALIS (acetylcholine attention/encoding)
        # Manages encoding/retrieval mode, broadcasts ACh to cortex/hippocampus
        self.nucleus_basalis = NucleusBasalisSystem(NucleusBasalisConfig())

        # NEUROMODULATOR COORDINATION
        # Implements biological interactions between systems (DA-ACh, NE-ACh, DA-NE)
        self.coordination = NeuromodulatorCoordination()

    def broadcast_to_regions(self, regions: Dict[str, Any]) -> None:
        """Broadcast coordinated neuromodulator levels to all regions.

        Applies biological coordination between systems before broadcasting:
        - NE-ACh: Optimal encoding at moderate arousal (inverted-U)
        - DA-ACh: High reward without novelty suppresses encoding
        - DA-NE: High uncertainty + reward enhances both

        **Dopamine Projection Specificity**:
        Dopamine innervation varies dramatically across brain regions:
        - **Striatum**: Dense VTA/SNc projections (100% strength)
        - **Prefrontal Cortex**: Strong VTA projection (80% strength)
        - **Cortex**: Sparse, layer-specific innervation (30% strength, mainly L5)
        - **Hippocampus**: Minimal direct projection (10% strength, via septum/VTA)
        - **Thalamus**: Moderate SNc projection (20% strength)
        - **Cerebellum**: No dopamine (0% strength, uses climbing fiber errors)

        This matches biological reality where reward learning primarily affects
        striatum and PFC, with minimal direct effect on hippocampus and cerebellum.

        Args:
            regions: Dictionary of brain regions (cortex, hippocampus, pfc, striatum, cerebellum)
        """
        # Get raw neuromodulator signals
        dopamine_raw = self.vta.get_global_dopamine()
        norepinephrine = self.locus_coeruleus.get_norepinephrine()
        acetylcholine = self.nucleus_basalis.get_acetylcholine()

        # Apply biological coordination between systems
        # 1. NE-ACh: Optimal encoding at moderate arousal (inverted-U)
        acetylcholine = self.coordination.coordinate_ne_ach(
            norepinephrine, acetylcholine
        )

        # 2. DA-ACh: High reward without novelty suppresses encoding
        acetylcholine = self.coordination.coordinate_da_ach(
            dopamine_raw, acetylcholine
        )

        # 3. DA-NE: High uncertainty + reward enhances both
        # Note: Requires prediction error, which is computed in brain.py
        # This coordination is handled in brain._update_neuromodulators()

        # Biologically accurate dopamine projection strengths
        # Based on anatomical innervation density from VTA/SNc
        dopamine_projections = {
            "striatum": 1.0,        # Dense VTA/SNc projection (direct pathway)
            "prefrontal": 0.8,      # Strong VTA projection (mesocortical pathway)
            "pfc": 0.8,             # Alias for prefrontal
            "cortex": 0.3,          # Sparse, layer-specific (mainly L5)
            "hippocampus": 0.1,     # Minimal direct (mostly via septum/VTA)
            "thalamus": 0.2,        # Moderate SNc projection
            "cerebellum": 0.0,      # No dopamine (uses climbing fiber errors)
        }

        # Broadcast coordinated signals to all regions with region-specific DA scaling
        for region_name, region in regions.items():
            if hasattr(region, 'set_neuromodulators'):
                # Scale dopamine based on biological projection strength
                da_scale = dopamine_projections.get(region_name, 0.5)  # Default 50% for unknown regions
                dopamine_scaled = dopamine_raw * da_scale

                region.set_neuromodulators(dopamine_scaled, norepinephrine, acetylcholine)

    def compute_uncertainty(
        self,
        pfc_spikes: torch.Tensor,
        hippocampus_diagnostics: Dict[str, Any],
    ) -> float:
        """Compute uncertainty signal for LC.

        Args:
            pfc_spikes: Current PFC activity pattern
            hippocampus_diagnostics: Hippocampus diagnostic information

        Returns:
            Uncertainty level (0.0 to 1.0)
        """
        uncertainty = 0.0

        # 1. PFC conflict/uncertainty (when multiple goals compete)
        if pfc_spikes is not None:
            pfc_activity = compute_firing_rate(pfc_spikes)
            # High diffuse activity = uncertainty about what to remember
            if 0.3 < pfc_activity < 0.7:
                uncertainty += 0.3

        # 2. Hippocampus pattern matching uncertainty
        pattern_comparison = hippocampus_diagnostics.get('pattern_comparison', {})
        if 'dg_similarity' in pattern_comparison:
            similarity = pattern_comparison['dg_similarity']
            # Middle similarity = ambiguous (neither novel nor familiar)
            if 0.3 < similarity < 0.7:
                uncertainty += 0.4

        # 3. Novelty contributes to uncertainty
        if 'dg_similarity' in pattern_comparison:
            novelty = 1.0 - pattern_comparison['dg_similarity']
            uncertainty += 0.3 * novelty

        return min(1.0, uncertainty)

    def compute_intrinsic_reward(
        self,
        ca1_spikes: torch.Tensor,
        stored_pattern_exists: bool,
        novelty_signal: float,
    ) -> float:
        """Compute intrinsic reward from hippocampal activity.

        Args:
            ca1_spikes: CA1 output activity
            stored_pattern_exists: Whether hippocampus has stored pattern
            novelty_signal: Current novelty level

        Returns:
            Intrinsic reward value
        """
        intrinsic_reward = 0.0

        if ca1_spikes is not None:
            ca1_activity = compute_firing_rate(ca1_spikes)

            # MATCH: High CA1 activity when stored pattern exists
            if stored_pattern_exists and ca1_activity > 0.5:
                match_strength = (ca1_activity - 0.5) * 2.0
                intrinsic_reward += 0.3 * match_strength

            # NOVELTY: Moderate CA1 activity for new patterns
            if not stored_pattern_exists and 0.2 < ca1_activity < 0.5:
                intrinsic_reward += 0.2 * novelty_signal

        return intrinsic_reward

    def compute_prediction_error(
        self,
        predicted_value: float,
        actual_reward: float,
    ) -> float:
        """Compute TD prediction error.

        Args:
            predicted_value: Value predicted by critic
            actual_reward: Actual reward received

        Returns:
            Prediction error (RPE)
        """
        return actual_reward - predicted_value

    def deliver_reward(
        self,
        external_reward: float,
        intrinsic_reward: float,
        prediction_error: float,
    ) -> None:
        """Deliver reward signals to neuromodulator systems.

        Args:
            external_reward: Reward from environment
            intrinsic_reward: Internal curiosity/novelty reward
            prediction_error: TD prediction error
        """
        # Combine rewards
        total_reward = external_reward + intrinsic_reward

        # Update VTA with prediction error (phasic dopamine burst/dip)
        self.vta.process_reward(
            reward=total_reward,
            prediction_error=prediction_error,
        )

    def update_uncertainty(self, uncertainty: float) -> None:
        """Update LC with uncertainty signal.

        Args:
            uncertainty: Computed uncertainty level
        """
        self.locus_coeruleus.update_uncertainty(uncertainty)

    def update_encoding_mode(self, encoding: bool) -> None:
        """Update NB encoding/retrieval mode.

        Args:
            encoding: True for encoding mode, False for retrieval
        """
        if encoding:
            self.nucleus_basalis.enter_encoding_mode()
        else:
            self.nucleus_basalis.enter_retrieval_mode()

    def reset(self) -> None:
        """Reset all neuromodulator systems."""
        self.vta.reset()
        self.locus_coeruleus.reset()
        self.nucleus_basalis.reset()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get neuromodulator system diagnostics."""
        return {
            'dopamine': self.vta.get_global_dopamine(),
            'norepinephrine': self.locus_coeruleus.get_norepinephrine(),
            'acetylcholine': self.nucleus_basalis.get_acetylcholine(),
            'vta': self.vta.get_state(),
            'locus_coeruleus': self.locus_coeruleus.get_state(),
            'nucleus_basalis': self.nucleus_basalis.get_state(),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get checkpoint state for all neuromodulator systems."""
        return {
            'vta': self.vta.get_state() if hasattr(self.vta, 'get_state') else {},
            'locus_coeruleus': self.locus_coeruleus.get_state() if hasattr(self.locus_coeruleus, 'get_state') else {},
            'nucleus_basalis': self.nucleus_basalis.get_state() if hasattr(self.nucleus_basalis, 'get_state') else {},
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load checkpoint state for all neuromodulator systems."""
        if 'vta' in state_dict and hasattr(self.vta, 'load_state'):
            self.vta.load_state(state_dict['vta'])
        if 'locus_coeruleus' in state_dict and hasattr(self.locus_coeruleus, 'load_state'):
            self.locus_coeruleus.load_state(state_dict['locus_coeruleus'])
        if 'nucleus_basalis' in state_dict and hasattr(self.nucleus_basalis, 'load_state'):
            self.nucleus_basalis.load_state(state_dict['nucleus_basalis'])
