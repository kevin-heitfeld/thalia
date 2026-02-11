"""
Dynamic Brain - Component Graph Executor

This module implements a flexible brain architecture where the brain is
treated as a directed graph of neural components (regions and axons).

DynamicBrain supports:
- Arbitrary number of regions and axons
- Flexible topologies (not limited to fixed connectivity)
- User-defined custom regions via NeuralRegionRegistry
- Dynamic region addition/removal
- Plugin architecture for external regions
- Clock-driven execution with axonal delays

Architecture:
    DynamicBrain = Graph of components
    - nodes: regions (NeuralRegion)
    - edges: axons (AxonalProjection)
    - execution: clock-driven sequential
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, TypeVar, cast

import torch
import torch.nn as nn

from thalia.typing import (
    BrainSpikesDict,
    RegionSpikesDict,
    RegionName,
    SpikesSourceKey,
    compound_key,
    parse_compound_key,
)
from thalia.utils import compute_firing_rate, validate_spike_tensors

from .axonal_projection import AxonalProjection
from .configs import BrainConfig, NeuralRegionConfig
from .oscillator import OscillatorManager
from .regions import (
    NeuralRegion,
    Cortex,
    Hippocampus,
    RewardEncoder,
    Striatum,
)


RegionT = TypeVar('RegionT', bound=NeuralRegion[NeuralRegionConfig])


class DynamicBrain(nn.Module):
    """Dynamic brain constructed from graph.

    DynamicBrain builds arbitrary topologies from registered regions:
    - Flexible graph vs. hardcoded regions
    - User-extensible via NeuralRegionRegistry
    - Arbitrary connectivity patterns
    - Plugin support for external regions
    """

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self.config.device)

    @property
    def dt_ms(self) -> float:
        """Timestep duration in milliseconds."""
        return self.config.dt_ms

    @property
    def current_time(self) -> float:
        """Get current simulation time in milliseconds."""
        return self._current_time

    def get_region(self, region_name: RegionName, region_type: Type[RegionT]) -> Optional[RegionT]:
        """Get region by name and type, or None if not present."""
        region = self.regions[region_name] if region_name in self.regions else None
        if region is not None and isinstance(region, region_type):
            return cast(RegionT, region)
        return None

    def get_first_region_of_type(self, region_type: Type[RegionT]) -> Optional[RegionT]:
        """Get the first region of a given type, or None if not present."""
        for region in self.regions.values():
            if isinstance(region, region_type):
                return cast(RegionT, region)
        return None

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: BrainConfig,
        regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]],
        connections: Dict[Tuple[RegionName, SpikesSourceKey], AxonalProjection],
    ):
        """Initialize DynamicBrain from graph.

        Args:
            config: Brain configuration (device, dt_ms, oscillators, etc.)
            regions: Dict mapping region names to instances
            connections: Dict mapping (source, target) tuples to pathways
        """
        super().__init__()

        # DISABLE GRADIENTS
        # Thalia uses local learning rules (STDP, BCM, Hebbian, three-factor)
        # that do NOT require backpropagation. Disabling gradients provides:
        # - Performance boost (no autograd overhead)
        # - Memory savings (no gradient storage)
        # - Biological plausibility (no non-local error signals)
        torch.set_grad_enabled(False)

        # =================================================================
        # BRAIN CONFIG & COMPONENTS
        # =================================================================
        self.config = config

        # Store regions as nn.ModuleDict for proper parameter tracking
        self.regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]] = nn.ModuleDict(regions)

        # Store connections with tuple keys for easy lookup
        # Also register in ModuleDict for parameter tracking
        self.connections: Dict[Tuple[RegionName, SpikesSourceKey], AxonalProjection] = connections
        self._connection_modules = nn.ModuleDict(
            {f"{src}_to_{tgt}": pathway for (src, tgt), pathway in connections.items()}
        )

        # Current simulation time
        self._current_time: float = 0.0

        # =================================================================
        # OSCILLATOR MANAGER
        # =================================================================
        self.oscillators = OscillatorManager(
            dt_ms=self.dt_ms,
            device=self.device,
            delta_freq=self.config.delta_frequency_hz,
            theta_freq=self.config.theta_frequency_hz,
            alpha_freq=self.config.alpha_frequency_hz,
            beta_freq=self.config.beta_frequency_hz,
            gamma_freq=self.config.gamma_frequency_hz,
            ripple_freq=self.config.ripple_frequency_hz,
            couplings=None,  # Use default couplings (theta-gamma, etc.)
        )

        # =================================================================
        # REINFORCEMENT LEARNING STATE
        # =================================================================
        self._last_action: Optional[int] = None

        # =================================================================
        # INITIALIZE TEMPORAL PARAMETERS
        # =================================================================
        # Broadcast dt_ms to all regions for initial setup
        # Regions compute decay factors, phase increments, etc.
        self.set_timestep(self.dt_ms)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, input_spikes: Optional[BrainSpikesDict] = None) -> BrainSpikesDict:
        """Run one timestep of the brain."""
        if input_spikes is None:
            input_spikes = {}
        else:
            for region_name, spikes in input_spikes.items():
                validate_spike_tensors(spikes, context=f"{self.__class__.__name__}.forward")

        # === CLOCK-DRIVEN EXECUTION (ADR-003) ===
        # All regions execute every timestep.
        # Axonal delays are handled by pathway CircularDelayBuffers.
        # This ensures continuous dynamics: membrane decay, recurrent connections,
        # oscillators, and short-term plasticity all evolve every timestep.

        # =====================================================================
        # SPIKE PROPAGATION
        # =====================================================================
        # At timestep T:
        # 1. Read delayed outputs from buffers (spikes sent at T-delay)
        # 2. Combine with external input_spikes
        # 3. Execute regions → produce outputs at T
        # 4. Write outputs to buffers → will be read at T+delay
        #
        # At T=0: buffers are empty, so delayed_outputs are zeros
        # At T>=delay: buffers have data, delayed_outputs have actual spikes
        # =====================================================================

        # STEP 1: Read delayed outputs from all pathways (spikes from T-delay)
        region_inputs: BrainSpikesDict = {name: {} for name in self.regions.keys()}

        for (_src, _tgt), pathway in self.connections.items():
            # Read delayed outputs WITHOUT writing or advancing
            delayed_outputs: BrainSpikesDict = pathway.read_delayed_outputs()

            # Parse target region from compound key "target:population" (connection tuple)
            target_region, _target_population = parse_compound_key(_tgt)

            # Route inputs by SOURCE name (biological: synapses ARE the routing)
            for source_region, population_dict in delayed_outputs.items():
                if target_region not in region_inputs:
                    region_inputs[target_region] = {}

                for source_population, delayed_tensor in population_dict.items():
                    # Construct source_name key matching synaptic_weights keys
                    # e.g., "thalamus:relay", "hippocampus:ca1"
                    source_name = compound_key(source_region, source_population)

                    # Combine if source already has input from another pathway (logical OR)
                    if source_name in region_inputs[target_region]:
                        region_inputs[target_region][source_name] = (
                            region_inputs[target_region][source_name] | delayed_tensor
                        )
                    else:
                        region_inputs[target_region][source_name] = delayed_tensor

        # STEP 2: Combine with external sensory input_spikes
        for region_name, population_inputs in input_spikes.items():
            if region_name not in region_inputs:
                region_inputs[region_name] = {}

            for population_name, population_input in population_inputs.items():
                # Combine sensory with pathway inputs (logical OR)
                if population_name in region_inputs[region_name]:
                    region_inputs[region_name][population_name] = (
                        region_inputs[region_name][population_name] | population_input
                    )
                else:
                    region_inputs[region_name][population_name] = population_input

        # STEP 3: Execute all regions with combined inputs → produce outputs at T
        outputs: BrainSpikesDict = {}
        for region_name, region in self.regions.items():
            # Get accumulated inputs for this region (empty dict if no inputs)
            region_input: RegionSpikesDict = region_inputs.get(region_name, {})

            # Execute region (empty dict is valid for recurrent/spontaneous activity)
            region_output: RegionSpikesDict = region.forward(region_input)

            # Store output
            outputs[region_name] = region_output

        # STEP 4: Write current outputs to pathways and advance buffers
        for (src, _tgt), pathway in self.connections.items():
            # Write source region's current output to buffer (for T+delay reads)
            if src in outputs:
                source_output: BrainSpikesDict = {src: outputs[src]}
                pathway.write_and_advance(source_output)

        # Advance simulation time
        self._current_time += self.dt_ms

        return outputs

    def consolidate(self, duration_ms: float) -> Dict[str, Any]:
        """Trigger memory consolidation via spontaneous replay.

        No explicit coordination - just set acetylcholine low and run brain forward.
        Hippocampus spontaneously replays high-priority patterns via sharp-wave ripples.

        Biological mechanism:
        1. Lower acetylcholine (0.1) → sleep/rest mode
        2. Hippocampus CA3 spontaneously reactivates stored patterns
        3. Replay probability ∝ synaptic tag strength (Frey-Morris tagging)
        4. Ripples occur at ~1-3 Hz during low ACh
        5. Restore acetylcholine (0.7) → awake/encoding mode

        Args:
            duration_ms: Consolidation duration in milliseconds (default 1000ms = 1 second)

        Returns:
            Dict with consolidation statistics:
                - ripples: Number of sharp-wave ripples detected
                - duration_ms: Total consolidation duration
                - ripple_rate_hz: Ripples per second

        Raises:
            ValueError: If hippocampus not present in brain
        """
        # Check for hippocampus
        hippocampus = self.get_first_region_of_type(Hippocampus)
        if hippocampus is None:
            raise ValueError(
                "Hippocampus required for consolidation. "
                "Brain must include 'hippocampus' region."
            )

        # Enter consolidation mode (low acetylcholine)
        hippocampus.enter_consolidation_mode()
        hippocampus.set_neuromodulators(acetylcholine=0.1)

        # Run brain forward - replay happens automatically
        n_timesteps = int(duration_ms / self.dt_ms)
        ripple_count = 0

        for _ in range(n_timesteps):
            self.forward(None)  # No sensory input during sleep
            if hippocampus.ripple_detected:
                ripple_count += 1

        # Return to encoding mode (high acetylcholine)
        hippocampus.set_neuromodulators(acetylcholine=0.7)
        hippocampus.exit_consolidation_mode()

        # Compute ripple rate
        ripple_rate_hz = ripple_count / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        return {
            "ripples": ripple_count,
            "duration_ms": duration_ms,
            "ripple_rate_hz": ripple_rate_hz,
        }

    def select_action(self, explore: bool = True) -> tuple[int, float]:
        """Select action based on current striatum state.

        Uses striatum to select actions based on accumulated evidence.

        Args:
            explore: Whether to allow exploration (epsilon-greedy)

        Returns:
            (action, confidence): Selected action index and confidence [0, 1]

        Raises:
            ValueError: If striatum not found in brain
        """
        striatum = self.get_first_region_of_type(Striatum)
        if striatum is None:
            raise ValueError(
                "Striatum not found. Cannot select action. "
                "Brain must include 'striatum' region for RL tasks."
            )

        # Striatum has finalize_action method for action selection
        result = striatum.finalize_action(explore=explore)

        # Extract action from result dict
        action = result["selected_action"]

        # Compute confidence from probabilities or net votes
        probs = result.get("probs")
        if probs is not None:
            # Softmax case: use max probability as confidence
            confidence = float(probs.max().item())
        else:
            # Argmax case: use normalized net votes as confidence
            net_votes = result["net_votes"]
            if net_votes.sum() > 0:
                confidence = float(net_votes[action].item() / net_votes.sum().item())
            else:
                confidence = 1.0 / len(net_votes)  # Uniform if no votes

        # Store for deliver_reward
        self._last_action = action

        return action, confidence

    def deliver_reward(self, external_reward: Optional[float] = None) -> None:
        """Deliver reward signal via RewardEncoder region for VTA processing.

        **Spiking Dopamine Architecture**: Reward is encoded as population-coded
        spikes in RewardEncoder, which feeds into VTA to modulate DA neuron firing.
        VTA DA spikes are then broadcast to all regions with DA receptors
        (Striatum, PFC, Hippocampus, Cortex) where they are converted to
        concentration dynamics via NeuromodulatorReceptor.

        Args:
            external_reward: Task-based reward value (typically -1 to +1),
                           or None for pure intrinsic reward

        Raises:
            ValueError: If reward_encoder or striatum not found

        Note:
            Actual learning happens continuously in region forward() passes using
            the spiking dopamine signal. This method only delivers reward to
            RewardEncoder - the rest happens automatically through VTA and DA receptors.
        """
        striatum = self.get_first_region_of_type(Striatum)
        if striatum is None:
            raise ValueError(
                "Striatum not found. Cannot deliver reward. "
                "Brain must include 'striatum' region for RL tasks."
            )

        # Get RewardEncoder region
        reward_encoder = self.get_first_region_of_type(RewardEncoder)
        if reward_encoder is None:
            raise ValueError(
                "RewardEncoder not found. Cannot deliver reward. "
                "Brain must include 'reward_encoder' region for spiking DA system."
            )

        # Compute total reward (external + intrinsic if available)
        intrinsic_reward = self._compute_intrinsic_reward()
        if external_reward is None:
            total_reward = intrinsic_reward
        else:
            total_reward = external_reward + intrinsic_reward
            total_reward = max(-2.0, min(2.0, total_reward))

        # Deliver reward to RewardEncoder as population-coded spikes
        # RewardEncoder converts scalar → spike pattern, which VTA decodes
        reward_encoder.set_reward(total_reward)

        # Update exploration statistics based on reward
        # Striatum applies continuous learning automatically in forward() using spiking DA
        striatum.update_performance(total_reward)

    def _get_cortex_l4_activity(self) -> Optional[float]:
        """Helper to get cortex L4 activity for neuromodulator computations."""
        cortex = self.get_first_region_of_type(Cortex)
        if cortex is not None:
            return compute_firing_rate(cortex.l4_spikes)
        return None

    def _get_hippocampus_ca1_activity(self) -> Optional[float]:
        """Helper to get hippocampus CA1 activity for neuromodulator computations."""
        hippocampus = self.get_first_region_of_type(Hippocampus)
        if hippocampus is not None:
            return compute_firing_rate(hippocampus.ca1_spikes)
        return None

    def _compute_intrinsic_reward(self) -> float:
        """Compute intrinsic reward from the brain's internal objectives.

        This implements the free energy principle: the brain rewards itself
        for minimizing prediction error (surprise). Intrinsic reward is
        ALWAYS computed - it's the brain's continuous self-evaluation.

        Sources:
        1. **Cortex predictive coding**: Low prediction error → good model of the world → reward
        2. **Hippocampus pattern completion**: High similarity → successful recall → reward

        This is biologically plausible:
        - VTA dopamine neurons respond to internal prediction errors
        - Curiosity and "eureka" moments are intrinsically rewarding
        - The brain learns even without external feedback

        Returns:
            Intrinsic reward in range [-1, 1]
        """
        reward = 0.0
        n_sources = 0

        # =====================================================================
        # 1. CORTEX PREDICTIVE CODING (free energy minimization)
        # =====================================================================
        # Low prediction error = good model of the world = reward
        # L4 activity = prediction error
        # L4 = input - (L5+L6 prediction), so low L4 = good prediction = reward
        l4_activity = self._get_cortex_l4_activity()
        if l4_activity is not None:
            # L4 activity is prediction error (0 = perfect prediction, 1 = max error)
            # Map to reward: 0 → +1, 0.5 → 0, 1 → -1
            cortex_reward = 1.0 - 2.0 * l4_activity
            cortex_reward = max(-1.0, min(1.0, cortex_reward))
            reward += cortex_reward
            n_sources += 1

        # =====================================================================
        # 2. HIPPOCAMPUS PATTERN COMPLETION (memory recall quality)
        # =====================================================================
        # High pattern similarity = successful memory retrieval = reward
        # Biology: VTA observes CA1 output activity. Strong coherent firing = successful recall.
        # We infer similarity from CA1 spike rate (observable signal).
        ca1_activity = self._get_hippocampus_ca1_activity()
        if ca1_activity is not None:
            # CA1 firing rate as proxy for retrieval quality
            # High rate = strong recall, low rate = weak/no recall

            # Map CA1 activity [0, 1] to reward [-1, 1]
            # 0.5 activity = neutral (0 reward), >0.5 = positive, <0.5 = negative
            hippo_reward = 2.0 * ca1_activity - 1.0
            # Weight slightly less than cortex (memory is secondary to prediction)
            reward += 0.5 * hippo_reward
            n_sources += 1

        # =====================================================================
        # Average across sources
        # =====================================================================
        if n_sources > 0:
            reward = reward / n_sources
        else:
            # No signals → assume moderate intrinsic reward
            reward = 0.0

        return max(-1.0, min(1.0, reward))

    # =========================================================================
    # ADAPTIVE TIMESTEP AND OSCILLATOR COORDINATION
    # =========================================================================

    def set_timestep(self, new_dt_ms: float) -> None:
        """Change simulation timestep adaptively during training.

        Updates dt_ms and propagates temporal parameter updates to:
        - All regions (neurons, STP, learning strategies)
        - Oscillator manager (phase increments)
        - Pathway manager (delay buffers)

        Use cases:
        - Memory replay: 10x speedup (dt=10ms) during consolidation
        - Critical learning: Slow down to 0.1ms for precise timing
        - Energy efficiency: Larger dt when dynamics are stable

        Args:
            new_dt_ms: New timestep in milliseconds (must be positive)

        Raises:
            ValueError: If new_dt_ms <= 0
        """
        if new_dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {new_dt_ms}")

        # Update brain dt
        self.config.dt_ms = new_dt_ms

        # Update all regions
        for region in self.regions.values():
            region.update_temporal_parameters(new_dt_ms)

        # Update all connections/pathways
        for pathway in self.connections.values():
            pathway.update_temporal_parameters(new_dt_ms)

    def _broadcast_oscillator_phases(self) -> None:
        """Broadcast oscillator phases to all regions.

        Advances oscillators by dt_ms and updates all regions with current
        phases for all frequency bands (delta, theta, alpha, beta, gamma, ripple).

        This enables:
        - Delta slow-wave sleep consolidation
        - Theta-driven encoding/retrieval in hippocampus
        - Alpha attention gating
        - Beta motor control and working memory
        - Gamma feature binding in cortex
        - Ripple sharp-wave replay

        **Note on Emergent Oscillations**:

        **Theta (4-12 Hz)**: Disabled - emerges from medial septum pacemaker neurons
        (cholinergic and GABAergic) that phase-lock hippocampal OLM interneurons.
        OLM dendritic inhibition creates emergent encoding/retrieval separation.
        This is biologically accurate: hippocampal theta arises from septal drive,
        not a central oscillator.

        **Gamma (30-80 Hz)**: Disabled - two gamma frequencies naturally emerge from
        L6a-TRN-relay (~40Hz) and L6b-relay (~60Hz) feedback loops. This is biologically
        accurate: cortical gamma arises from corticothalamic interactions, not a
        central oscillator.
        """
        # Advance oscillators
        self.oscillators.advance(dt_ms=self.dt_ms)

        # Get all phases, signals, and coupled amplitudes
        phases = self.oscillators.get_phases()
        signals = self.oscillators.get_signals()
        coupled_amplitudes = self.oscillators.get_coupled_amplitudes()

        # Broadcast directly to all regions
        for region in self.regions.values():
            region.set_oscillator_phases(phases, signals, coupled_amplitudes)
