"""
Trial execution coordinator for EventDrivenBrain.

This module extracts trial execution logic (forward, select_action, deliver_reward)
from the EventDrivenBrain god object, following the existing manager pattern.

Author: Thalia Project
Date: December 2025
"""

from typing import Dict, Optional, Any, TYPE_CHECKING
import torch

from thalia.events import Event, EventType, SpikePayload, get_axonal_delay

if TYPE_CHECKING:
    from thalia.events import EventScheduler
    from .pathway_manager import PathwayManager
    from .neuromodulator_manager import NeuromodulatorManager
    from .oscillator import OscillatorManager


class TrialCoordinator:
    """Manages trial execution flow (forward passes, action selection, reward delivery).

    This coordinator handles the high-level trial APIs that were previously in
    EventDrivenBrain. It delegates to specialized managers for subsystem coordination:
    - PathwayManager: Inter-region connections
    - NeuromodulatorManager: VTA, LC, NB systems
    - OscillatorManager: Brain-wide rhythms

    Responsibilities:
    1. Forward passes - Encoding, maintenance, retrieval
    2. Action selection - Query striatum for actions
    3. Reward delivery - Compute RPE, broadcast dopamine, trigger learning
    4. State tracking - Last action, last outputs for learning

    Architecture:
    - Follows existing manager pattern (PathwayManager, NeuromodulatorManager)
    - Maintains backward compatibility (same external API)
    - Enables independent testing of trial execution logic
    """

    def __init__(
        self,
        regions: Dict[str, Any],
        pathways: "PathwayManager",
        neuromodulators: "NeuromodulatorManager",
        oscillators: "OscillatorManager",
        config: Any,
        spike_counts: Dict[str, int],
        vta: Any,
        brain_time: Any,  # Reference to brain's _current_time (mutable container)
        mental_simulation: Optional[Any] = None,
        dyna_planner: Optional[Any] = None,
    ):
        """Initialize trial coordinator.

        Args:
            regions: Dict of region name -> EventDrivenRegion adapter
            pathways: PathwayManager for inter-region connections
            neuromodulators: NeuromodulatorManager for VTA/LC/NB
            oscillators: OscillatorManager for brain-wide rhythms
            config: Brain configuration (SimpleNamespace from EventDrivenBrain)
            spike_counts: Shared spike count dict (updated in-place)
            vta: VTA neuromodulator for dopamine
            brain_time: Reference to brain's _current_time (shared state)
            mental_simulation: Optional mental simulation coordinator
            dyna_planner: Optional Dyna-style planning system
        """
        self.regions = regions
        self.pathways = pathways
        self.neuromodulators = neuromodulators
        self.oscillators = oscillators
        self.config = config
        self._spike_counts = spike_counts
        self.vta = vta
        self._brain_time = brain_time  # Share time with brain
        self.mental_simulation = mental_simulation
        self.dyna_planner = dyna_planner

        # State tracking for learning
        self._last_action: Optional[int] = None
        self._last_pfc_output: Optional[torch.Tensor] = None
        self._last_cortex_output: Optional[torch.Tensor] = None

        # For sequential execution
        self._events_processed: int = 0

    def forward(
        self,
        sensory_input: Optional[torch.Tensor],
        n_timesteps: int,
        scheduler: "EventScheduler",
        parallel_executor: Optional[Any],
        process_events_fn: Any,
        update_neuromodulators_fn: Any,
        get_cortex_input_fn: Any,
        criticality_monitor: Optional[Any],
    ) -> Dict[str, Any]:
        """Execute forward pass with event scheduling.

        This is the main trial execution method - handles encoding, maintenance,
        and retrieval through natural brain dynamics.

        Args:
            sensory_input: Input pattern [input_size], or None for maintenance
            n_timesteps: Number of timesteps to process
            scheduler: EventScheduler for sequential mode
            parallel_executor: Optional ParallelExecutor for parallel mode
            process_events_fn: Callback to process events (Brain._process_events_until)
            update_neuromodulators_fn: Callback to update neuromodulators (Brain._update_neuromodulators)
            get_cortex_input_fn: Callback to get cortex input (Brain._get_cortex_input)
            criticality_monitor: Optional criticality monitor

        Returns:
            Dict with region activities and metrics
        """
        # Validate input if provided
        if sensory_input is not None:
            assert sensory_input.shape[-1] == self.config.input_size, (
                f"TrialCoordinator.forward: sensory_input has shape {sensory_input.shape} "
                f"but input_size={self.config.input_size}. Check that input matches brain config."
            )

        # Choose execution path: parallel or sequential
        if parallel_executor is not None:
            # PARALLEL EXECUTION
            end_time = self._brain_time[0] + n_timesteps * self.config.dt_ms

            # Get effective input (zero tensor for consolidation)
            cortex_input = get_cortex_input_fn(sensory_input)

            # Inject sensory input
            parallel_executor.inject_sensory_input(
                cortex_input,
                target="cortex",
                time=self._brain_time[0],
            )

            # Run parallel simulation
            result = parallel_executor.run_until(end_time)

            # Update shared state
            self._brain_time[0] = end_time
            self._spike_counts.update(result["spike_counts"])
            self._events_processed = result["events_processed"]

            results = {
                "cortex_activity": torch.zeros(self.config.cortex_size),
                "hippocampus_activity": torch.zeros(self.config.hippocampus_size),
                "pfc_activity": torch.zeros(self.config.pfc_size),
                "spike_counts": self._spike_counts.copy(),
                "events_processed": self._events_processed,
                "final_time": self._brain_time[0],
            }
        else:
            for t in range(n_timesteps):
                step_time = self._brain_time[0] + t * self.config.dt_ms

                # Advance oscillators (once per timestep)
                self.oscillators.advance(self.config.dt_ms)

                # =====================================================================
                # PHASE 3: GOAL MANAGEMENT
                # =====================================================================
                # Before processing input, check if we have active hierarchical goals
                # and whether current goal should be updated
                pfc_region = self.regions.get("pfc")
                if pfc_region and hasattr(pfc_region.impl, 'goal_manager') and pfc_region.impl.goal_manager is not None:
                    goal_mgr = pfc_region.impl.goal_manager

                    # Update current goal's progress if we have one
                    current_goal = goal_mgr.get_current_goal()
                    if current_goal is not None and self._last_pfc_output is not None:
                        goal_mgr.update_progress(current_goal, self._last_pfc_output)

                        # If current goal completed, pop it and activate parent
                        if current_goal.status.value == "completed":
                            goal_mgr.pop_goal()
                            # Try to push parent goal if exists
                            if current_goal.parent_goal is not None:
                                goal_mgr.push_goal(current_goal.parent_goal)

                    # Advance time counter for deadline tracking
                    goal_mgr.advance_time()

                # Schedule sensory input through thalamus (or zero input for consolidation)
                # Sensory → Thalamus → Cortex pathway for biologically accurate relay
                sensory_input_tensor = get_cortex_input_fn(sensory_input)
                delay = get_axonal_delay("sensory", "thalamus")
                event = Event(
                    time=step_time + delay,
                    event_type=EventType.SENSORY,
                    source="sensory_input",
                    target="thalamus",
                    payload=SpikePayload(spikes=sensory_input_tensor.detach().clone()),
                )
                scheduler.schedule(event)

                # Update all neuromodulators continuously
                update_neuromodulators_fn()

            # Update time
            end_time = self._brain_time[0] + n_timesteps * self.config.dt_ms

            # Process all events up to end_time
            process_events_fn(end_time)

            self._brain_time[0] = end_time

            results = {
                "spike_counts": self._spike_counts.copy(),
                "events_processed": self._events_processed,
                "final_time": self._brain_time[0],
            }

        # Capture PFC output for decoder (language model, mental simulation)
        pfc_region = self.regions.get("pfc")
        if pfc_region and hasattr(pfc_region.impl, 'state') and pfc_region.impl.state is not None:
            if pfc_region.impl.state.spikes is not None:
                self._last_pfc_output = pfc_region.impl.state.spikes.float().squeeze(0).clone()

        return results

    def select_action(self, explore: bool = True, use_planning: bool = True) -> tuple[int, float]:
        """Select action based on current striatum state.

        Uses the striatum's finalize_action method which handles:
        - Accumulated NET votes (D1-D2)
        - UCB exploration bonus
        - Softmax selection

        If use_planning=True and model-based planning is enabled:
        - Uses MentalSimulationCoordinator for tree search
        - Returns best action from simulated rollouts
        - Falls back to striatum if planning disabled

        Phase 3: If hierarchical goals enabled:
        - Checks current goal for policy override
        - Uses goal's policy if available (options framework)
        - Falls back to standard action selection

        Args:
            explore: Whether to allow exploration
            use_planning: Whether to use mental simulation

        Returns:
            (action, confidence): Selected action index and confidence [0, 1]
        """
        # =====================================================================
        # PHASE 3: GOAL-DRIVEN ACTION SELECTION
        # =====================================================================
        # Check if we have an active goal with a policy (option)
        pfc = self.regions.get("pfc")
        if pfc and hasattr(pfc.impl, 'goal_manager') and pfc.impl.goal_manager is not None:
            goal_mgr = pfc.impl.goal_manager
            current_goal = goal_mgr.get_current_goal()

            # If goal has a policy (it's an option), use it
            if current_goal is not None and current_goal.policy is not None:
                if self._last_pfc_output is not None:
                    # Execute option policy
                    action = current_goal.policy(self._last_pfc_output)
                    self._last_action = action
                    return action, 1.0  # High confidence - using learned option

        # Phase 2: Mental simulation planning
        if use_planning and self.mental_simulation is not None:
            # Get current state from PFC working memory
            current_state = self._last_pfc_output
            if current_state is None:
                # No state yet, fall back to model-free
                pass
            else:
                # Get goal context from PFC working memory
                goal_context = self._last_pfc_output

                # Plan best action using mental simulation
                available_actions = list(range(self.config.n_actions))
                best_action, best_rollout = self.mental_simulation.plan_best_action(
                    current_state=current_state,
                    available_actions=available_actions,
                    goal_context=goal_context
                )

                # Confidence from rollout (higher value = higher confidence)
                confidence = float(torch.sigmoid(torch.tensor(best_rollout.cumulative_value)).item())

                self._last_action = best_action
                return best_action, confidence

        # Model-free action selection (standard)
        striatum = self.regions["striatum"]
        result = striatum.impl.finalize_action(explore=explore)

        action = result.get("selected_action", 0)
        probs = result.get("probs", None)

        if probs is not None:
            confidence = float(probs[action].item())
        else:
            confidence = 1.0

        self._last_action = action

        return action, confidence

    def deliver_reward(
        self,
        external_reward: Optional[float],
        compute_intrinsic_reward_fn: Any,
    ) -> None:
        """Deliver external reward signal for learning.

        Coordinates reward processing:
        1. Combines external reward with intrinsic reward
        2. Queries striatum for expected value
        3. Computes RPE via VTA
        4. Broadcasts dopamine to all regions
        5. Triggers striatum learning

        Args:
            external_reward: Task-based reward value (-1 to +1), or None for pure intrinsic
            compute_intrinsic_reward_fn: Callback to compute intrinsic reward (Brain._compute_intrinsic_reward)
        """
        # =====================================================================
        # STEP 1: Compute total reward (external + intrinsic)
        # =====================================================================
        intrinsic_reward = compute_intrinsic_reward_fn()

        if external_reward is None:
            # No external feedback - use pure intrinsic
            total_reward = intrinsic_reward
        else:
            # External feedback provided - ADD to intrinsic
            total_reward = external_reward + intrinsic_reward
            # Clamp to reasonable range (dopamine saturation)
            total_reward = max(-2.0, min(2.0, total_reward))

        # =====================================================================
        # STEP 2: Get expected value from striatum
        # =====================================================================
        striatum = self.regions["striatum"]
        expected = striatum.impl.get_expected_value(self._last_action)

        # =====================================================================
        # STEP 3: Compute RPE and deliver to VTA
        # =====================================================================
        self.vta.deliver_reward(
            external_reward=total_reward,
            expected_value=expected
        )

        # =====================================================================
        # STEP 4: Get dopamine signal from VTA and broadcast to ALL regions
        # =====================================================================
        dopamine = self.vta.get_global_dopamine()

        for region_adapter in self.regions.values():
            region_adapter.impl.set_dopamine(dopamine)

        # =====================================================================
        # STEP 5: Trigger striatum learning (D1/D2 plasticity)
        # =====================================================================
        if self._last_action is not None:
            striatum.impl.deliver_reward(total_reward)

            # Update value estimate
            striatum.impl.update_value_estimate(self._last_action, total_reward)

            # Store experience for replay (if Dyna planning enabled)
            if self.dyna_planner is not None:
                current_state = self._last_pfc_output
                next_state = self._last_pfc_output  # Same state after action in working memory
                goal_context = self._last_pfc_output

                if current_state is not None:
                    self.dyna_planner.store_experience(
                        state=current_state,
                        action=self._last_action,
                        reward=total_reward,
                        next_state=next_state,
                        done=False,
                        goal_context=goal_context,
                    )

        # =====================================================================
        # STEP 6: PHASE 3 - Adaptive temporal discounting (learn_from_choice)
        # =====================================================================
        # If hyperbolic discounting is enabled, adapt k parameter based on outcomes.
        # This enables meta-learning: the brain learns its own impulsivity rate.
        #
        # Biology: PFC damage → increased temporal discounting (more impulsive).
        # Here we model the adaptive process: good delayed choices → more patience.
        pfc = self.regions["pfc"]
        if (hasattr(pfc.impl, 'discounter') and
            pfc.impl.discounter is not None and
            self._last_action is not None):

            # Determine if this was a delayed reward choice
            # For now, use total_reward > 0 as proxy for "good outcome"
            # In future, could track explicit delayed vs immediate choices
            outcome_quality = (total_reward + 1.0) / 2.0  # Map [-1, 1] → [0, 1]
            outcome_quality = max(0.0, min(1.0, outcome_quality))

            # Simple heuristic: If reward is positive, assume delayed choice was made
            # If reward is negative, assume immediate choice or bad delayed choice
            chose_delayed = total_reward > 0.0

            # For learning, we need to know what the choice was between
            # Default values (can be enhanced with actual choice tracking)
            immediate_value = 0.5  # Baseline immediate reward
            delayed_value = 1.0    # Potential delayed reward
            delay = 10             # Typical delay in timesteps

            pfc.impl.discounter.learn_from_choice(
                chose_delayed=chose_delayed,
                immediate_value=immediate_value,
                delayed_value=delayed_value,
                delay=delay,
                outcome_quality=outcome_quality,
            )

        # =====================================================================
        # STEP 7: PHASE 3 - OPTION LEARNING
        # =====================================================================
        # Track option success and cache successful policies
        # This implements the options framework for hierarchical RL
        if hasattr(pfc.impl, 'goal_manager') and pfc.impl.goal_manager is not None:
            goal_mgr = pfc.impl.goal_manager
            current_goal = goal_mgr.get_current_goal()

            # If we have an active goal that was just completed/failed
            if current_goal is not None and current_goal.policy is not None:
                # Determine success from reward
                success = total_reward > 0.0

                # Record option attempt for this goal
                goal_mgr.record_option_attempt(current_goal.name, success)

                # Get current success rate
                success_rate = goal_mgr.get_option_success_rate(current_goal.name)

                # If this policy is consistently successful, cache it as a reusable option
                # This is Hebbian-like: "neurons that fire together wire together"
                # Here: "goals achieved together get cached together"
                if (success_rate > goal_mgr.config.option_discovery_threshold and
                    current_goal.name not in goal_mgr.options):
                    goal_mgr.cache_option(
                        goal=current_goal,
                        policy=current_goal.policy,
                        success_rate=success_rate
                    )

    def deliver_reward_with_counterfactual(
        self,
        external_reward: Optional[float],
        counterfactual_action: int,
        compute_intrinsic_reward_fn: Any,
    ) -> None:
        """Deliver reward with counterfactual learning signal.

        Extended reward delivery that includes counterfactual comparison:
        - Learns from chosen action (positive/negative RPE)
        - Learns from unchosen action (inverted RPE for contrast)

        This implements counterfactual learning for faster policy improvement.

        Args:
            external_reward: Task-based reward (-1 to +1)
            counterfactual_action: Action that wasn't chosen
            compute_intrinsic_reward_fn: Callback to compute intrinsic reward
        """
        # First deliver standard reward
        self.deliver_reward(external_reward, compute_intrinsic_reward_fn)

        # Then deliver counterfactual signal
        intrinsic_reward = compute_intrinsic_reward_fn()
        total_reward = external_reward + intrinsic_reward if external_reward is not None else intrinsic_reward
        total_reward = max(-2.0, min(2.0, total_reward))

        # Counterfactual gets inverted reward (if actual was good, counterfactual was bad)
        counterfactual_reward = -total_reward * 0.5  # Weaker signal (50% of actual)

        striatum = self.regions["striatum"]
        striatum.impl.update_value_estimate(counterfactual_action, counterfactual_reward)

    def get_last_action(self) -> Optional[int]:
        """Get the last selected action."""
        return self._last_action

    def get_last_pfc_output(self) -> Optional[torch.Tensor]:
        """Get the last PFC output (for state tracking)."""
        return self._last_pfc_output

    def get_last_cortex_output(self) -> Optional[torch.Tensor]:
        """Get the last cortex output (for state tracking)."""
        return self._last_cortex_output

    def reset_state(self) -> None:
        """Reset trial state (action tracking, outputs)."""
        self._last_action = None
        self._last_pfc_output = None
        self._last_cortex_output = None
