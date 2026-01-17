"""Mixin to consolidate common state restoration logic across regions.

This mixin provides helper methods for loading common state components (neuron state,
conductances, neuromodulators, STP, etc.) to eliminate ~200-300 lines of duplication
across 15-18 region implementations.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


class StateLoadingMixin:
    """Mixin providing common state restoration helper methods.

    Consolidates duplicated state loading logic across regions. Regions inheriting
    this mixin can:
    1. Use helper methods for common state (neurons, conductances, traces, etc.)
    2. Override _load_custom_state() for region-specific logic
    3. Call super().load_state() to get all common restoration automatically

    Usage Example:
        class MyRegion(NeuralRegion, StateLoadingMixin):
            def load_state(self, state: MyRegionState) -> None:
                # Common restoration (membrane, conductances, neuromodulators, STP)
                super().load_state(state)

                # Custom region-specific restoration
                self._load_custom_state(state)

            def _load_custom_state(self, state: MyRegionState) -> None:
                # Region-specific state restoration
                if state.working_memory is not None:
                    self.working_memory.data = state.working_memory.to(self.device)
    """

    def _restore_neuron_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore neuron membrane potentials and refractory state.

        Args:
            state_dict: Dictionary containing state tensors

        Restores:
            - neurons.membrane (v_mem)
            - neurons.refractory_time
        """
        if not hasattr(self, "neurons"):
            return

        # Membrane potential (try both "membrane" and "v_mem" keys)
        v_mem = state_dict.get("membrane")
        if v_mem is None:
            v_mem = state_dict.get("v_mem")

        if v_mem is not None:
            self.neurons.membrane.data = v_mem.to(self.device)

        # Refractory time
        refractory = state_dict.get("refractory_time")
        if refractory is not None and hasattr(self.neurons, "refractory_time"):
            self.neurons.refractory_time.data = refractory.to(self.device)

    def _restore_conductances(self, state_dict: Dict[str, Any]) -> None:
        """Restore synaptic conductances.

        Args:
            state_dict: Dictionary containing state tensors

        Restores:
            - neurons.g_E (excitatory conductance)
            - neurons.g_I (inhibitory conductance)
            - neurons.g_adaptation (adaptation conductance)
        """
        if not hasattr(self, "neurons"):
            return

        # Excitatory conductance (try both "g_exc" and "g_E" keys)
        g_exc = state_dict.get("g_exc")
        if g_exc is None:
            g_exc = state_dict.get("g_E")

        if g_exc is not None and hasattr(self.neurons, "g_E"):
            self.neurons.g_E.data = g_exc.to(self.device)

        # Inhibitory conductance (try both "g_inh" and "g_I" keys)
        g_inh = state_dict.get("g_inh")
        if g_inh is None:
            g_inh = state_dict.get("g_I")

        if g_inh is not None and hasattr(self.neurons, "g_I"):
            self.neurons.g_I.data = g_inh.to(self.device)

        # Adaptation conductance (check both g_adaptation and g_adapt)
        g_adapt = state_dict.get("g_adaptation")
        if g_adapt is not None:
            if hasattr(self.neurons, "g_adapt"):
                self.neurons.g_adapt.data = g_adapt.to(self.device)
            elif hasattr(self.neurons, "g_adaptation"):
                self.neurons.g_adaptation.data = g_adapt.to(self.device)

    def _restore_learning_traces(self, state_dict: Dict[str, Any]) -> None:
        """Restore learning-related traces (eligibility, BCM thresholds, etc.).

        Args:
            state_dict: Dictionary containing state tensors

        Restores:
            - eligibility_trace (three-factor learning)
            - bcm_threshold (BCM learning)
            - stdp_trace_pre (STDP pre-synaptic trace)
            - stdp_trace_post (STDP post-synaptic trace)
        """
        # Eligibility trace (three-factor learning)
        eligibility = state_dict.get("eligibility_trace")
        if eligibility is not None and hasattr(self, "eligibility_trace"):
            self.eligibility_trace.data = eligibility.to(self.device)

        # BCM threshold
        bcm_threshold = state_dict.get("bcm_threshold")
        if bcm_threshold is not None and hasattr(self, "bcm_threshold"):
            self.bcm_threshold.data = bcm_threshold.to(self.device)

        # STDP traces
        stdp_pre = state_dict.get("stdp_trace_pre")
        if stdp_pre is not None and hasattr(self, "stdp_trace_pre"):
            self.stdp_trace_pre.data = stdp_pre.to(self.device)

        stdp_post = state_dict.get("stdp_trace_post")
        if stdp_post is not None and hasattr(self, "stdp_trace_post"):
            self.stdp_trace_post.data = stdp_post.to(self.device)

    def _restore_neuromodulators(self, state_dict: Dict[str, Any]) -> None:
        """Restore neuromodulator levels.

        Args:
            state_dict: Dictionary containing state tensors

        Restores:
            - dopamine (DA)
            - acetylcholine (ACh)
            - norepinephrine (NE)

        Note: Handles both self.{modulator} and self.state.{modulator} patterns.
        """
        # Try self.{modulator} pattern first (direct attributes)
        # Then try self.state.{modulator} pattern (state object)

        # Dopamine
        dopamine = state_dict.get("dopamine")
        if dopamine is not None:
            if hasattr(self, "dopamine") and isinstance(self.dopamine, torch.Tensor):
                self.dopamine.data = dopamine.to(self.device)
            elif hasattr(self, "state") and hasattr(self.state, "dopamine"):
                self.state.dopamine = dopamine

        # Acetylcholine
        acetylcholine = state_dict.get("acetylcholine")
        if acetylcholine is not None:
            if hasattr(self, "acetylcholine") and isinstance(self.acetylcholine, torch.Tensor):
                self.acetylcholine.data = acetylcholine.to(self.device)
            elif hasattr(self, "state") and hasattr(self.state, "acetylcholine"):
                self.state.acetylcholine = acetylcholine

        # Norepinephrine
        norepinephrine = state_dict.get("norepinephrine")
        if norepinephrine is not None:
            if hasattr(self, "norepinephrine") and isinstance(self.norepinephrine, torch.Tensor):
                self.norepinephrine.data = norepinephrine.to(self.device)
            elif hasattr(self, "state") and hasattr(self.state, "norepinephrine"):
                self.state.norepinephrine = norepinephrine

    def _restore_stp_state(self, state_dict: Dict[str, Any], stp_attr_name: str = "stp") -> None:
        """Restore short-term plasticity (STP) state.

        Args:
            state_dict: Dictionary containing state tensors
            stp_attr_name: Name of the STP attribute (default: "stp")

        Restores:
            - stp.u (utilization variable)
            - stp.x (available resources)

        STP state is stored as nested dict:
            {"u": torch.Tensor, "x": torch.Tensor}
        """
        stp_state_key = f"{stp_attr_name}_state"
        stp_state = state_dict.get(stp_state_key)

        if stp_state is not None and hasattr(self, stp_attr_name):
            stp_module = getattr(self, stp_attr_name)

            # Restore utilization
            if "u" in stp_state and hasattr(stp_module, "u"):
                stp_module.u.data = stp_state["u"].to(self.device)

            # Restore resources
            if "x" in stp_state and hasattr(stp_module, "x"):
                stp_module.x.data = stp_state["x"].to(self.device)

    def _restore_multi_stp_state(self, state_dict: Dict[str, Any], stp_names: list[str]) -> None:
        """Restore multiple STP modules (common in multi-pathway regions).

        Args:
            state_dict: Dictionary containing state tensors
            stp_names: List of STP attribute names (e.g., ["stp_ff", "stp_fb"])

        Example:
            # Thalamus has two STP modules
            self._restore_multi_stp_state(state_dict, [
                "stp_sensory_relay",
                "stp_l6_feedback"
            ])
        """
        for stp_name in stp_names:
            self._restore_stp_state(state_dict, stp_attr_name=stp_name)

    def load_state(self, state: Any) -> None:
        """Restore region state from checkpoint.

        This base implementation restores all common state components. Regions
        should override this and call super().load_state(state) to get common
        restoration, then add custom logic via _load_custom_state().

        Args:
            state: Region-specific state object (must have .to_dict() method)

        Raises:
            AttributeError: If state object doesn't have to_dict() method
        """
        if not hasattr(state, "to_dict"):
            raise AttributeError(
                f"State object {type(state).__name__} must implement to_dict() method"
            )

        state_dict = state.to_dict()

        # Restore common components
        self._restore_neuron_state(state_dict)
        self._restore_conductances(state_dict)
        self._restore_learning_traces(state_dict)
        self._restore_neuromodulators(state_dict)

        # STP restoration is often region-specific (multiple modules)
        # so we don't automatically call it here - regions should call
        # _restore_stp_state() or _restore_multi_stp_state() explicitly

        # Custom region-specific logic
        self._load_custom_state(state)

    def _load_custom_state(self, state: Any) -> None:
        """Override this method for region-specific state restoration.

        Args:
            state: Region-specific state object

        Example:
            def _load_custom_state(self, state: PrefrontalState) -> None:
                # Working memory
                if state.working_memory is not None:
                    self.working_memory.data = state.working_memory.to(self.device)

                # Update gate
                if state.update_gate is not None:
                    self.update_gate.data = state.update_gate.to(self.device)
        """
        # Default: no custom state
