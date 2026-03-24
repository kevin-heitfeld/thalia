"""
Prefrontal Cortex — Agranular Laminar Cortex with Working-Memory Specialisation.

The PFC is *not* a special-purpose black box.  It is a six-layer cortical column
whose parameters are tuned for executive control and working memory:

* **Long tau_mem in L2/3 (200 ms)** — sustained spiking activity = working memory.
* **Dense strong L2/3 recurrence (connectivity=1.0, weight 5×)** — WM attractors.
* **Dense mesocortical DA to L2/3 (30% vs. 7.5% standard)** — D1-gated updating.
* **D1/D2 receptor split on L2/3** — bidirectional dopamine gating.

Working memory is *not* an explicit buffer.  It emerges from attractor dynamics
in the L2/3 recurrent circuit — the same mechanism as biological PFC.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import CorticalColumnConfig
from thalia.brain.synapses import NMReceptorType
from thalia.typing import (
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)

from .cortical_column import CorticalColumn
from .neural_region import ReceptorSpec
from .population_names import CortexPopulation
from .region_registry import register_region


@register_region(
    "prefrontal_cortex",
    aliases=["pfc", "prefrontal"],
    description=(
        "Agranular laminar cortex with WM-attractor L2/3, DA-gated executive control, "
        "and emergent goal representations"
    ),
)
class PrefrontalCortex(CorticalColumn):
    """Prefrontal cortex: CorticalColumn with WM-attractor L2/3 and goal emergence."""

    # Mesocortical DA gates WM updates (D1 receptor gating).
    # NE from LC modulates PFC signal-to-noise.
    # ACh from NB drives attentional selection and WM encoding mode.
    # 5-HT from DRN (5-HT1A/2A) encodes patience / temporal discounting: reduces
    #   impulsive DA-driven plasticity (Crockett et al. 2009; Clarke et al. 2007).
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.DA_MESOCORTICAL,
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.ACH,
        NeuromodulatorChannel.SHT,
    ]

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def __init__(
        self,
        config: CorticalColumnConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialise PFC as a CorticalColumn with WM specialisations."""
        super().__init__(config, population_sizes, region_name, device=device)

        # =====================================================================
        # EMERGENT GOAL SYSTEM
        # =====================================================================
        # Goals arise from sustained L2/3 spike patterns — no symbolic objects.
        # The abstract/concrete split mirrors the rostral–caudal PFC hierarchy
        # (Badre & D'Esposito 2009): rostral 30% = abstract / slow goals,
        # caudal 70% = concrete / fast sub-goals.
        self.n_abstract = int(self.l23_pyr_size * 0.3)
        self.n_concrete = self.l23_pyr_size - self.n_abstract

        # Split PFC neurons into abstract/concrete populations
        # Abstract: long time constants (tau ~500ms), slow update
        # Concrete: short time constants (tau ~100ms), fast update
        self.abstract_neurons: torch.Tensor
        self.concrete_neurons: torch.Tensor
        self.register_buffer("abstract_neurons", torch.arange(self.n_abstract, device=device))
        self.register_buffer("concrete_neurons", torch.arange(self.n_abstract, self.n_abstract + self.n_concrete, device=device))

        # Goal transition matrix: learned associations between WM patterns
        # "When abstract pattern A is active, which concrete pattern B follows?"
        # This replaces explicit goal decomposition
        self.transition_weights: torch.Tensor
        self.register_buffer("transition_weights", torch.zeros(self.n_concrete, self.n_abstract, device=device))  # [concrete, abstract]

        # Goal value associations: OFC-like value mapping
        # Maps WM patterns to expected value (learned from experience).
        # Shape: [l23_pyr_size] — per-neuron scalar value (not a weight matrix).
        self.value_weights: torch.Tensor
        self.register_buffer("value_weights", torch.rand(self.l23_pyr_size, device=device) * 0.2)  # [l23_pyr_size]

        # Synaptic tags for goal patterns (Frey-Morris)
        # Recently-activated goals get tagged for consolidation
        self.goal_tags: torch.Tensor
        self.register_buffer("goal_tags", torch.zeros(self.l23_pyr_size, device=device))
        self.tag_decay = 0.95  # Same as hippocampal tags

        # =====================================================================
        # SEROTONIN RECEPTOR (5-HT1A and 5-HT2A on L2/3 pyramidals — from DRN)
        # =====================================================================
        # Biology: DRN projects dense 5-HT to PFC L2/3 pyramidals.
        #   - 5-HT1A (inhibitory, Gi): reduces excitability, lowers impulsive
        #     plasticity (patience / temporal discounting signal).
        #   - 5-HT2A (excitatory, Gq): increases attention gain; net effect
        #     at physiological concentrations is patience-dominance.
        # Implementation: 5-HT modulates learning rate via the patience gate,
        # attenuating DA-driven goal consolidation under high serotonin.
        # Kinetics: tau_rise ~10 ms, tau_decay ~200 ms (SERT reuptake).
        # 5-HT1A (Gi → GIRK): patience / temporal discounting gate on L2/3
        self._init_receptors([
            ReceptorSpec(NMReceptorType.SHT_1A, NeuromodulatorChannel.SHT, self.l23_pyr_size, "_sht_concentration_l23"),
        ], device)

    # =========================================================================
    # EMERGENT GOAL SYSTEM
    # =========================================================================

    def update_goal_tags(
        self,
        wm_pattern: torch.Tensor,  # [l23_pyr_size] - current WM activity
    ) -> None:
        """Tag currently active goal patterns.

        Similar to hippocampal synaptic tagging:
        - Active patterns get tagged
        - Tags decay over time
        - Dopamine consolidates tagged patterns

        Args:
            wm_pattern: Current working memory activity [l23_pyr_size]
        """
        # Decay existing tags
        self.goal_tags *= self.tag_decay

        # Tag current pattern (threshold to avoid noise)
        active_mask = wm_pattern > 0.3
        self.goal_tags[active_mask] = torch.maximum(self.goal_tags[active_mask], wm_pattern[active_mask])

    def learn_transition(
        self,
        abstract_pattern: torch.Tensor,  # [n_abstract] - parent goal
        concrete_pattern: torch.Tensor,  # [n_concrete] - subgoal that followed
        learning_rate: float = 0.01,
    ) -> None:
        """Learn goal hierarchies via Hebbian association.

        When abstract goal A leads to concrete subgoal B:
        - Strengthen transition A→B
        - This is how "decomposition" is learned, not programmed

        Replaces: Explicit Goal.add_subgoal()

        Args:
            abstract_pattern: Parent goal pattern [n_abstract]
            concrete_pattern: Subgoal pattern that followed [n_concrete]
            learning_rate: Learning rate for Hebbian update (default 0.01)
        """
        # Outer product: strengthen associations between co-active patterns
        # transition[concrete, abstract] += lr * concrete × abstract
        dW = learning_rate * torch.outer(concrete_pattern, abstract_pattern)
        self.transition_weights += dW

        # Normalize to prevent runaway weights
        self.transition_weights.clamp_(0.0, 1.0)

    def consolidate_valuable_goals(
        self,
        dopamine: float,
        learning_rate: float = 0.01,
    ) -> None:
        """Strengthen value associations for tagged goals.

        Similar to hippocampal consolidation:
        - Tagged patterns (recent goals) get consolidated with dopamine
        - High dopamine → high value goals become more prominent

        Args:
            dopamine: Current dopamine level (gates consolidation)
            learning_rate: Learning rate for value update (default 0.01)
        """
        if dopamine < 0.01:
            return  # No consolidation without dopamine

        # Strengthen value weights for tagged goals
        # High tag + high DA → increase value association
        value_update = learning_rate * dopamine * self.goal_tags
        self.value_weights += value_update

        # Keep values bounded
        self.value_weights.clamp_(-1.0, 1.0)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Full laminar cortical pass then emergent goal learning on L2/3 spikes.

        Args:
            synaptic_inputs:       Point-to-point conductance inputs per SynapseId.
            neuromodulator_inputs: Broadcast neuromodulatory spike vectors.

        Returns:
            RegionOutput mapping CortexPopulation → bool spike tensors for all
            five pyramidal layers and their inhibitory sub-populations.
        """
        # 5-HT receptor update
        self._update_receptors(neuromodulator_inputs)

        # Full layered cortical processing (layers, DA/NE/ACh, STDP, homeostasis).
        region_outputs = super()._step(synaptic_inputs, neuromodulator_inputs)

        if not GlobalConfig.LEARNING_DISABLED:
            # After super()._step(), _da_concentration_l23 holds the current-step
            # DA level for L2/3.  Use it to drive goal consolidation.
            l23_spikes: torch.Tensor = region_outputs[CortexPopulation.L23_PYR]
            l23_float = l23_spikes.float()
            da_level = self._da_concentration_l23.mean().item()

            # Tag recently active L2/3 patterns (synaptic tagging for consolidation)
            self.update_goal_tags(l23_float)

            # Learn abstract→concrete goal transitions via Hebbian association
            abstract_pat = l23_float[self.abstract_neurons]
            concrete_pat = l23_float[self.concrete_neurons]
            if abstract_pat.sum() > 0.1 and concrete_pat.sum() > 0.1:
                self.learn_transition(
                    abstract_pat,
                    concrete_pat,
                    learning_rate=self.config.learning_rate,
                )

            # Dopamine-gated consolidation: reward strengthens goal patterns
            self.consolidate_valuable_goals(
                dopamine=da_level,
                learning_rate=self.config.learning_rate,
            )

        return region_outputs

    # =========================================================================
    # LEARNING UTILITIES
    # =========================================================================

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        """Return DA-modulated learning kwargs for feedforward synapses.

        :attr:`_da_concentration_l23` is set by :meth:`CorticalColumn.forward`
        *before* ``_post_forward`` calls the learning strategies, so this always
        returns the current timestep's DA level — no off-by-one.

        Returns:
            ``learning_rate``: base LR scaled by mesocortical DA (excitatory)
                and attenuated by 5-HT (patience / temporal-discounting gate).
            ``neuromodulator``: raw DA level (used by three-factor STDP rules).
        """
        da_level = self._da_concentration_l23.mean().item()
        sht_level = self._sht_concentration_l23.mean().item()
        # DA excites plasticity; 5-HT passed separately for asymmetric LTP/LTD gating.
        effective_lr = self.config.learning_rate * (1.0 + da_level)
        return {
            "learning_rate": effective_lr,
            "neuromodulator": da_level,
            "dopamine": da_level,
            "serotonin": sht_level,
        }
