"""
Cortex - Unsupervised Hebbian Learning

The cerebral cortex learns through unsupervised Hebbian plasticity, discovering
statistical structure in sensory inputs without explicit teaching signals.

Key Features:
=============
1. HEBBIAN STDP: Spike-timing dependent plasticity
   - Pre before post → LTP (strengthen connection)
   - Post before pre → LTD (weaken connection)
   - No external teaching signal required

2. BCM HOMEOSTASIS: Sliding threshold prevents runaway learning
   - High activity → raise threshold → favor LTD
   - Low activity → lower threshold → favor LTP

3. LATERAL INHIBITION: Competition between neurons
   - Winner-take-all dynamics
   - Creates sparse, selective representations

4. NEUROMODULATION: ACh gates plasticity magnitude (not direction)

Biological Basis:
=================
- Visual cortex V1 learns oriented edge detectors
- Auditory cortex learns frequency tuning
- Association cortex learns correlational structure

When to Use:
============
- Feature extraction from sensory data
- Unsupervised clustering/categorization
- When you DON'T have explicit target labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    LearningRule,
    NeuromodulatorSystem,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.learning import hebbian_update, update_bcm_threshold


@dataclass
class CortexConfig(RegionConfig):
    """Configuration specific to cortical regions."""

    hebbian_lr: float = 0.01
    stdp_tau_plus_ms: float = 20.0
    stdp_tau_minus_ms: float = 20.0
    bcm_tau_ms: float = 1000.0
    bcm_target_rate_hz: float = 10.0
    lateral_inhibition: bool = True
    inhibition_strength: float = 0.3
    heterosynaptic_ratio: float = 0.3
    soft_bounds: bool = True


class AcetylcholineSystem(NeuromodulatorSystem):
    """Acetylcholine neuromodulatory system for attention gating."""

    def __init__(self, tau_ms: float = 100.0, device: str = "cpu"):
        super().__init__(tau_ms, device)
        self.baseline = 0.5
        self.level = self.baseline

    def compute(
        self,
        novelty: float = 0.0,
        attention: float = 0.0,
        uncertainty: float = 0.0,
        **kwargs: Any,
    ) -> float:
        boost = 0.3 * novelty + 0.4 * attention + 0.3 * uncertainty
        self.level = min(1.0, self.baseline + boost)
        return self.level

    def get_learning_modulation(self) -> float:
        return 0.5 + self.level * 1.5


class Cortex(BrainRegion):
    """Cerebral cortex with unsupervised Hebbian learning."""

    def __init__(self, config: RegionConfig):
        if not isinstance(config, CortexConfig):
            config = CortexConfig(
                n_input=config.n_input,
                n_output=config.n_output,
                neuron_type=config.neuron_type,
                learning_rate=config.learning_rate,
                w_max=config.w_max,
                w_min=config.w_min,
                target_firing_rate_hz=config.target_firing_rate_hz,
                dt_ms=config.dt_ms,
                device=config.device,
            )

        self.cortex_config: CortexConfig = config  # type: ignore
        super().__init__(config)

        self.bcm_threshold = torch.ones(1, config.n_output, device=self.device) * 0.5
        self.ach = AcetylcholineSystem(device=config.device)
        self.input_trace = torch.zeros(config.n_input, device=self.device)
        self.output_trace = torch.zeros(config.n_output, device=self.device)
        self.recent_spikes = torch.zeros(config.n_output, device=self.device)

    def _get_learning_rule(self) -> LearningRule:
        return LearningRule.HEBBIAN

    def _initialize_weights(self) -> torch.Tensor:
        weights = torch.exp(torch.randn(self.config.n_output, self.config.n_input) * 0.5 - 2.0)
        weights = weights * self.config.w_max * 0.3
        return weights.clamp(self.config.w_min, self.config.w_max).to(self.device)

    def _create_neurons(self) -> ConductanceLIF:
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=5.0, tau_I=10.0,
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def forward(
        self,
        input_spikes: torch.Tensor,
        recurrent_input: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        if self.neurons.membrane is None:
            self.neurons.reset_state(input_spikes.shape[0])

        decay = 1.0 - self.config.dt_ms / self.cortex_config.stdp_tau_plus_ms
        self.input_trace = self.input_trace * decay + input_spikes.squeeze()

        g_exc = torch.matmul(input_spikes, self.weights.T)
        if recurrent_input is not None:
            g_exc = g_exc + recurrent_input

        g_inh = self.recent_spikes.unsqueeze(0) * self.cortex_config.inhibition_strength if self.cortex_config.lateral_inhibition else None

        output_spikes, voltage = self.neurons(g_exc, g_inh)

        self.output_trace = self.output_trace * decay + output_spikes.squeeze()
        self.recent_spikes = self.recent_spikes * 0.9 + output_spikes.squeeze()
        self.state.spikes = output_spikes
        self.state.t += 1

        return output_spikes

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        novelty: float = 0.0,
        attention: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.ach.compute(novelty=novelty, attention=attention)
        effective_lr = self.cortex_config.hebbian_lr * self.ach.get_learning_modulation()

        old_weights = self.weights.detach().clone()
        self.weights = hebbian_update(
            self.weights, self.input_trace, output_spikes,
            learning_rate=effective_lr, w_max=self.config.w_max,
            heterosynaptic_ratio=self.cortex_config.heterosynaptic_ratio,
        )

        dw = self.weights - old_weights
        ltp = dw[dw > 0].sum().item() if (dw > 0).any() else 0.0
        ltd = dw[dw < 0].sum().item() if (dw < 0).any() else 0.0

        if output_spikes.sum() > 0:
            rate_hz = output_spikes.mean().item() * (1000.0 / self.config.dt_ms)
            self.bcm_threshold = update_bcm_threshold(
                self.bcm_threshold, rate_hz, self.cortex_config.bcm_target_rate_hz,
                tau=self.cortex_config.bcm_tau_ms, min_threshold=0.01, max_threshold=2.0,
            )

        self.ach.decay(self.config.dt_ms)
        return {"ltp": ltp, "ltd": ltd, "net_change": ltp + ltd, "ach_level": self.ach.level}

    def reset(self) -> None:
        super().reset()
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.recent_spikes.zero_()
        self.ach.reset()
        if self.neurons is not None:
            self.neurons.reset_state(1)
