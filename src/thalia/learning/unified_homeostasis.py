"""Unified homeostasis mechanism for synaptic scaling and intrinsic plasticity."""

from __future__ import annotations

import torch


class UnifiedHomeostasis:
    """Unified homeostasis mechanism for synaptic scaling and intrinsic plasticity."""

    @staticmethod
    def compute_excitability_modulation(
        activity_history: torch.Tensor,
        activity_target: float,
        tau: float,
    ) -> torch.Tensor:
        """Compute excitability modulation factor based on activity deviation from target.

        Arguments:
            activity_history: Tensor of recent activity (e.g., firing rates) for a population.
            activity_target: Desired target activity level (e.g., target firing rate).
            tau: Time constant for modulation (controls sensitivity to deviations).

        Returns:
            modulation: A multiplicative factor where >1 means increased excitability (lower threshold)
                        and <1 means decreased excitability (higher threshold).
        """
        # Error from target
        error = activity_history - activity_target

        # Modulation: high activity → lower excitability
        modulation = 1.0 - error / tau

        # Bound to reasonable range
        modulation = modulation.clamp(0.5, 2.0)

        return modulation
