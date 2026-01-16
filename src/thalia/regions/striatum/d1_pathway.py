"""
D1 Pathway - Direct/Go Pathway

D1-MSNs express D1 dopamine receptors and form the DIRECT pathway:
- Projects directly to output nuclei (GPi/SNr)
- Dopamine EXCITES D1 receptors → facilitates action initiation
- Learning: DA+ → LTP (strengthen successful actions)
            DA- → LTD (weaken unsuccessful actions)

Biological role: "GO" signal for action selection
"""

from __future__ import annotations

from typing import Dict, Any

import torch

from .pathway_base import StriatumPathway, StriatumPathwayConfig


class D1Pathway(StriatumPathway):
    """
    D1-MSN population implementing the direct/Go pathway.

    Key features:
    - Positive dopamine modulation (DA+ → strengthen, DA- → weaken)
    - Direct inhibitory projections to output nuclei
    - Facilitates action initiation and selection

    Three-factor rule for D1:
        Δw = eligibility × dopamine × learning_rate

    When dopamine is positive (reward):
        - Eligible synapses strengthen (LTP)
        - Non-eligible synapses weaken slightly (heterosynaptic LTD)

    When dopamine is negative (punishment):
        - Eligible synapses weaken (LTD)
        - Non-eligible synapses strengthen slightly (heterosynaptic LTP)
    """

    def __init__(self, config: StriatumPathwayConfig):
        """Initialize D1 pathway (direct/Go pathway).

        Args:
            config: Pathway configuration with n_input, n_output, learning rates
        """
        super().__init__(config)
        self.pathway_name = "D1"

    def apply_dopamine_modulation(
        self,
        dopamine: float,
        heterosynaptic_ratio: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Apply D1-specific dopamine modulation using ThreeFactorStrategy.

        D1 RESPONSE TO DOPAMINE:
        - Dopamine burst (DA+): Strengthen eligible synapses (LTP)
        - Dopamine dip (DA-): Weaken eligible synapses (LTD)

        This implements the "GO" signal: rewarded actions become stronger.

        Args:
            dopamine: Dopamine signal (RPE), typically in [-1, +1]
            heterosynaptic_ratio: Not used (kept for API compatibility)

        Returns:
            Metrics dict from strategy:
                - ltp: Total LTP magnitude
                - ltd: Total LTD magnitude
                - net_change: Net weight change
                - modulator: Dopamine value used
                - eligibility_mean: Average eligibility trace
        """
        # Use strategy to compute weight update
        # D1: positive dopamine modulation (DA+ → strengthen, DA- → weaken)
        new_weights, metrics = self.learning_strategy.compute_update(
            weights=self.weights.data,
            pre=torch.ones(1, device=self.device),  # Dummy (eligibility already accumulated)
            post=torch.ones(1, device=self.device),  # Dummy (eligibility already accumulated)
            modulator=dopamine,
        )

        # Update weights
        self.weights.data = new_weights

        # Add pathway identifier
        metrics['pathway'] = 'D1'
        metrics['dopamine_sign'] = 'positive' if dopamine > 0 else 'negative' if dopamine < 0 else 'zero'

        return metrics
