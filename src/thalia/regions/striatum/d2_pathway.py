"""
D2 Pathway - Indirect/NoGo Pathway

D2-MSNs express D2 dopamine receptors and form the INDIRECT pathway:
- Projects indirectly through GPe → STN → GPi/SNr
- Dopamine INHIBITS D2 receptors → suppresses action inhibition
- Learning: DA+ → LTD (weaken action suppression)
            DA- → LTP (strengthen action suppression)

Biological role: "NOGO" signal for action suppression
"""

from __future__ import annotations

from .pathway_base import StriatumPathway, StriatumPathwayConfig


class D2Pathway(StriatumPathway):
    """
    D2-MSN population implementing the indirect/NoGo pathway.

    Key features:
    - INVERTED dopamine modulation (DA+ → weaken, DA- → strengthen)
    - Indirect pathway through GPe and STN
    - Suppresses inappropriate actions

    Three-factor rule for D2 (INVERTED):
        Δw = eligibility × (-dopamine) × learning_rate

    When dopamine is positive (reward):
        - Eligible synapses WEAKEN (LTD) - "stop inhibiting the good action"
        - Non-eligible synapses strengthen slightly (heterosynaptic LTP)

    When dopamine is negative (punishment):
        - Eligible synapses STRENGTHEN (LTP) - "inhibit the bad action more"
        - Non-eligible synapses weaken slightly (heterosynaptic LTD)

    This implements opponent learning:
    - D1 learns to DO rewarded actions
    - D2 learns to NOT DO punished actions
    """

    def __init__(self, config: StriatumPathwayConfig):
        """Initialize D2 pathway (indirect/NoGo pathway).

        D2 pathways use INVERTED dopamine modulation:
        - Dopamine burst (DA+): WEAKEN eligible synapses (LTD)
        - Dopamine dip (DA-): STRENGTHEN eligible synapses (LTP)

        This implements the "NOGO" signal: punished actions become more inhibited.

        Args:
            config: Pathway configuration with n_input, n_output, learning rates
        """
        super().__init__(config)
        self.pathway_name = "D2"
        self.dopamine_polarity = -1.0  # Inverted modulation
