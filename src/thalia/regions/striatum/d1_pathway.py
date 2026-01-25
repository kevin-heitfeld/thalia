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

        D1 pathways use DIRECT dopamine modulation:
        - Dopamine burst (DA+): Strengthen eligible synapses (LTP)
        - Dopamine dip (DA-): Weaken eligible synapses (LTD)

        This implements the "GO" signal: rewarded actions become stronger.

        Args:
            config: Pathway configuration with n_input, n_output, learning rates
        """
        super().__init__(config)
        self.pathway_name = "D1"
        self.dopamine_polarity = 1.0  # Direct modulation
