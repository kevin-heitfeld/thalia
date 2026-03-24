"""Re-exporting facade for biological reference data.

The actual definitions live in three focused modules:

- :mod:`.bio_firing_ranges` — ``RegionSpec``, ``REGION_SPECS``, and all
  region/population lookup functions.
- :mod:`.bio_spectral_specs` — ``EEG_BANDS``, coherence specs, PAC specs.
- :mod:`.bio_neuromodulator_specs` — neuromodulator tonic concentration ranges.

All public names are re-exported here so that existing ``from .bio_ranges import …``
statements continue to work unchanged.
"""

from .bio_firing_ranges import (
    REGION_SPECS,
    RegionSpec,
    _UNSET,
    adaptation_expected_for,
    bio_range,
    burst_window_ms_for,
    ei_ratio_thresholds,
    expected_dominant_band,
    integration_tau_range,
    is_pacemaker_population,
    skip_burst_check_for,
    skip_sync_check_for,
)
from .bio_neuromodulator_specs import (
    nm_tonic_range,
)
from .bio_spectral_specs import (
    CFCSpec,
    CoherenceSpec,
    CFC_SPECS,
    COHERENCE_SPECS,
    EEG_BANDS,
    freq_to_band,
)

__all__ = [
    # Firing rate reference data and lookup functions
    "REGION_SPECS",
    "RegionSpec",
    "_UNSET",
    "adaptation_expected_for",
    "bio_range",
    "burst_window_ms_for",
    "ei_ratio_thresholds",
    "expected_dominant_band",
    "integration_tau_range",
    "is_pacemaker_population",
    "skip_burst_check_for",
    "skip_sync_check_for",
    # Neuromodulator reference data
    "nm_tonic_range",
    # Spectral reference data and lookup functions
    "CFC_SPECS",
    "CFCSpec",
    "COHERENCE_SPECS",
    "EEG_BANDS",
    "CoherenceSpec",
    "freq_to_band",
]
