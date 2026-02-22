# TODO

- Re-enable:
  - Synaptic weights: `GLOBAL_WEIGHT_SCALE`
  - Plasticity: `GLOBAL_LEARNING_ENABLED`
  - Homeostasis: `GLOBAL_HOMEOSTASIS_ENABLED`
  - Baseline noise: `baseline_noise_conductance_enabled`
  - Neuromodulation: `enable_neuromodulation`
  - SNR baseline drive: `baseline_drive`
- Review CA3 persistence
  - `ca3_persistent_gain`
- Implement synaptic scaling for all regions (currently only implemented for Cortex)
- Implement spillover for all regions?
- Are current diagnostics sufficient? Add real brain diagnostics like EEG?
  - Spike Raster Plots
