# Configuration Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-16 21:25:28
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all configuration dataclasses in Thalia.

Total: 3 configuration classes

## Configuration Classes

### [``HippocampusConfig``](../../src/thalia/regions/hippocampus/config.py#L50)

**Source**: [`thalia/regions/hippocampus/config.py`](../../src/thalia/regions/hippocampus/config.py)

**Description**: Configuration for hippocampus (trisynaptic circuit).

**Fields**:

| Field | Type | Default |
|-------|------|-------|
| `learning_rate` | `float` | `LEARNING_RATE_ONE_SHOT` |
| `dg_sparsity` | `float` | `HIPPOCAMPUS_SPARSITY_TARGET` |
| `dg_inhibition` | `float` | `5.0` |
| `ca3_recurrent_strength` | `float` | `0.4` |
| `ca3_sparsity` | `float` | `0.1` |
| `ca2_sparsity` | `float` | `0.12` |
| `ca2_plasticity_resistance` | `float` | `0.1` |
| `ca1_sparsity` | `float` | `0.15` |
| `coincidence_window` | `float` | `5.0` |
| `enable_spillover` | `bool` | `True` |
| `spillover_mode` | `str` | `'connectivity'` |
| `spillover_strength` | `float` | `0.18` |
| `match_threshold` | `float` | `0.3` |
| `nmda_tau` | `float` | `50.0` |
| `nmda_threshold` | `float` | `0.4` |
| `nmda_steepness` | `float` | `12.0` |
| `ampa_ratio` | `float` | `0.05` |
| `ca3_ca2_learning_rate` | `float` | `0.001` |
| `ec_ca2_learning_rate` | `float` | `0.01` |
| `ca2_ca1_learning_rate` | `float` | `0.005` |
| `ec_ca1_learning_rate` | `float` | `0.5` |
| `ffi_threshold` | `float` | `0.3` |
| `ffi_strength` | `float` | `0.8` |
| `ffi_tau` | `float` | `5.0` |
| `dg_to_ca3_delay_ms` | `float` | `0.0` |
| `ca3_to_ca2_delay_ms` | `float` | `0.0` |
| `ca2_to_ca1_delay_ms` | `float` | `0.0` |
| `ca3_to_ca1_delay_ms` | `float` | `0.0` |
| `ca3_persistent_tau` | `float` | `300.0` |
| `ca3_persistent_gain` | `float` | `3.0` |
| `ec_l3_input_size` | `int` | `0` |
| `theta_gamma_enabled` | `bool` | `True` |
| `stp_enabled` | `bool` | `True` |
| `stp_mossy_type` | `STPType` | `STPType.FACILITATING_STRONG` |
| `stp_ca3_ca2_type` | `STPType` | `STPType.DEPRESSING` |
| `stp_ca2_ca1_type` | `STPType` | `STPType.FACILITATING` |
| `stp_ec_ca2_type` | `STPType` | `STPType.DEPRESSING` |
| `stp_schaffer_type` | `STPType` | `STPType.DEPRESSING` |
| `stp_ec_ca1_type` | `STPType` | `STPType.DEPRESSING` |
| `stp_ca3_recurrent_type` | `STPType` | `STPType.DEPRESSING_FAST` |
| `adapt_increment` | `float` | `0.5` |
| `ca3_feedback_inhibition` | `float` | `0.3` |
| `gap_junctions_enabled` | `bool` | `True` |
| `gap_junction_strength` | `float` | `0.12` |
| `gap_junction_threshold` | `float` | `0.25` |
| `gap_junction_max_neighbors` | `int` | `8` |
| `heterosynaptic_ratio` | `float` | `0.1` |
| `theta_reset_persistent` | `bool` | `True` |
| `theta_reset_fraction` | `float` | `0.5` |
| `phase_diversity_init` | `bool` | `True` |
| `phase_jitter_std_ms` | `float` | `5.0` |
| `use_her` | `bool` | `True` |
| `her_k_hindsight` | `int` | `4` |
| `her_replay_ratio` | `float` | `0.8` |
| `her_strategy` | `str` | `'future'` |
| `her_goal_tolerance` | `float` | `0.1` |
| `her_buffer_size` | `int` | `1000` |
| `use_multiscale_consolidation` | `bool` | `False` |
| `fast_trace_tau_ms` | `float` | `60000.0` |
| `slow_trace_tau_ms` | `float` | `3600000.0` |
| `consolidation_rate` | `float` | `0.001` |
| `slow_trace_contribution` | `float` | `0.1` |

**Used By**:

- [`hippocampus`](../../src/thalia/regions/hippocampus/trisynaptic.py#L154)

---

### [``LayeredCortexConfig``](../../src/thalia/regions/cortex/config.py#L28)

**Source**: [`thalia/regions/cortex/config.py`](../../src/thalia/regions/cortex/config.py)

**Description**: Configuration for layered cortical microcircuit.

**Fields**:

| Field | Type | Default |
|-------|------|-------|
| `l4_sparsity` | `float` | `0.15` |
| `l23_sparsity` | `float` | `0.1` |
| `l5_sparsity` | `float` | `0.2` |
| `l6a_sparsity` | `float` | `0.12` |
| `l6b_sparsity` | `float` | `0.15` |
| `l23_recurrent_strength` | `float` | `0.3` |
| `l23_recurrent_decay` | `float` | `0.9` |
| `input_to_l4_strength` | `float` | `2.0` |
| `l4_to_l23_strength` | `float` | `1.5` |
| `l23_to_l5_strength` | `float` | `1.5` |
| `l23_to_l6a_strength` | `float` | `0.8` |
| `l23_to_l6b_strength` | `float` | `2.0` |
| `l23_top_down_strength` | `float` | `0.2` |
| `l6a_to_trn_strength` | `float` | `0.8` |
| `l6b_to_relay_strength` | `float` | `0.6` |
| `enable_spillover` | `bool` | `True` |
| `spillover_mode` | `str` | `'connectivity'` |
| `spillover_strength` | `float` | `0.15` |
| `gap_junctions_enabled` | `bool` | `True` |
| `gap_junction_strength` | `float` | `0.12` |
| `gap_junction_threshold` | `float` | `0.25` |
| `gap_junction_max_neighbors` | `int` | `8` |
| `a_plus` | `float` | `STDP_A_PLUS_CORTEX` |
| `a_minus` | `float` | `STDP_A_MINUS_CORTEX` |
| `l23_recurrent_w_min` | `float` | `-1.5` |
| `l23_recurrent_w_max` | `float` | `1.0` |
| `adapt_increment` | `float` | `ADAPT_INCREMENT_CORTEX_L23` |
| `ffi_threshold` | `float` | `0.3` |
| `ffi_strength` | `float` | `0.8` |
| `ffi_tau` | `float` | `5.0` |
| `l4_to_l23_delay_ms` | `float` | `2.0` |
| `l23_to_l5_delay_ms` | `float` | `2.0` |
| `l23_to_l6a_delay_ms` | `float` | `2.0` |
| `l23_to_l6b_delay_ms` | `float` | `3.0` |
| `l6a_to_trn_delay_ms` | `float` | `10.0` |
| `l6b_to_relay_delay_ms` | `float` | `5.0` |
| `gamma_attention_width` | `float` | `0.3` |
| `bcm_enabled` | `bool` | `False` |
| `bcm_config` | `Optional[BCMConfig]` | `None` |
| `robustness` | `Optional[RobustnessConfig]` | `None` |
| `use_layer_heterogeneity` | `bool` | `False` |
| `layer_tau_mem` | `Dict[str, float]` | `(factory)` |
| `layer_v_threshold` | `Dict[str, float]` | `(factory)` |
| `layer_adaptation` | `Dict[str, float]` | `(factory)` |

**Used By**:

- [`cortex`](../../src/thalia/regions/cortex/layered_cortex.py#L160)

---

### [``StriatumConfig``](../../src/thalia/regions/striatum/config.py#L23)

**Source**: [`thalia/regions/striatum/config.py`](../../src/thalia/regions/striatum/config.py)

**Description**: Configuration specific to striatal regions (behavior only, no sizes).

**Fields**:

| Field | Type | Default |
|-------|------|-------|
| `d1_d2_ratio` | `float` | `0.5` |
| `learning_rate` | `float` | `0.005` |
| `lateral_inhibition` | `bool` | `True` |
| `inhibition_strength` | `float` | `2.0` |
| `d1_lr_scale` | `float` | `1.0` |
| `d2_lr_scale` | `float` | `1.0` |
| `d1_da_sensitivity` | `float` | `1.0` |
| `d2_da_sensitivity` | `float` | `1.0` |
| `homeostatic_soft` | `bool` | `True` |
| `homeostatic_rate` | `float` | `0.1` |
| `activity_decay` | `float` | `EMA_DECAY_FAST` |
| `baseline_pressure_enabled` | `bool` | `True` |
| `baseline_pressure_rate` | `float` | `0.015` |
| `baseline_target_net` | `float` | `0.0` |
| `softmax_action_selection` | `bool` | `True` |
| `softmax_temperature` | `float` | `2.0` |
| `adaptive_exploration` | `bool` | `True` |
| `performance_window` | `int` | `10` |
| `performance_exploration_scale` | `float` | `0.3` |
| `min_tonic_dopamine` | `float` | `0.1` |
| `max_tonic_dopamine` | `float` | `0.5` |
| `use_td_lambda` | `bool` | `True` |
| `td_lambda` | `float` | `0.9` |
| `td_gamma` | `float` | `0.99` |
| `td_lambda_accumulating` | `bool` | `True` |
| `ucb_exploration` | `bool` | `True` |
| `ucb_coefficient` | `float` | `2.0` |
| `uncertainty_temperature` | `float` | `0.05` |
| `min_exploration_boost` | `float` | `0.05` |
| `rpe_enabled` | `bool` | `True` |
| `rpe_learning_rate` | `float` | `0.1` |
| `rpe_initial_value` | `float` | `0.0` |
| `tonic_dopamine` | `float` | `0.3` |
| `tonic_modulates_d1_gain` | `bool` | `True` |
| `tonic_d1_gain_scale` | `float` | `0.5` |
| `tonic_modulates_exploration` | `bool` | `True` |
| `tonic_exploration_scale` | `float` | `0.1` |
| `beta_modulation_strength` | `float` | `0.3` |
| `fsi_enabled` | `bool` | `True` |
| `fsi_ratio` | `float` | `0.02` |
| `gap_junctions_enabled` | `bool` | `True` |
| `gap_junction_strength` | `float` | `0.15` |
| `gap_junction_threshold` | `float` | `0.25` |
| `gap_junction_max_neighbors` | `int` | `10` |
| `use_goal_conditioning` | `bool` | `True` |
| `pfc_size` | `int` | `128` |
| `goal_modulation_strength` | `float` | `0.5` |
| `goal_modulation_lr` | `float` | `LEARNING_RATE_HEBBIAN_SLOW` |
| `d1_to_output_delay_ms` | `float` | `15.0` |
| `d2_to_output_delay_ms` | `float` | `25.0` |
| `stp_enabled` | `bool` | `True` |
| `heterogeneous_stp` | `bool` | `False` |
| `stp_variability` | `float` | `0.3` |
| `stp_seed` | `Optional[int]` | `None` |
| `use_multiscale_eligibility` | `bool` | `False` |
| `fast_eligibility_tau_ms` | `float` | `500.0` |
| `slow_eligibility_tau_ms` | `float` | `60000.0` |
| `eligibility_consolidation_rate` | `float` | `0.01` |
| `slow_trace_weight` | `float` | `0.3` |
| `growth_enabled` | `bool` | `True` |
| `reserve_capacity` | `float` | `0.5` |

**Used By**:

- [`striatum`](../../src/thalia/regions/striatum/striatum.py#L151)

---

## 💡 Configuration Best Practices

### Common Patterns

1. **Start with defaults**: All configs have biologically-motivated defaults
2. **Override selectively**: Only change what's needed for your task
3. **Validate early**: Use config validation before training
4. **Document changes**: Keep notes on why you changed defaults

### Performance Considerations

- **Layer sizes**: Larger = more capacity but slower training
- **Sparsity**: Higher sparsity = faster but less connectivity
- **Learning rates**: Too high = instability, too low = slow learning
- **Time constants**: Should match biological ranges (ms scale)

### Common Pitfalls

⚠️ **Too small networks**: Use at least 64 neurons per region

⚠️ **Mismatched time scales**: Keep tau values in biological range (5-200ms)

⚠️ **Extreme learning rates**: Stay within 0.0001-0.01 range

⚠️ **Disabled plasticity**: Ensure learning strategies are enabled

## 📚 Related Documentation

- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - Components using these configs
- [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md) - Learning rules and parameters
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Configuration examples
- [ENUMERATIONS_REFERENCE.md](ENUMERATIONS_REFERENCE.md) - Enum types used in configs
