# Parameter Dependencies

This document catalogues derived parameter relationships across Thalia's
configuration system.  Some are enforced programmatically; others are
documented conventions.

## Auto-Computed Parameters

### Axonal Delay Standard Deviation

Every inter-region connection has a mean delay (`axonal_delay_ms`) and a
standard deviation (`axonal_delay_std_ms`).  When constructing connections
via `BrainBuilder.connect()`, if `axonal_delay_std_ms` is not provided it
defaults to **30 %** of the mean:

```
axonal_delay_std_ms = 0.30 × axonal_delay_ms
```

This fraction (`DEFAULT_DELAY_STD_FRACTION` in `brain_builder.py`) reflects
the normal distribution of axon diameters and myelination levels within a
tract.  All ~90+ connections in the default preset rely on the auto-computed
default — override only when a specific pathway has empirically different
variability.

### Homeostasis Effective Time Constant

A warning fires during config construction when

```
τ_eff = dt_ms / lr_per_ms  <  1000 ms
```

for any homeostatic learning rate.  Biological homeostasis operates on
minutes-to-hours timescales; sub-second τ_eff is almost certainly a bug.
(See `NeuralRegionConfig.__post_init__`.)

## Documented Conventions (Not Enforced)

### Mesocortical DA Layer Fractions

DA innervation density is defined relative to **L5** (the primary
mesocortical target).  Default values on `CorticalColumnConfig`:

| Layer | Fraction | Ratio to L5 |
|-------|----------|-------------|
| L2/3  | 0.075    |  25 %       |
| L4    | 0.030    |  10 %       |
| L5    | 0.300    | 100 %       |
| L6a   | 0.135    |  45 %       |
| L6b   | 0.135    |  45 %       |

PFC overrides `da_fractions.l23` to 0.30 (100 % of L5) for working-memory
gating.  These ratios are **not** enforced programmatically — changing
`da_fractions.l5` alone does not automatically rescale the other layers.

### DRN Serotonin Drive Multiplier

`DorsalRapheNucleus` multiplies its config's `tonic_drive_gain` by a fixed
**×20** factor to obtain the serotonin drive conductance:

```python
serotonin_drive_gain = self.config.tonic_drive_gain * 20.0
```

With `tonic_drive_gain = 0.78` (default), this yields a serotonin drive of
15.6.  Changing `tonic_drive_gain` without understanding the ×20 multiplier
will produce unexpected firing rates.

### Central Amygdala Threshold Relationship

CeM (medial) threshold is set to **0.9 × CeL** (lateral) threshold:

```
CeM v_threshold = 1.215  (= 1.35 × 0.9)
CeL v_threshold = 1.35
```

This ensures CeM fires slightly more easily than CeL, matching the
disinhibitory circuit: CeL ON-cells → CeM → output.

### Delay Step Quantisation

All delays are discretised at build time:

```
delay_steps = round(delay_ms / dt_ms)
```

At the default `dt_ms = 1.0`, this is a 1:1 mapping.  At coarser timesteps,
short delays (< 2 ms) may collapse to a single step — verify with
`brain.get_connection_delays()` after construction.

## Homeostatic Sub-Config Cross-References

The three homeostatic sub-configs on `NeuralRegionConfig` are independently
tuneable but operate on the same set of neurons:

| Sub-config              | Acts on          | Timescale     | Primary effect          |
|-------------------------|------------------|---------------|-------------------------|
| `homeostatic_gain`      | g_L scale        | τ_ms (seconds)| Intrinsic excitability  |
| `homeostatic_threshold` | Spike threshold   | lr_per_ms     | Firing selectivity      |
| `synaptic_scaling`      | All input weights | lr_per_ms     | Total synaptic drive    |

Setting gain τ_ms very small while keeping threshold lr_per_ms large creates
conflicting drives.  The cross-config tau warning in `__post_init__` catches
the most egregious cases.
