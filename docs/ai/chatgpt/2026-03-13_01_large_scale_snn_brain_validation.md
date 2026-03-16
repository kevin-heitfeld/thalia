For large-scale spiking simulations, researchers do **not rely on rasters alone**. The standard validation set in computational neuroscience typically includes four quantitative diagnostics that detect instability, pathological synchrony, or unrealistic dynamics.

Below are the most informative ones.

---

# 1. Population rate spectrum (oscillation structure)

Compute the power spectrum of the global population firing rate.

[
R(t) = \frac{1}{N}\sum_i s_i(t)
]

where (s_i(t)) is the spike train.

Then compute the PSD of (R(t)).

### Expected biological signature

| regime                 | spectrum          |
| ---------------------- | ----------------- |
| healthy cortex         | (1/f)-like slope  |
| gamma oscillations     | 30–80 Hz bump     |
| theta/alpha            | 4–12 Hz peak      |
| pathological synchrony | huge narrow spike |

Healthy networks show **broadband structure**, not razor-sharp peaks.

### Why it matters

Pathological synchronization often **looks fine in rasters** but shows up as:

* strong spectral spikes
* narrowband oscillations.

---

# 2. Pairwise spike correlation distribution

Compute correlations between neuron spike trains:

[
C_{ij} = corr(s_i, s_j)
]

Look at the distribution of (C_{ij}).

### Biological ranges

| network                | mean correlation |
| ---------------------- | ---------------- |
| cortex in vivo         | 0.01–0.1         |
| strong assemblies      | 0.1–0.2          |
| pathological synchrony | >0.3             |

Healthy cortex is **weakly correlated but not independent**.

### Diagnostic

Plot:

```
histogram(pairwise_correlations)
```

Expected shape:

* centered near **0.02–0.05**
* long but sparse positive tail.

---

# 3. Inter-spike interval statistics (ISI irregularity)

Compute the **coefficient of variation of ISI**:

[
CV = \frac{\sigma_{ISI}}{\mu_{ISI}}
]

### Biological regimes

| neuron type         | CV   |
| ------------------- | ---- |
| Poisson-like cortex | ~1   |
| bursting            | >1.2 |
| pacemaker           | <0.5 |

Healthy cortex shows **CV ≈ 1** (asynchronous irregular regime).

If CV drops near 0, the network is **too deterministic**.

---

# 4. Avalanche statistics (criticality test)

Neural systems often operate near **critical branching dynamics**.

Procedure:

1. Bin spikes (1–5 ms bins).
2. Define avalanche = contiguous active bins.
3. Measure avalanche size distribution.

Expected power-law:

[
P(s) \propto s^{-1.5}
]

This is widely reported in cortical recordings.

### Interpretation

| distribution           | meaning                   |
| ---------------------- | ------------------------- |
| exponential            | subcritical (too quiet)   |
| power-law              | critical (healthy)        |
| heavy giant avalanches | supercritical (epileptic) |

---

# What healthy SNN simulations look like

A well-tuned large network usually satisfies:

| metric                | target    |
| --------------------- | --------- |
| mean firing rate      | 0.1–10 Hz |
| ISI CV                | ~1        |
| pairwise correlation  | 0.01–0.05 |
| population spectrum   | broadband |
| E/I conductance ratio | 3–6       |

If those hold simultaneously, the model is almost always in a realistic regime.

---

# What most people miss

The **single most revealing diagnostic** is actually:

### membrane potential distribution

In vivo cortical neurons show:

```
mean ≈ -60 mV
std ≈ 4–6 mV
long depolarization tail
```

If your voltage distributions resemble that, the conductance regime is correct.

Your earlier violin plots already suggest you are **close to that regime**.

---

# If you want a stronger validation

Large SNN brain projects (e.g., Blue Brain Project and Human Brain Project) also validate using:

**propagation tests**

Inject a small stimulus and measure:

* signal latency
* amplification
* decay across regions.

This ensures the network **transmits information instead of exploding**.
