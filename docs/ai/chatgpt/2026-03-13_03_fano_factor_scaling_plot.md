The single visualization most neuroscientists use to quickly judge whether a spiking network behaves like cortex is the **Fano factor vs time-bin plot** (often combined with rate scaling). It compresses irregularity, correlation, and population dynamics into one curve.

---

# Fano Factor Scaling Plot

The **Fano factor** measures spike-count variability:

[
F = \frac{\mathrm{Var}[N]}{\mathrm{E}[N]}
]

Where (N) is the number of spikes in a time window.

You compute this across **different bin sizes**.

---

## Procedure

1. Pool spikes from the population.
2. Bin them at increasing window sizes:

```
1 ms
2 ms
5 ms
10 ms
20 ms
50 ms
100 ms
```

3. Compute

[
F(\Delta t) = \frac{Var[N(\Delta t)]}{E[N(\Delta t)]}
]

Plot **Fano factor vs bin size**.

---

# Expected biological pattern

Real cortex (awake animals) shows:

```
bin size (ms)      Fano factor
------------------------------
1 ms               ~1
5 ms               ~1.1
10 ms              ~1.2
20 ms              ~1.4
50 ms              ~1.7
100 ms             ~2
```

Graphically it looks like:

```
F
2.2 |                     *
2.0 |                 *
1.8 |             *
1.6 |         *
1.4 |      *
1.2 |   *
1.0 | *
    +-----------------------------
      1  5 10 20 50 100 ms
```

Key properties:

* **starts near 1** → Poisson-like spiking
* **gradually increases** → correlations at larger scales

---

# What pathological networks look like

### Too deterministic

```
F ≈ 0.2–0.5
flat line
```

Indicates:

* pacemaker locking
* overly deterministic neurons
* insufficient noise.

---

### Synchronous oscillations

```
F explodes rapidly
F > 4 by 20 ms
```

Indicates:

* network bursting
* runaway excitation.

---

### Dead network

```
F ≈ 1 but mean firing rate ≈ 0
```

Appears healthy but is silent.

---

# Why this plot is powerful

It simultaneously detects:

| issue           | symptom           |
| --------------- | ----------------- |
| synchrony       | steep Fano growth |
| determinism     | Fano < 1          |
| dead network    | low rate          |
| critical regime | smooth growth     |

One curve replaces several diagnostics.

---

# Implementation

Example using your spike raster:

```python
import numpy as np

bin_sizes = [1,2,5,10,20,50,100]  # ms
fano = []

for b in bin_sizes:
    counts = spike_counts.reshape(-1, b).sum(axis=1)
    fano.append(np.var(counts) / np.mean(counts))

plt.plot(bin_sizes, fano, marker='o')
plt.xscale("log")
plt.xlabel("Bin size (ms)")
plt.ylabel("Fano factor")
```

---

# What large brain simulations check

Projects like the Blue Brain Project and Allen Brain Atlas use variants of this to ensure simulated populations exhibit:

* **asynchronous irregular firing**
* weak correlations
* scale-dependent variability.

If your network produces the rising curve above, it is almost always in the correct dynamical regime.
