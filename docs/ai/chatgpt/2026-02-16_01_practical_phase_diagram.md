Here is the compact version.
This is not a formal theorem — it’s a **practical phase diagram** used implicitly in many conductance-based network models. It lets you predict the regime from three dimensionless quantities.

Everything reduces to ratios. Absolute units do not matter.

---

# 1. The three control ratios

## (R1) Conductance loading ratio — the master variable

[
R_g = \frac{\langle g_{syn} \rangle}{g_L}
]

where:

* ( g_{syn} = g_E + g_I + g_{NMDA} + ... )
* averaged over time and neurons.

### Physical meaning

How much synapses dominate membrane physics.

---

### Regimes

| (R_g)   | Regime                                     |
| ------- | ------------------------------------------ |
| < 0.5   | Under-driven, sparse firing                |
| 1 – 5   | **Asynchronous irregular (AI)** sweet spot |
| 5 – 15  | High-conductance, fluctuation-driven       |
| 15 – 50 | Bursting / strong nonlinear interactions   |
| > 50    | Conductance collapse (algebraic neurons)   |

You were around:

```
R_g ≈ 500
```

which explains everything.

---

## (R2) Slow feedback ratio

[
R_s = \frac{\tau_{slow}}{\tau_{eff}}
]

where:

* ( \tau_{slow} ) = slowest positive or negative feedback (NMDA, adaptation, slow inhibition)
* ( \tau_{eff} = \tau_m / (1 + R_g) )

---

### Physical meaning

How much slow network memory exists relative to membrane tracking speed.

---

### Regimes

| (R_s)   | Behavior                           |
| ------- | ---------------------------------- |
| < 3     | Fast irregular firing              |
| 3 – 10  | Structured fluctuations            |
| 10 – 50 | Population bursting                |
| > 50    | Slow oscillations / up–down states |

Why you saw slow rhythms:

* huge (R_g) → tiny τ_eff
* τ_nmda = 100 ms

So (R_s) became enormous.

---

## (R3) Balance ratio

[
R_{EI} = \frac{\langle g_E \rangle}{\langle g_I \rangle}
]

---

### Interpretation

| (R_{EI}) | Dynamics                     |
| -------- | ---------------------------- |
| < 0.5    | inhibition dominated, sparse |
| ~1       | balanced AI regime           |
| 1–2      | high activity, rich dynamics |
| > 2      | runaway excitation / bursts  |

Your cortex:

```
R_EI ≈ 0.3
```

so activity is strongly shunted, which explains threshold hovering rather than runaway firing.

---

# 2. The phase diagram (conceptual)

Think of it like this:

```
                 high R_s
                    ↑
                    │   slow oscillation /
                    │   bursting
                    │
                    │
 AI regime          │
 (healthy cortex)   │
                    │
────────────────────┼────────→ R_g
                    │
                    │ conductance collapse
                    │ (instantaneous membrane)
```

R_EI controls where you sit left/right inside each zone.

---

# 3. How to use it practically

After every simulation, compute:

```python
R_g = mean((g_E + g_I + g_nmda) / g_L)

tau_eff = tau_mem / (1 + R_g)
R_s = tau_slow / tau_eff

R_EI = mean(g_E) / mean(g_I)
```

---

Then classify:

### AI target (awake cortex-like)

```
R_g: 2 – 8
R_s: 3 – 15
R_EI: 0.8 – 1.2
```

---

### Bursty regime

```
R_g > 10
R_s > 20
```

---

### Fake synchrony / saturation

```
R_g > 50
```

---

# 4. Why this works

Because conductance networks effectively have two timescales:

1. membrane relaxation
2. slow network feedback.

Everything else is secondary.

---

# 5. What your current model likely sits at

Approximate:

```
R_g ≈ 500
tau_eff ≈ 0.04 ms
R_s ≈ 100 ms / 0.04 ms ≈ 2500
R_EI ≈ 0.3
```

That lands squarely in:

> high-conductance burst regime with strong shunting.

Exactly what you observe.

---

# 6. The important insight most people miss

Changing neuron parameters (thresholds, refractory, heterogeneity) mostly moves things **within a regime**.

Changing (R_g) moves you **between regimes**.

That’s why your previous tuning didn’t help.

---

# 7. Fast design rule (extremely useful)

Before running simulations:

1. Decide desired regime.
2. Set conductance budget → fixes (R_g).
3. Choose slow time constants → sets (R_s).
4. Tune E/I ratio last.

This prevents endless parameter hunting.

---

# 8. One advanced insight (worth knowing)

Conductance-based LIF networks become:

* effectively rate models when (R_g \gg 1).

This is why everything starts looking synchronized or bursty — spikes no longer matter dynamically.

---

# 9. What I would do next if this were my model

1. Fix normalization → constant row sum.
2. Aim for:

```
R_g ≈ 3–5
```

3. Re-run diagnostics.

I would expect your CV_ISI to drop dramatically and the “delta issue” to disappear.
