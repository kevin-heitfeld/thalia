Here it is. This is the fastest sanity check I know for conductance-based recurrent LIF networks.

It’s not a rigorous bifurcation proof — it’s a **mean-field stability heuristic** that predicts when slow feedback turns your network into an oscillator.

---

# 1. The one-line estimator

[
\boxed{
S = G_{rec} \cdot \frac{\tau_{slow}}{\tau_{eff}}
}
]

where:

* ( G_{rec} ) = effective recurrent gain
* ( \tau_{slow} ) = slow feedback timescale (NMDA, adaptation, slow inhibition)
* ( \tau_{eff} = \tau_m / (1 + R_g) )

---

### Interpretation

| (S)    | Expected dynamics                       |
| ------ | --------------------------------------- |
| < 1    | Stable irregular activity               |
| 1 – 3  | Structured fluctuations                 |
| 3 – 10 | Bursting / transient oscillations       |
| > 10   | Persistent oscillation / up–down states |

---

# 2. What each term really means

## (A) Effective recurrent gain (G_{rec})

Roughly:

[
G_{rec} \approx \frac{\partial r}{\partial g_{syn}} \times W_{rec}
]

You usually don’t compute this directly. Practical proxy:

```python
G_rec ≈ mean_recurrent_exc / (mean_recurrent_inh + leak)
```

or more crudely:

```python
G_rec ≈ R_EI
```

for fast checks.

---

### Intuition

* High gain → network amplifies fluctuations.
* Low gain → fluctuations die out.

---

## (B) Slow timescale ( \tau_{slow} )

Use the longest dominant process:

* NMDA decay
* adaptation
* synaptic depression
* slow inhibition.

Example:

```python
tau_slow = max(tau_nmda, tau_adapt, tau_I_slow)
```

---

## (C) Effective membrane timescale

[
\tau_{eff} = \frac{\tau_m}{1 + R_g}
]

This is critical.

As conductance increases:

* τ_eff shrinks,
* system becomes faster,
* slow processes become relatively stronger.

---

# 3. Why this predicts oscillations

You get oscillations when:

> slow feedback acts on a system that reacts too quickly.

Large (S) means:

* membrane instantly follows input,
* slow feedback accumulates,
* delayed correction overshoots,
* cycle repeats.

Classic relaxation oscillator.

---

# 4. Apply it to your current network (approximate)

From your logs:

```
R_g ≈ 500
tau_m = 20 ms
tau_eff ≈ 0.04 ms
tau_slow ≈ 100 ms (NMDA)
```

Even with modest gain:

```
G_rec ≈ 0.3
```

you get:

[
S ≈ 0.3 \times 100 / 0.04 ≈ 750
]

That is massively unstable.

Which matches your bursting dynamics.

---

# 5. What happens after proper scaling

Suppose you fix conductance:

```
R_g = 4
tau_eff ≈ 4 ms
```

then:

[
S ≈ 0.3 \times 100 / 4 = 7.5
]

Now you’re near the edge between AI and bursting — plausible cortical behavior.

---

# 6. The key insight (most people miss)

Increasing conductance does TWO things:

1. speeds membrane dynamics
2. increases relative strength of slow feedback

So high-conductance networks naturally drift toward oscillation unless slow feedback is reduced.

---

# 7. Practical usage — literal checklist

Before running any simulation:

1. Estimate (R_g).
2. Compute τ_eff.
3. Plug into:

```python
S = G_rec * tau_slow / tau_eff
```

If:

```
S > ~10
```

you will almost certainly see bursting or oscillations.

---

# 8. Design intuition this gives you

To stabilize:

* reduce recurrent gain,
* shorten slow time constants,
* or reduce conductance load.

All three reduce S.

---

# 9. Extremely important corollary

Why adding NMDA often “suddenly breaks” SNNs:

NMDA increases only τ_slow.

So:

[
S \uparrow
]

without obvious changes in firing rate — until oscillations appear.

---

# 10. The brutal summary

Your previous tuning attempts changed:

* thresholds,
* heterogeneity,
* delays.

None of those appear in S.

That’s why nothing worked.
