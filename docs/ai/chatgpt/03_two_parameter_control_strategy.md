Here’s a concise, physics-grounded approach to **stabilize high-conductance recurrent SNNs** using just two knobs: **effective recurrent gain** and **slow-timescale feedback**. This is the “thermostat” for AI dynamics I mentioned.

---

# 1. Two-Parameter Control Overview

You want to keep your **stability estimator**:

[
S = G_{rec} \frac{\tau_{slow}}{\tau_{eff}}
]

in the safe AI range, e.g.,

[
1 \lesssim S \lesssim 3
]

We can control (S) by tuning **just two effective parameters**:

1. **Normalized recurrent gain (G_{rec})**

   * Combines excitatory and inhibitory weights across the recurrent network.
   * Proxy:

   [
   G_{rec} \approx \frac{\langle g_\text{exc} \rangle}{\langle g_\text{inh} \rangle + g_\text{leak}}
   ]

2. **Effective slow timescale (\tau_{slow})**

   * Dominated by the largest slow process (NMDA, adaptation, slow inhibition).
   * Reduces to a single scalar:

   [
   \tau_{slow} = \max(\tau_\text{NMDA}, \tau_\text{adapt}, \tau_\text{GABA-B})
   ]

---

# 2. Control Rules

Let (S^*) be your target AI-stable value (e.g., 2.0).

**Rule 1 — Adjust Gain:**

[
G_{rec} \gets \frac{S^* \cdot \tau_{eff}}{\tau_{slow}}
]

* Downscale excitatory synapses or scale inhibition up if (G_{rec}) is too high.
* Can be applied **row-wise** for local stability in heterogeneous networks.

**Rule 2 — Adjust Slow Timescale:**

[
\tau_{slow} \gets \frac{S^* \cdot \tau_{eff}}{G_{rec}}
]

* Reduce NMDA or adaptation time constants if network is oscillatory.
* Increase slow timescale if network is too damped (quiet, low variability).

---

# 3. Practical Workflow

1. Compute **effective τ_eff** for your current conductance regime:

[
\tau_{eff} = \frac{\tau_m}{1 + R_g}, \quad R_g = \frac{\sum g_\text{syn}}{g_\text{leak}}
]

2. Measure or estimate **G_rec**:

```python
G_rec = mean_exc / (mean_inh + g_leak)
```

3. Compute current S:

```python
S = G_rec * tau_slow / tau_eff
```

4. Apply **one of the two adjustments**:

* **If S > S*** → reduce G_rec or τ_slow
* **If S < S*** → increase G_rec or τ_slow

5. Iterate once — this usually lands you in AI regime.

---

# 4. Key Notes

* You **don’t need to tweak every weight individually**; just scale excitatory vs inhibitory balance and/or slow processes.
* This **thermostat works for heterogeneous networks** if applied to averages per layer/region.
* Works for **both conductance-based and current-based LIF networks** (just interpret τ_eff accordingly).

---

# 5. Optional Enhancement

Add a **secondary check** using **CV_ISI**:

[
\text{CV}_{ISI} \approx 0.8 - 1.2 \quad \text{(AI)}
]

* If CV_ISI < 0.5 → reduce S
* If CV_ISI > 1.5 → increase S

This closes the loop for networks where mean-field estimates are rough.
