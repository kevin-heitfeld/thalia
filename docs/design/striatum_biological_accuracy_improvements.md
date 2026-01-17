# Striatum Biological Accuracy Investigation

**Date**: January 14, 2026
**Status**: Investigation & Recommendations
**Context**: Post-refactoring analysis of D1/D2 implementation

## Executive Summary

The current striatum implementation is **biologically sound** at the systems level (opponent pathways, inverted dopamine, three-factor rule). This investigation identifies **4 areas** where biological realism can be enhanced at the **circuit and synaptic levels**.

---

## Current Implementation: Strengths

### ✅ **Excellent** (No Changes Needed)

1. **Opponent Pathway Architecture**
   - D1 (Go) vs D2 (NoGo) with separate populations
   - Independent synaptic weights per pathway
   - Competition via `NET = D1_activity - D2_activity`
   - **Biological basis**: Validated by decades of research (Frank 2005, Kravitz et al. 2010)

2. **Inverted Dopamine Modulation**
   ```python
   # D1: DA+ → LTP, DA- → LTD
   d1_update = eligibility * dopamine

   # D2: DA+ → LTD, DA- → LTP (inverted)
   d2_update = eligibility * (-dopamine)
   ```
   - **Biological basis**: D1 receptors (Gs-coupled, excitatory), D2 receptors (Gi-coupled, inhibitory)
   - **Result**: Reward strengthens "Go" while weakening "NoGo"

3. **Three-Factor Learning Rule**
   - Pre-post activity creates eligibility traces ("synaptic tags")
   - Dopamine (arriving later) gates plasticity
   - **Biological basis**: Experimentally validated (Yagishita et al. 2014)

4. **Multi-Source Architecture**
   - Separate weights for cortex, hippocampus, thalamus per pathway
   - Each MSN integrates convergent inputs
   - **Biological basis**: MSNs receive ~10,000 synapses from multiple regions

---

## Areas for Biological Enhancement

### 1. **Temporal Delays** ⚠️ **Partially Accurate**

#### Current Implementation
```python
# Fixed delays per pathway
d1_delay = 15ms  # Direct pathway
d2_delay = 25ms  # Indirect pathway (10ms longer)
```

#### Issues
1. **Constant delays**: Biological delays are **distance-dependent** and **variable**
2. **No synaptic delays**: Current delays model circuit pathways, not individual synapses
3. **Missing jitter**: Biological spike timing has ~1-2ms variability

#### Biological Reality

**Synaptic Transmission Delays** (should be per-synapse):
- Chemical synapse: 0.5-1.0ms (vesicle release + diffusion + receptor binding)
- Axonal conduction: 0.1-5.0ms depending on distance and myelination
- Dendritic integration: 1-3ms (AMPA rise time)

**Circuit Pathway Delays** (current implementation):
- D1 direct: Striatum → GPi/SNr → Thalamus (2 synapses, ~10-20ms)
- D2 indirect: Striatum → GPe → STN → GPi/SNr → Thalamus (4-5 synapses, ~20-40ms)

#### Recommendations

**Option A: Per-Synapse Delays** (Most Accurate)
```python
# In AxonalProjection (already implemented!)
projection = AxonalProjection(
    sources=[("cortex", "l5")],
    target="striatum",
    delay_ms=1.5,  # Synaptic + axonal delay
    device=device,
)

# In striatum.add_input_source_striatum()
self.input_delays[source_key] = {
    "synaptic": 0.8,    # Chemical transmission
    "axonal": distance_dependent_delay(source, target),
    "dendritic": 1.5,   # AMPA rise time
}
```

**Benefits**:
- ✅ Biologically accurate per-connection delays
- ✅ Supports distance-dependent conduction velocities
- ✅ Can model dendritic integration delays
- ✅ AxonalProjection already has CircularDelayBuffer infrastructure

**Option B: Pathway-Specific Jitter** (Quick Enhancement)
```python
# Add variability to current delays
d1_delay = np.random.normal(15.0, 2.0)  # 15±2ms
d2_delay = np.random.normal(25.0, 3.0)  # 25±3ms
```

**Benefits**:
- ✅ Captures spike timing variability
- ✅ Minimal code change
- ⚠️ Still not per-synapse accurate

---

### 2. **STP Dynamics** ⚠️ **Good, Can Be Enhanced**

#### Current Implementation
```python
# Per-source STP with fixed parameters
corticostriatal: U=0.4, tau_d=200ms, tau_f=10ms (depressing)
thalamostriatal: U=0.25, tau_d=400ms, tau_f=500ms (facilitating)
hippocampal: U=0.46, tau_d=500ms, tau_f=400ms (depressing)
```

#### Issues
1. **Homogeneous within pathway**: All cortical synapses use same STP
2. **No activity dependence**: STP parameters don't adapt to firing patterns
3. **Missing calcium dynamics**: Biological STP depends on residual [Ca²⁺]

#### Biological Reality

**STP Heterogeneity**:
- Even within corticostriatal pathway, synapses show **10-fold variation** in U
- **Location-dependent**: Proximal dendrites (strong depression), distal (weak)
- **Target-dependent**: Cortex→D1 MSNs (more depressing) vs Cortex→D2 MSNs (less)

**Activity-Dependent Adaptation**:
```
After 1 min of 10Hz stimulation:
- U increases by 20-40% (facilitation buildup)
- tau_d decreases by 30% (faster recovery)
```

**Calcium-Based STP** (Tsodyks-Markram with [Ca²⁺]):
```python
# More biologically accurate model
u_jump = U * (1 - u) * (1 + α * residual_calcium)
x_depletion = u * x * (1 + β * residual_calcium)
residual_calcium *= exp(-dt / tau_ca)  # tau_ca ~ 100ms
```

#### Recommendations

**Option A: Heterogeneous STP** (High Impact)
```python
# In add_input_source_striatum()
for i in range(n_input):
    # Sample U from distribution (biological variability)
    U_synapse = np.random.lognormal(mean_U, std_U)

    # Location-dependent: proximal = strong depression
    distance_to_soma = compute_dendritic_distance(synapse_id)
    U_synapse *= (1.0 + 0.5 * distance_to_soma / max_distance)

self.stp_modules[key] = ShortTermPlasticity(
    n_pre=n_input,
    n_post=n_neurons,
    config=get_stp_config_heterogeneous(U_distribution),
    per_synapse=True,  # Already supported!
)
```

**Benefits**:
- ✅ Matches experimental heterogeneity
- ✅ Richer temporal filtering per synapse
- ✅ Already use per_synapse=True, just need parameter sampling

**Option B: Calcium-Dependent STP** (Most Accurate)
```python
# Extend ShortTermPlasticity with calcium variable
class CalciumSTP(ShortTermPlasticity):
    def __init__(self, ...):
        super().__init__(...)
        self.residual_ca = torch.zeros(n_pre, n_post)
        self.tau_ca = 100.0  # ms
        self.alpha_ca = 0.3  # Ca facilitation factor

    def forward(self, pre_spikes):
        # Standard Tsodyks-Markram
        efficacy = self.u * self.x

        # Calcium-modulated dynamics
        ca_boost = 1.0 + self.alpha_ca * self.residual_ca
        u_jump = self.U * (1 - self.u) * ca_boost
        self.u += pre_spikes.unsqueeze(-1) * u_jump

        # Calcium accumulation and decay
        self.residual_ca += pre_spikes.unsqueeze(-1) * 0.1
        self.residual_ca *= exp(-dt / self.tau_ca)

        return efficacy
```

**Benefits**:
- ✅ Captures short-term facilitation buildup
- ✅ Explains paired-pulse facilitation/depression
- ✅ Biologically grounded mechanism
- ⚠️ Adds computational cost (extra state variable)

---

### 3. **Synaptic Integration** ⚠️ **Simplified**

#### Current Implementation
```python
# Linear summation of currents
d1_current = sum([weights[source] @ spikes[source] for source in inputs])

# ConductanceLIF integration
d1_spikes = d1_neurons(g_exc=d1_current, g_inh=fsi_inhibition)
```

#### Issues
1. **No dendritic compartments**: MSNs have extensive dendrites with nonlinear integration
2. **Linear summation**: Biological dendrites have **NMDA spikes**, **calcium spikes**, **dendritic plateau potentials**
3. **Missing NMDA receptors**: Corticostriatal synapses are 80% NMDA, 20% AMPA

#### Biological Reality

**MSN Dendritic Architecture**:
- **Spiny dendrites**: 5-7 primary dendrites, branching 3-5 times
- **Spine density**: 1-2 spines per μm (10,000-20,000 total spines)
- **Dendritic length**: 200-500 μm from soma
- **Compartmentalization**: Proximal vs distal synapses integrate differently

**Nonlinear Dendritic Integration**:
```python
# Simplified two-compartment model
proximal_input = sum(weights_proximal @ spikes)  # Direct to soma
distal_input = sum(weights_distal @ spikes)      # Attenuated

# Dendritic spike threshold
if distal_input > theta_dendrite:
    dendritic_spike = True
    effective_distal = distal_input * 3.0  # Amplification
else:
    effective_distal = distal_input * 0.3  # Attenuation

total_current = proximal_input + effective_distal
```

**NMDA Receptor Dynamics**:
```python
# NMDA has voltage-dependent Mg²⁺ block
g_nmda = weights @ spikes * nmda_unblock(V_membrane)

def nmda_unblock(V):
    """Mg²⁺ unblock function (Jahr & Stevens 1990)"""
    return 1.0 / (1.0 + 0.28 * exp(-0.062 * V))

# Total excitatory current
g_exc = 0.2 * g_ampa + 0.8 * g_nmda  # 80% NMDA
```

#### Recommendations

**Option A: Two-Compartment Model** (Moderate Impact)
```python
class TwoCompartmentMSN(nn.Module):
    """MSN with soma + distal dendrite compartments."""

    def __init__(self, n_neurons):
        self.soma = ConductanceLIF(n_neurons)
        self.dendrite = ConductanceLIF(n_neurons, theta=5.0)  # Higher threshold
        self.coupling_conductance = 0.1  # Weak coupling

    def forward(self, proximal_input, distal_input):
        # Dendritic spike generation
        dend_spikes, V_dend = self.dendrite(distal_input)

        # Dendritic amplification if spike
        dend_contribution = torch.where(
            dend_spikes > 0,
            distal_input * 3.0,  # Amplified
            distal_input * 0.3,  # Attenuated
        )

        # Soma integration
        total_current = proximal_input + dend_contribution
        soma_spikes, V_soma = self.soma(total_current)

        return soma_spikes, V_soma, V_dend
```

**Benefits**:
- ✅ Captures proximal/distal asymmetry
- ✅ Models dendritic spike amplification
- ✅ More realistic integration
- ⚠️ Requires splitting inputs into proximal/distal

**Option B: NMDA Receptor Model** (High Biological Fidelity)
```python
# In _integrate_multi_source_inputs()
ampa_current = weights @ spikes  # Fast, voltage-independent
nmda_current = (weights @ spikes) * self._nmda_unblock(V_membrane)

# NMDA is 80% of corticostriatal transmission
g_exc = 0.2 * ampa_current + 0.8 * nmda_current

# Requires tracking membrane voltage per neuron
def _nmda_unblock(self, V):
    """Mg²⁺ unblock (Jahr & Stevens 1990)"""
    return 1.0 / (1.0 + 0.28 * torch.exp(-0.062 * V))
```

**Benefits**:
- ✅ Biologically accurate receptor kinetics
- ✅ Voltage-dependent coincidence detection
- ✅ Explains burst firing preference
- ⚠️ Requires per-neuron voltage tracking

---

### 4. **Eligibility Trace Dynamics** ⚠️ **Good, Can Be Enhanced**

#### Current Implementation
```python
# Exponential decay with source-specific tau
eligibility = eligibility * exp(-dt / tau_ms) + outer(post, pre) * lr

# Source-specific taus
cortex: tau = 1000ms
hippocampus: tau = 300ms
thalamus: tau = 500ms
```

#### Issues
1. **Simple exponential decay**: Biological tags have **multi-timescale** dynamics
2. **No protein synthesis dependence**: Long-term tags require **protein synthesis** (1-2 hours)
3. **Missing tag consolidation**: Strong stimulation creates **stronger, longer-lasting tags**

#### Biological Reality

**Synaptic Tagging** (Frey & Morris 1997, Yagishita et al. 2014):
- **Early phase** (1-30 min): CaMKII activation, immediate tagging
- **Late phase** (30 min - 2 hours): Protein synthesis-dependent consolidation
- **Tag strength**: Proportional to postsynaptic [Ca²⁺] during activity

**Multi-Timescale Traces**:
```python
# Fast trace: immediate eligibility
eligibility_fast = eligibility_fast * exp(-dt / 200ms) + spike_coincidence

# Slow trace: consolidated tags (protein synthesis)
eligibility_slow = eligibility_slow * exp(-dt / 3600000ms) + eligibility_fast * consolidation_rate

# Total eligibility for learning
eligibility_total = eligibility_fast + 0.5 * eligibility_slow
```

**Calcium-Dependent Tag Strength**:
```
Weak stimulation (1 spike pair): Tag lasts ~200ms
Strong stimulation (10 spike pairs): Tag lasts ~2000ms
Very strong (tetanus): Tag lasts hours (protein synthesis)
```

#### Recommendations

**Option A: Multi-Timescale Eligibility** (High Impact)
```python
# In _update_pathway_eligibility()
# Fast trace (immediate, decays in ~500ms)
fast_decay = exp(-dt / 500.0)
self._eligibility_fast[key] = (
    self._eligibility_fast[key] * fast_decay +
    torch.outer(post_spikes, pre_spikes) * lr
)

# Slow trace (consolidated, decays in ~60 seconds)
slow_decay = exp(-dt / 60000.0)
consolidation_rate = 0.01  # 1% of fast trace consolidates per timestep
self._eligibility_slow[key] = (
    self._eligibility_slow[key] * slow_decay +
    self._eligibility_fast[key] * consolidation_rate
)

# Combined eligibility for dopamine gating
eligibility_dict[key] = self._eligibility_fast[key] + 0.3 * self._eligibility_slow[key]
```

**Benefits**:
- ✅ Supports multi-second credit assignment
- ✅ Explains delayed reward learning (>5 seconds)
- ✅ Models tag consolidation
- ⚠️ Doubles eligibility memory (fast + slow traces)

**Option B: Calcium-Dependent Tag Strength** (Most Accurate)
```python
# Track postsynaptic calcium per neuron
self.calcium = torch.zeros(n_neurons, device=device)

# In forward pass
self.calcium = self.calcium * exp(-dt / 50.0) + post_spikes * 0.5

# In eligibility update
tag_strength = torch.clamp(self.calcium / 1.0, 0.1, 3.0)  # 0.1x to 3x
eligibility_update = torch.outer(post_spikes, pre_spikes) * tag_strength.unsqueeze(-1)

# Stronger calcium → stronger, longer-lasting tags
calcium_dependent_tau = base_tau * (1.0 + 2.0 * self.calcium)
decay = exp(-dt / calcium_dependent_tau)
```

**Benefits**:
- ✅ Biologically grounded mechanism
- ✅ Explains tetanus-induced LTP
- ✅ Models activity-dependent tag consolidation
- ⚠️ Requires per-neuron calcium state

---

## Implementation Priority

### **Phase 1: Quick Wins** (1-2 weeks)
1. **Heterogeneous STP** (Option A)
   - Sample U from distribution per synapse
   - Location-dependent parameters
   - **Impact**: Medium biological accuracy, low complexity

2. **Multi-Timescale Eligibility** (Option A)
   - Add fast + slow traces
   - **Impact**: High (enables multi-second credit assignment)

### **Phase 2: High-Fidelity Enhancements** (2-4 weeks)
3. **Per-Synapse Delays via AxonalProjection**
   - Use existing CircularDelayBuffer infrastructure
   - **Impact**: High biological accuracy, moderate complexity

4. **NMDA Receptor Model** (Option B)
   - Voltage-dependent Mg²⁺ unblock
   - **Impact**: High (explains burst preference, coincidence detection)

### **Phase 3: Advanced Features** (1-2 months)
5. **Two-Compartment MSNs**
   - Soma + dendritic compartments
   - **Impact**: Medium (captures spatial integration)

6. **Calcium-Dependent STP & Eligibility**
   - Residual [Ca²⁺] modulation
   - **Impact**: High (unified mechanism, most biologically accurate)

---

## References

### Core Biology
- Frank, M.J. (2005). Dynamic dopamine modulation in the basal ganglia. *TRENDS in Cognitive Sciences* 9(2), 60-67.
- Kravitz, A.V. et al. (2010). Regulation of parkinsonian motor behaviours by optogenetic control of basal ganglia circuitry. *Nature* 466, 622-626.
- Yagishita, S. et al. (2014). A critical time window for dopamine actions on the structural plasticity of dendritic spines. *Science* 345(6204), 1616-1620.

### Synaptic Timing
- Markram, H. et al. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science* 275(5297), 213-215.

### STP Heterogeneity
- Ding, J. et al. (2008). Corticostriatal and thalamostriatal synapses have distinctive properties. *Journal of Neuroscience* 28(25), 6483-6492.
- Ellender, T.J. et al. (2013). Heterogeneous properties of central lateral and parafascicular thalamic synapses in the striatum. *Journal of Physiology* 591(1), 257-272.

### Dendritic Integration
- Carter, A.G. & Sabatini, B.L. (2004). State-dependent calcium signaling in dendritic spines of striatal medium spiny neurons. *Neuron* 44(3), 483-493.
- Day, M. et al. (2008). Selective elimination of glutamatergic synapses on striatopallidal neurons in Parkinson disease models. *Nature Neuroscience* 11, 1067-1069.

### NMDA Receptors
- Jahr, C.E. & Stevens, C.F. (1990). Voltage dependence of NMDA-activated macroscopic conductances predicted by single-channel kinetics. *Journal of Neuroscience* 10(9), 3178-3182.

### Synaptic Tagging
- Frey, U. & Morris, R.G. (1997). Synaptic tagging and long-term potentiation. *Nature* 385, 533-536.
- Redondo, R.L. & Morris, R.G. (2011). Making memories last: the synaptic tagging and capture hypothesis. *Nature Reviews Neuroscience* 12, 17-30.

---

## Conclusion

The current striatum implementation is **biologically sound at the systems level** (opponent pathways, dopamine modulation, three-factor learning). The recommended enhancements target **circuit and synaptic realism**:

1. **Temporal delays**: Move from pathway-level to per-synapse delays (AxonalProjection already supports this!)
2. **STP dynamics**: Add heterogeneity and calcium dependence
3. **Synaptic integration**: Add NMDA receptors and dendritic compartments
4. **Eligibility traces**: Add multi-timescale dynamics and calcium-dependent consolidation

**All enhancements maintain the core architecture** while increasing biological fidelity at finer scales.
