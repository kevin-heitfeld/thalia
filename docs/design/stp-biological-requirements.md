# Short-Term Plasticity (STP) - Biological Requirements by Region

**Date**: December 21, 2025
**Status**: Design Decision
**Purpose**: Document biological justification for STP across brain regions

---

## Executive Summary

Short-term plasticity (STP) modulates synaptic strength on fast timescales (10ms-1s) based on recent presynaptic activity. This document summarizes the biological evidence for STP across Thalia brain regions and provides configuration recommendations.

**Current Implementation Status**:
- ✅ **Hippocampus**: Fully implemented (4 pathways with correct types)
- ✅ **Cortex**: L2/3 recurrent depression implemented
- ✅ **Prefrontal**: Recurrent depression implemented
- ⚠️ **Cerebellum**: NOT implemented (HIGH PRIORITY)
- ⚠️ **Thalamus**: NOT implemented (HIGH PRIORITY)
- ⚠️ **Striatum**: NOT implemented (MODERATE PRIORITY)

---

## Region-by-Region Analysis

### 1. HIPPOCAMPUS ✅ IMPLEMENTED

**Biological Documentation**: Most extensively studied STP in the brain

**Pathways**:
| Pathway | Type | U (release prob) | Functional Role |
|---------|------|------------------|-----------------|
| Mossy Fibers (DG→CA3) | **FACILITATING** | 0.03 | Pattern separation via burst amplification (10×!) |
| Schaffer Collaterals (CA3→CA1) | DEPRESSING | 0.5 | Novelty detection, prevents CA3 dominance |
| EC→CA1 (direct) | DEPRESSING | 0.35 | Initial stimulus strongest for comparison |
| CA3 recurrent | **DEPRESSING_FAST** | 0.6 | Prevents frozen attractors, enables pattern transitions |

**Key References**:
- Salin et al. (1996): PNAS 93:13304-13309 - Mossy fiber U=0.03
- Dobrunz & Stevens (1997): Schaffer collateral STP
- Nicoll & Schmitz (2005): Synaptic plasticity at hippocampal mossy fibre synapses

**Implementation**: `src/thalia/regions/hippocampus/trisynaptic.py:251-310`

**Default Setting**: `stp_enabled: bool = True` ✅ (as of Dec 21, 2025)

**Biological Justification**:
- Mossy fiber facilitation is **CRITICAL** - first spike barely transmits (3% efficacy), but repeated DG activity causes massive facilitation (up to 10× enhancement). This implements pattern separation by amplifying only strong/repeated DG bursts.
- CA3 recurrent depression is **CRITICAL** - without it, the same neurons fire every timestep due to self-reinforcement. Depression allows pattern transitions.

---

### 2. CORTEX ✅ IMPLEMENTED

**Biological Documentation**: Well-characterized pathway-specific STP

**Pathways**:
| Pathway | Type | U | Functional Role |
|---------|------|---|-----------------|
| L4→L2/3 | Weak facilitation | 0.2 | Temporal filtering, change detection |
| L2/3↔L2/3 recurrent | **DEPRESSING** | 0.6 | Gain control, prevents runaway excitation |
| L5→L2/3 feedback | Mild depression | 0.3 | Top-down modulation with adaptation |
| Pyramidal→Interneuron | Very strong depression | 0.7 | Fast FFI with rapid fatigue |

**Key References**:
- Tsodyks & Markram (1997): PNAS 94:719-723 - Foundational STP model
- Markram et al. (1998): PNAS 95:5323-5328 - Differential signaling
- Reyes et al. (1998): Developmental Science

**Implementation**: `src/thalia/regions/cortex/layered_cortex.py:381,861`

**Default Setting**: Always enabled (hardcoded for L2/3 recurrent)

**Biological Justification**:
- L2/3 recurrent depression prevents the same neurons from dominating activity
- Enables divisive normalization and gain control
- Critical for temporal dynamics and working memory flexibility

---

### 3. PREFRONTAL CORTEX ✅ IMPLEMENTED

**Biological Documentation**: Essential for working memory dynamics

**Pathways**:
| Pathway | Type | U | Functional Role |
|---------|------|---|-----------------|
| PFC↔PFC recurrent | **DEPRESSING** | 0.15-0.2 | Working memory updating, prevents frozen attractors |
| PFC→Striatum | Mild depression | 0.35 | Phasic command signals for action selection |

**Key References**:
- Wang et al. (2013): Working memory networks with STP
- Compte et al. (2000): Attractor dynamics
- Haber et al. (2000): PFC→striatum projections

**Implementation**: `src/thalia/regions/prefrontal.py:457-463`

**Default Setting**: `stp_recurrent_enabled: bool = True` ✅

**Biological Justification**:
- Without STP, working memory patterns persist indefinitely (frozen attractors)
- Depression enables updating: active synapses weaken, allowing new patterns to take over
- Critical for cognitive flexibility

---

### 4. CEREBELLUM ⚠️ NOT IMPLEMENTED (HIGH PRIORITY)

**Biological Documentation**: **Strongest evidence in the brain** for STP

**Pathways**:
| Pathway | Type | U | Functional Role |
|---------|------|---|-----------------|
| **Parallel fibers→Purkinje** | **DEPRESSING** | 0.5-0.7 | **CRITICAL**: Temporal high-pass filter for timing |
| Mossy fibers→Granule cells | FACILITATING | 0.15-0.25 | Burst detection for sparse coding |
| Climbing fibers→Purkinje | Reliable | 0.9 | Error signal must be unambiguous |

**Key References**:
- **Dittman et al. (2000)**: Nature 403:530-534 - Classic PF→Purkinje STP paper
- Atluri & Regehr (1996): Delayed release at granule cell synapses
- Isope & Barbour (2002): Facilitation at mossy fiber synapses

**RECOMMENDED Configuration**:
```python
# Add to CerebellumConfig
stp_enabled: bool = True  # ENABLE BY DEFAULT
stp_pf_purkinje_type: STPType = STPType.DEPRESSING  # U=0.5-0.7
stp_mf_granule_type: STPType = STPType.FACILITATING  # U=0.2
# Climbing fiber: no STP (reliable)
```

**Biological Justification**:
- **Parallel fiber depression is perhaps THE MOST important STP in the brain**
- Implements temporal high-pass filter: fresh inputs signal new patterns, sustained inputs fade
- Without it, cerebellar timing precision collapses
- Enables sub-millisecond timing discrimination critical for motor control
- Mossy fiber facilitation enables burst detection for sparse coding expansion layer

**Priority**: **HIGHEST** - This is a critical missing feature

---

### 5. THALAMUS ⚠️ NOT IMPLEMENTED (HIGH PRIORITY)

**Biological Documentation**: Essential for sensory gating

**Pathways**:
| Pathway | Type | U | Functional Role |
|---------|------|---|-----------------|
| Sensory input→Relay neurons | DEPRESSING | 0.3-0.5 | Change detection, prevents saturation |
| Cortex L6→Thalamus | **DEPRESSING** | 0.6-0.8 | "Searchlight" attention control |

**Key References**:
- Sherman & Guillery (2002): The role of the thalamus in cortical function
- Castro-Alamancos (2002): Different forms of synaptic plasticity
- Swadlow & Gusev (2001): Thalamic relay cell firing

**RECOMMENDED Configuration**:
```python
# Add to ThalamicRelayConfig
stp_enabled: bool = True  # ENABLE BY DEFAULT
stp_sensory_relay_type: STPType = STPType.DEPRESSING  # U=0.4
stp_l6_feedback_type: STPType = STPType.DEPRESSING_FAST  # U=0.7
```

**Biological Justification**:
- Sensory adaptation: prevents response saturation to sustained stimuli
- Change detection: fresh sensory inputs get preferential relay
- L6→thalamus depression implements attentional modulation ("searchlight")
- Critical for sensory gating and selective attention

**Priority**: **HIGH** - Essential for realistic sensory processing

---

### 6. STRIATUM ⚠️ NOT IMPLEMENTED (MODERATE PRIORITY)

**Biological Documentation**: Well-characterized

**Pathways**:
| Pathway | Type | U | Functional Role |
|---------|------|---|-----------------|
| Cortex→MSNs | DEPRESSING | 0.4 | Context-dependent filtering |
| Thalamus→MSNs | Weak facilitation | 0.25 | Phasic input amplification |

**Key References**:
- Charpier et al. (1999): Corticostriatal EPSPs
- Partridge et al. (2000): Synaptic plasticity in striatum
- Ding et al. (2008): Thalamostriatal facilitation

**RECOMMENDED Configuration**:
```python
# Add to StriatumConfig
stp_enabled: bool = True  # ENABLE BY DEFAULT
stp_corticostriatal_type: STPType = STPType.DEPRESSING  # U=0.4
stp_thalamostriatal_type: STPType = STPType.FACILITATING  # U=0.25
```

**Biological Justification**:
- Depression prevents sustained cortical input from saturating striatum
- Novelty detection: fresh inputs get stronger transmission
- Balances phasic (thalamus) and tonic (cortex) command signals
- Critical for action selection dynamics

**Priority**: **MODERATE** - Dopaminergic modulation may partially compensate, but STP is still important

---

## Implementation Recommendations

### Immediate (Before Phase 0)
1. ✅ **Enable hippocampus STP by default** (DONE Dec 21, 2025)
2. Verify cortex and prefrontal implementations are correctly enabled

### High Priority (Phase 0 or alongside)
1. **Cerebellum**: Add STP configuration and implementation
   - PF→Purkinje depression (CRITICAL for timing)
   - MF→Granule facilitation
2. **Thalamus**: Add STP configuration and implementation
   - Sensory relay depression
   - L6 feedback depression

### Medium Priority (Phase 1-2)
1. **Striatum**: Add STP configuration and implementation
   - Corticostriatal depression
   - Thalamostriatal facilitation

### State Management Impact

Each region with STP needs state serialization:
```python
@dataclass
class RegionState:
    # For each STP module, add:
    stp_<pathway>_u: Optional[torch.Tensor] = None  # Release probability
    stp_<pathway>_x: Optional[torch.Tensor] = None  # Available resources
```

**Example** (CerebellumState):
```python
@dataclass
class CerebellumState:
    # ... existing fields ...

    # STP state (when stp_enabled=True)
    stp_pf_purkinje_u: Optional[torch.Tensor] = None
    stp_pf_purkinje_x: Optional[torch.Tensor] = None
    stp_mf_granule_u: Optional[torch.Tensor] = None
    stp_mf_granule_x: Optional[torch.Tensor] = None
```

---

## Summary Table

| Region | Status | Default | Priority | Biological Evidence |
|--------|--------|---------|----------|-------------------|
| **Hippocampus** | ✅ Implemented | ON | - | Strongest (4 pathways) |
| **Cortex** | ✅ Implemented | ON | - | Strong (multiple pathways) |
| **Prefrontal** | ✅ Implemented | ON | - | Strong (WM dynamics) |
| **Cerebellum** | ⚠️ Missing | **SHOULD BE ON** | **HIGHEST** | **Strongest in brain** (PF depression critical) |
| **Thalamus** | ⚠️ Missing | **SHOULD BE ON** | **HIGH** | Strong (sensory gating) |
| **Striatum** | ⚠️ Missing | **SHOULD BE ON** | MODERATE | Moderate (action selection) |

---

## References

### Key Papers
1. **Tsodyks & Markram (1997)**: The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. PNAS 94:719-723
2. **Markram et al. (1998)**: Differential signaling via the same axon of neocortical pyramidal neurons. PNAS 95:5323-5328
3. **Dittman et al. (2000)**: Interplay between facilitation, depression, and residual calcium at three presynaptic terminals. Nature 403:530-534
4. **Sherman & Guillery (2002)**: The role of the thalamus in the flow of information to the cortex. Philosophical Transactions B 357:1695-1708
5. **Abbott & Regehr (2004)**: Synaptic computation. Nature 431:796-803

### Reviews
- **Zucker & Regehr (2002)**: Short-term synaptic plasticity. Annual Review of Physiology 64:355-405
- **Fioravante & Regehr (2011)**: Short-term forms of presynaptic plasticity. Current Opinion in Neurobiology 21:269-274

---

**Document Status**: FINALIZED
**Last Updated**: December 21, 2025
**Next Steps**: Implement STP for cerebellum, thalamus, striatum
**Related Documents**:
- `state-management-refactoring-plan.md` - STP state serialization
- `docs/api/LEARNING_STRATEGIES_API.md` - STP API reference
