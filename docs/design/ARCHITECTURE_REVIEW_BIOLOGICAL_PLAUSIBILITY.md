# Architecture Review: Biological Plausibility Issues in Planning Documents

**Date**: December 10, 2025
**Reviewer**: Thalia Architecture Team
**Status**: ✅ Phase 1 COMPLETE - Phase 2/3 remain to be implemented
**Last Updated**: December 10, 2025 (Phase 1 Week 2-3 completion)

---

## Summary of Issues and Resolutions

All three phase documents originally contained components that violated Thalia's core design principle:

> **NO BACKPROPAGATION. All learning must use local rules (STDP, Hebbian, error-corrective).**

### Implementation Status

| Phase | Component | Original Problem | Resolution Status |
|-------|-----------|------------------|-------------------|
| **Phase 1** | `GoalConditionedValueNetwork` | Used `nn.Module` + backprop | ✅ **FIXED & IMPLEMENTED** |
| **Phase 1** | TD(λ) Multi-Step Credit | N/A (always correct) | ✅ **IMPLEMENTED** |
| **Phase 2** | `WorldModel` | Used `nn.Module` + Adam + backprop | ✅ **DELETED & REDESIGNED** |
| **Phase 3** | `HyperbolicDiscounter` | ✅ No backprop (just math) | ✅ **OK (not yet implemented)** |
| **Phase 3** | `GoalHierarchyManager` | ✅ Data structure only | ✅ **OK (not yet implemented)** |

---

## Detailed Analysis

### Phase 1: Goal-Conditioned Values ✅ IMPLEMENTED

**Original Design** (lines 452-600 in PHASE1_TD_LAMBDA.md - NOW REMOVED):
```python
class GoalConditionedValueNetwork(nn.Module):
    """Neural network for goal-conditioned value function."""
    
    def __init__(...):
        self.goal_embedding = nn.Embedding(...)  # ❌ Learned via backprop
        self.state_encoder = nn.Linear(...)      # ❌ Trained via backprop
        self.goal_encoder = nn.Linear(...)       # ❌ Trained via backprop
        self.value_head = nn.Sequential(...)     # ❌ Backprop
```

**Problem**:
- Uses PyTorch `nn.Module` with gradient-based optimization
- Requires backpropagation through goal encoder
- Not biologically plausible - no neuron backpropagates errors

**Correct Approach**:
Goal-conditioned values should emerge from **existing striatum** mechanisms:

```python
# CORRECT: Extend existing striatum
class Striatum:
    def __init__(self, config):
        # EXISTING: Action values (model-free)
        self.action_values = torch.zeros(n_actions)
        
        # NEW: Goal-conditioned values (still local learning!)
        # Use separate value estimates per goal context
        self.goal_conditioned_values = torch.zeros(n_goals, n_actions)
        
        # Learn via existing three-factor rule:
        # Δw = eligibility × dopamine × goal_context
        # where goal_context comes from PFC working memory
    
    def get_value(self, action: int, goal_context: torch.Tensor):
        """Get value conditioned on goal from PFC."""
        # Goal context from PFC modulates which values to use
        # Uses gating (like PFC does), not backprop!
        weighted_values = self.goal_conditioned_values * goal_context
        return weighted_values[:, action].sum()
```

**Key Insight**: 
- PFC already maintains goal representations in working memory
- Striatum can condition on PFC via **gating/modulation**, not backprop
- This is how biology does it (PFC→striatum dopamine gating)

**✅ IMPLEMENTATION STATUS**:

The correct architecture has been fully implemented:

```python
# ✅ IMPLEMENTED: PFC goal context output
class PrefrontalCortex:
    def get_goal_context(self) -> torch.Tensor:
        """Get goal context for striatum modulation.
        Returns working memory as goal representation.
        """
        return self.state.working_memory

# ✅ IMPLEMENTED: Striatum goal modulation weights
class Striatum:
    def __init__(self, config):
        # ...
        if config.use_goal_conditioning:
            self.pfc_modulation_d1 = nn.Parameter(
                WeightInitializer.sparse_random(...)
            )
            self.pfc_modulation_d2 = nn.Parameter(
                WeightInitializer.sparse_random(...)
            )
    
    # ✅ IMPLEMENTED: Goal-conditioned forward pass
    def forward(self, input_spikes, pfc_goal_context=None):
        # Compute goal modulation via learned weights
        goal_mod_d1 = torch.sigmoid(
            self.pfc_modulation_d1 @ pfc_goal_context
        )
        goal_mod_d2 = torch.sigmoid(
            self.pfc_modulation_d2 @ pfc_goal_context
        )
        
        # Modulate D1/D2 activations
        d1_activation *= (1.0 + strength * (goal_mod_d1 - 0.5))
        d2_activation *= (1.0 + strength * (goal_mod_d2 - 0.5))
    
    # ✅ IMPLEMENTED: Goal-conditioned learning
    def _deliver_reward_d1_d2(self, da_level):
        # Extended three-factor rule: Δw = eligibility × dopamine × goal_context
        goal_weight_d1 = torch.sigmoid(
            self.pfc_modulation_d1 @ goal_context
        )
        d1_dw = d1_dw * goal_weight_d1.unsqueeze(1)
        
        # Hebbian learning of PFC modulation weights (local!)
        d1_hebbian = torch.outer(
            self._last_d1_spikes.float(),
            goal_context
        ) * da_level * pfc_lr
        self.pfc_modulation_d1.data += d1_hebbian
```

**Testing**: ✅ 15 unit tests passing
- PFC goal context output
- Striatum goal modulation weights
- Goal-conditioned forward pass
- Goal-conditioned learning
- PFC modulation weight updates (Hebbian)
- Goal-biased action selection

**Files Modified**:
- `src/thalia/regions/prefrontal.py` - Added `get_goal_context()` method
- `src/thalia/regions/striatum/config.py` - Added goal conditioning config
- `src/thalia/regions/striatum/striatum.py` - Added goal modulation and learning
- `tests/unit/test_goal_conditioned_values.py` - 15 comprehensive tests
- `docs/design/PHASE1_TD_LAMBDA.md` - Updated documentation

---

### Phase 1 Key Achievement: Biologically Plausible Goal-Conditioned Learning

**No backpropagation anywhere!** All learning uses:
1. **Three-factor rule** (striatal learning): Δw = eligibility × dopamine × goal_context
2. **Hebbian rule** (PFC modulation): Δw = outer(post_spikes, pre_spikes) × dopamine
3. **Local computation**: Every update uses only locally available information

**Key Differences from Original Proposal**:
- ❌ No `nn.Module` with backprop
- ❌ No separate "goal embedding" layer trained with gradients
- ❌ No global error propagation
- ✅ Uses existing PFC working memory for goal context
- ✅ Uses gating/modulation (biologically plausible)
- ✅ All learning is local (Hebbian, three-factor)

---

### Phase 2: World Model ✅ (FIXED)

**Original Design** (DELETED):
```python
class WorldModel(nn.Module):  # ❌ WRONG
    def __init__(...):
        self.encoder = nn.Sequential(...)  # ❌ Backprop
        self.state_predictor = nn.Linear(...)  # ❌ Backprop
        self.optimizer = torch.optim.Adam(...)  # ❌ Adam optimizer
```

**Corrected Design** (Updated in planning/__init__.py):
```
Model-based planning emerges from:
  1. PFC predictive coding (ALREADY EXISTS) → state predictions
  2. Hippocampus pattern completion (ALREADY EXISTS) → outcome predictions
  3. Striatum values (ALREADY EXISTS) → evaluate simulated states
  4. Coordination layer orchestrates the above
```

**Status**: ✅ Fixed - no separate WorldModel module needed

---

### Phase 3: Hierarchical Goals ✅

**Goal Hierarchy** (lines 100-350 in PHASE3_HIERARCHICAL.md):
```python
class GoalHierarchyManager:
    """Data structure for goal management."""
    # ✅ NO LEARNING - just data structure
    # ✅ NO BACKPROP - pure coordination
    # ✅ BIOLOGICALLY PLAUSIBLE
```

**Hyperbolic Discounting** (lines 394-500):
```python
class HyperbolicDiscounter:
    def discount(self, reward, delay):
        return reward / (1 + k * delay)  # ✅ Just math, no backprop
```

**Status**: ✅ These components are fine as-is

---

## Corrected Architecture

### Phase 1: Goal-Conditioned Values (Revised)

**Principle**: Goals are context from PFC that **gate** striatal values

```python
# In PFC (already exists!)
class PrefrontalCortex:
    def __init__(...):
        self.working_memory = torch.zeros(n_neurons)  # ✅ Holds goal
    
    def forward(self, input_spikes):
        # Working memory maintains goal representation
        # This is the "goal context" for striatum

# In Striatum (extend existing!)
class Striatum:
    def __init__(self, config):
        # Existing: Basic action values
        self.d1_weights = ...  # Go pathway
        self.d2_weights = ...  # No-go pathway
        
        # NEW: Goal-modulated connections
        self.pfc_modulation_weights = WeightInitializer.sparse_random(
            n_output=self.n_actions,
            n_input=pfc_size,
            sparsity=0.3,
            device=device,
        )
    
    def forward(self, input_spikes, pfc_goal_context):
        """Forward pass with goal context from PFC."""
        
        # Standard striatal processing
        d1_activity = self.d1_weights @ input_spikes
        d2_activity = self.d2_weights @ input_spikes
        
        # Goal modulation from PFC (gating, not backprop!)
        goal_modulation = torch.sigmoid(
            self.pfc_modulation_weights @ pfc_goal_context
        )
        
        # Modulate action selection by goal
        d1_activity = d1_activity * goal_modulation
        d2_activity = d2_activity * (1 - goal_modulation)
        
        return self.compete(d1_activity, d2_activity)
    
    def learn(self, ..., pfc_goal_context):
        """Three-factor learning with goal context."""
        
        # Existing: Δw = eligibility × dopamine
        # NEW: Δw = eligibility × dopamine × goal_context
        
        # Update striatal weights (local learning!)
        ltp = eligibility * dopamine * (pfc_goal_context > 0.5).float()
        self.d1_weights += self.config.learning_rate * ltp
        
        # Update PFC modulation weights (also local!)
        # Uses Hebbian: If PFC active + striatum active + reward
        pfc_modulation_update = torch.outer(
            self.output_spikes.float(),
            pfc_goal_context
        ) * dopamine
        self.pfc_modulation_weights += 0.001 * pfc_modulation_update
```

**Key Differences from Original**:
- ❌ No `nn.Module` with backprop
- ❌ No separate "goal embedding" layer
- ✅ Uses existing PFC working memory for goal context
- ✅ Uses gating/modulation (biologically plausible)
- ✅ All learning is local (Hebbian, three-factor)

---

### Phase 2: Model-Based Planning (Revised)

**Principle**: Planning = coordination of existing regions

```python
# In Brain (coordination layer)
class EventDrivenBrain:
    def simulate_rollout(
        self,
        current_state: torch.Tensor,
        actions: List[int],
        depth: int = 5,
    ) -> Rollout:
        """Mental simulation using existing regions."""
        
        # 1. PFC holds simulated state in working memory
        simulated_state = current_state
        self.pfc.set_context(simulated_state)
        
        rollout_values = []
        for step in range(depth):
            for action in actions:
                # 2. Hippocampus predicts next state (pattern completion)
                similar_experiences = self.hippocampus.retrieve_similar(
                    state=simulated_state,
                    action=action,
                    k=5,  # Top 5 similar experiences
                )
                
                # 3. PFC predictive coding generates prediction
                predicted_next_state = self.cortex.predict_next_state(
                    current=simulated_state,
                    action=action,
                    similar_experiences=similar_experiences,
                )
                
                # 4. Striatum evaluates simulated state
                value = self.striatum.evaluate_state(predicted_next_state)
                
                rollout_values.append((action, value))
                
            # Choose best action, update simulated state
            best_action = max(rollout_values, key=lambda x: x[1])[0]
            simulated_state = predicted_next_state
        
        return Rollout(states=..., actions=..., values=rollout_values)
```

**Key Points**:
- Uses PFC predictive coding (already exists!)
- Uses hippocampus pattern completion (already exists!)
- Uses striatum values (already exists!)
- No separate "world model" needed

---

## Action Items

### Phase 1 (Goal-Conditioned Values) ✅ COMPLETE

1. ✅ **Delete GoalConditionedValueNetwork** - DONE (never created)
2. ✅ **Implement PFC-Striatum gating** - DONE
3. ✅ **Add PFC.get_goal_context() method** - DONE
4. ✅ **Add goal modulation weights to Striatum** - DONE
5. ✅ **Extend three-factor learning with goal context** - DONE
6. ✅ **Write comprehensive tests** - DONE (15 tests passing)
7. ✅ **Update Phase 1 documentation** - DONE

### Phase 2 (Model-Based Planning) ⏳ PENDING

1. ✅ **Delete WorldModel** - DONE
2. ✅ **Update Phase 2 docs architecture** - DONE  
3. ⏳ **Implement PFC.simulate_future()** - TODO
4. ⏳ **Implement Hippocampus.predict_outcome()** - TODO
5. ⏳ **Implement Striatum.evaluate_state()** - TODO
6. ⏳ **Implement Brain.simulate_rollout()** - TODO
7. ⏳ **Write coordination tests** - TODO

### Phase 3 (Hierarchical Goals) ⏳ NOT STARTED

**File**: `docs/design/PHASE2_MODEL_BASED.md`

**Changes**:
- ✅ Already updated architecture overview
- Lines 100-400: Rewrite "Week 4" section (no WorldModel)
- Lines 400-600: Rewrite "Week 5" section (coordination, not separate module)
- Add new focus: Coordination utilities, not separate models

### Phase 3 Status

**File**: `docs/design/PHASE3_HIERARCHICAL.md`

**Status**: ✅ No changes needed
- Goal hierarchy is data structures (OK)
- Hyperbolic discounting is pure math (OK)
- Options learning would need review (check for backprop)

---

## Design Principles (Reinforced)

### ✅ DO

1. **Use existing regions**: PFC, hippocampus, striatum, cortex, cerebellum
2. **Local learning only**: STDP, Hebbian, BCM, three-factor rule, error-corrective
3. **Spike-based**: Binary spikes (0 or 1), not firing rates
4. **Neuromodulation**: Dopamine/ACh gates learning, doesn't carry gradients
5. **Coordination**: New capabilities emerge from region interactions

### ❌ DON'T

1. **No backpropagation**: Never use `.backward()` or autograd
2. **No nn.Module for learning**: Only use for structure (weights, layers) with manual updates
3. **No optimizers**: No Adam, SGD, RMSprop - use manual weight updates
4. **No global signals**: All learning must be local (neuron can't see distant errors)
5. **No separate "AI modules"**: Don't bolt on ANNs to SNN brain

---

## Questions for Discussion

1. **Goal-conditioned values**: Should we implement PFC-striatum gating now, or defer until Phase 2 when we have better PFC-region coordination?

2. **Hindsight Experience Replay**: Can HER work with local learning? The original paper uses backprop. May need to redesign as "goal relabeling" during hippocampal replay.

3. **Options learning**: Phase 3 proposes learning reusable policies. How do we learn these without backprop? Likely answer: Cache successful action sequences in hippocampus, retrieve via pattern completion.

4. **Timeline impact**: These architectural changes might extend Phase 1 timeline. Estimate: +1 week for proper PFC-striatum integration.

---

## References

**Biological Plausibility**:
- Lillicrap et al. (2020): "Backpropagation and the brain" - Why backprop is implausible
- Richards & Lillicrap (2019): "Dendritic solutions to the credit assignment problem"
- Whittington & Bogacz (2019): "Theories of error back-propagation in the brain"

**PFC-Striatum Gating**:
- O'Reilly & Frank (2006): "Making working memory work" - Gating architecture
- Collins & Frank (2013): "Cognitive control over learning"
- Chatham & Badre (2015): "Working memory management and predicted common neural mechanisms"

**Goal-Conditioned RL (without backprop)**:
- Schaul et al. (2015): "Universal Value Function Approximators" - Goal conditioning
- Andrychowicz et al. (2017): "Hindsight Experience Replay" - Original HER
- Note: These papers use backprop, we need to adapt to local learning!

---

**Current Status**: 
- ✅ **Phase 1 (TD(λ) + Goal-Conditioned Values)**: COMPLETE & TESTED
  - 19 TD(λ) tests passing
  - 15 goal-conditioned value tests passing
  - All implementations biologically plausible
  - Documentation updated
  
- ⏳ **Phase 2 (Model-Based Planning)**: Architecture corrected, implementation pending
  - WorldModel deleted
  - Emergent planning architecture documented
  - Ready for implementation
  
- ⏳ **Phase 3 (Hierarchical Goals)**: Architecture approved, not started
  - Goal hierarchy: data structures only (OK)
  - Hyperbolic discounting: pure math (OK)
  - Options learning: needs local learning approach

**Next Steps**: Proceed with Phase 2 implementation (Model-Based Planning via region coordination)

**Approval**: ✅ Phase 1 architecture validated and implemented correctly
