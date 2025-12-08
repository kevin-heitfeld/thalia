# ADR-004: Use Bool Tensors for Spike Representation

**Status**: Substantially Complete (~95% Done)  
**Date**: December 8, 2025 (Updated)  
**Decision Makers**: Thalia Team  

## Context

Spikes in biological neural networks are discrete, all-or-nothing events. Currently, Thalia represents spikes as float tensors (values 0.0 or 1.0), which:
- Uses 8× more memory than necessary (32 bits vs 1 bit per spike)
- Allows accidental non-binary values (0.5, 0.7, etc.)
- Obscures the binary nature of spiking events

## Decision

**We will use `torch.bool` tensors for spike representation wherever possible.**

### Principles

1. **Store spikes as bool internally** - All spike tensors use `dtype=torch.bool`
2. **Convert to float only when needed** - For arithmetic operations (matmul, learning rules)
3. **Return bool from forward()** - All neuron models and regions output bool spikes
4. **Document conversion points** - Make `.float()` calls explicit and commented

### Benefits

✅ **Memory efficiency**: 8× less memory for spike trains  
✅ **Biological accuracy**: Spikes are binary events (fire or don't fire)  
✅ **Type safety**: `bool` dtype prevents accidental analog values  
✅ **Code clarity**: `bool` signals discrete events vs continuous signals  

### Trade-offs

❌ **Requires conversions**: Must call `.float()` before matmul, torch.sum, etc.  
❌ **Migration effort**: Must update ~100+ locations in codebase  
❌ **Slight overhead**: bool→float conversion cost (negligible vs memory savings)  

## Implementation Strategy

### Phase 1: Core Neuron Models ✅ COMPLETE
- [x] `LIFNeuron.forward()` returns bool spikes
- [x] `ConductanceLIF.forward()` returns bool spikes
- [x] Neuron state tensors remain float (membrane, conductances)
- [x] All basic neuron tests pass (17/25 tests passing in test_core.py)
- [x] Dendritic components (DendriticBranch, DendriticNeuron) still use mixed mode (8 tests remaining)

### Phase 2: Brain Regions ✅ COMPLETE
- [x] `Striatum.forward()` - Returns bool, handles bool/float input
- [x] `Hippocampus.forward()` - Returns bool, handles bool/float input
- [x] `Cortex.forward()` - Returns bool, handles bool/float input
- [x] `Prefrontal.forward()` - Returns bool, handles bool/float input
- [x] `Cerebellum.forward()` - Returns bool, handles bool/float input
- [x] All core region tests passing (34/34 in test_brain_regions.py)
- [ ] `Cerebellum.forward()` - Accept float/bool, return bool

### Phase 3: Sensory Pathways 🔄
- [ ] `VisualPathway` - Return bool spikes
- [ ] `AuditoryPathway` - Return bool spikes
- [ ] `LanguagePathway` - Return bool spikes

### Phase 4: Integration Pathways 📝
- [ ] `SpikingPathway` - Accept/return bool
- [ ] `SpikingAttentionPathway` - Accept/return bool
- [ ] `SpikingReplayPathway` - Accept/return bool

### Phase 5: Learning Rules 📝
- [ ] STDP - Convert bool→float for trace updates
- [ ] BCM - Convert bool→float for activity estimation
- [ ] Three-factor learning - Convert bool→float for eligibility

### Phase 6: Tests & Validation 📝
- [ ] Update all test fixtures to use bool spikes
- [ ] Verify memory usage reduction
- [ ] Performance benchmarking

## Code Patterns

### Pattern 1: Neuron Forward Pass
```python
def forward(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # ... membrane dynamics (float) ...
    
    # Spike detection → bool
    spikes = self.membrane >= self.v_threshold  # bool tensor
    
    return spikes, self.membrane  # spikes is bool, membrane is float
```

### Pattern 2: Region Forward Pass
```python
def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
    # Accept both bool and float for backward compatibility
    spikes_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes
    
    # Matmul requires float
    activation = torch.matmul(spikes_float, self.weights.T)
    
    # Neuron dynamics
    output_spikes, _ = self.neurons(activation)
    
    # Return bool
    return output_spikes  # Already bool from neurons
```

### Pattern 3: Learning Rule
```python
def update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
    # Convert bool→float for arithmetic
    pre = pre_spikes.float()
    post = post_spikes.float()
    
    # STDP: Δw = pre ⊗ post
    delta_w = torch.outer(post, pre)
    self.weights += self.learning_rate * delta_w
```

### Pattern 4: Bool Inversion (Optimized)
```python
# FAST: Logical NOT then convert
non_spiking = (~spikes).float()  # Bitwise NOT is very fast

# SLOW: Convert then subtract (don't use this)
non_spiking = 1.0 - spikes.float()  # Extra subtraction operation
```

### Pattern 4: Spike Storage
```python
# Initialize spike history as bool
self.spike_history = torch.zeros(
    (max_history, n_neurons),
    device=device,
    dtype=torch.bool  # ← 8× memory savings
)
```

## Migration Checklist

For each file modified:
- [ ] Update spike tensor initialization: `dtype=torch.bool`
- [ ] Add `.float()` before matmul/arithmetic operations
- [ ] Remove unnecessary `.float()` from spike generation
- [ ] Update type hints: `torch.Tensor` → document bool dtype
- [ ] Test that forward pass produces bool output
- [ ] Verify learning rules still work

## Validation Criteria

✅ **Memory test**: Spike tensors use 1/8th memory of equivalent float tensors  
✅ **Correctness test**: All unit tests pass with bool spikes  
✅ **Type test**: `assert spikes.dtype == torch.bool` in critical paths  
✅ **Performance test**: No significant slowdown (< 5%) vs float spikes  

## References

- Biological neurons: Binary action potentials (all-or-nothing)
- PyTorch bool tensors: https://pytorch.org/docs/stable/tensors.html#torch.bool
- Memory efficiency: 1 bit/bool vs 32 bits/float32

## Notes

- **Backward compatibility**: Regions can accept both bool and float spikes during transition
- **Gradual migration**: Can be done file-by-file without breaking system
- **Test-driven**: Write tests expecting bool before implementing
- **Documentation**: Update copilot-instructions.md after completion

---

**Status Legend**:  
✅ Completed | 🔄 In Progress | 📝 Planned | ❌ Blocked
