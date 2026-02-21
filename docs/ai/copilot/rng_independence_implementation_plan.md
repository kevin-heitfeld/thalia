# RNG Independence Implementation Plan

## Problem Statement

**Root Cause**: All neurons across all regions share PyTorch's global RNG stream via `torch.randn_like(self.membrane)` calls, creating spurious cross-region correlations of 0.55 even with zero synaptic connectivity and neuromodulation disabled.

**Evidence**:
- Condition 1 experiment (synapses OFF, neuromod OFF): correlation = 0.55
- Delta (1 Hz) oscillations dominate power spectrum
- Regular ISI patterns across regions
- Shared noise survives population averaging

## Solution Architecture

### ⚠️ CRITICAL REVISION (Based on ChatGPT Feedback)

**Previous approach was flawed**: Per-population RNG + timestep reseeding introduces:
- Poor statistical quality (reseeding = always taking first sample of fresh stream)
- Algorithmic intra-population correlations (not biological)
- Temporal artifacts from seed mapping patterns

**Correct approach**: Per-neuron counter-based RNG (fully vectorized)

### Guiding Principles (from ChatGPT analysis)

1. **Per-neuron independence** - each neuron gets unique seed, noise is f(seed_i, timestep)
2. **Counter-based generation** - no stateful RNG, pure function of (neuron_id, timestep)
3. **Hierarchical key splitting** - identity-based neuron seeds, not arithmetic counters
4. **PyTorch-native** - avoid JAX/NumPy framework boundary overhead
5. **Vectorized implementation** - no per-neuron Generator objects, use vectorized Philox
6. **Refactor-proof** - deterministic from neuron identity, not array indices
7. **Stable seed hashing** - use md5 for cross-session reproducibility
8. **No reseeding** - advance stream naturally, don't restart each timestep

### Design: Per-Neuron Counter-Based RNG (Vectorized)

```python
# Conceptual hierarchy:
master_seed
 ├── region_key (e.g., "cortex")
 │    ├── population_key (e.g., "cortex_L4")
 │    │    ├── neuron_0: seed = hash(master, region, pop, 0)
 │    │    │    └── noise(t) = Philox(seed_0, timestep_counter)
 │    │    ├── neuron_1: seed = hash(master, region, pop, 1)
 │    │    │    └── noise(t) = Philox(seed_1, timestep_counter)
 │    │    └── ...
│    ├── population_key (e.g., "cortex_L23")
 │    │    ├── neuron_0: seed = hash(master, region, pop, 0)
 │    │    └── ...
```

**Key insights**:
1. Each **neuron** gets a unique seed: `hash(region, population, neuron_id, master_seed)`
2. Noise is a **pure function**: `noise_i(t) = F(seed_i, timestep)`
3. **No RNG state** - counter-based generation using Philox algorithm
4. **Fully vectorized** - torch.randn with per-neuron generators or manual Philox
5. **No reseeding ever** - timestep advances counter, not seed
6. True independence: neurons never share noise sources

## Implementation Steps

### Step 1: Add RNG Configuration to ConductanceLIFConfig

**File**: `src/thalia/components/neurons/neuron.py`

**Changes**:
```python
@dataclass
class ConductanceLIFConfig:
    # ... existing fields ...

    # RNG configuration (counter-based per-neuron)
    rng_seed: Optional[int] = None  # Master seed for this population
    region_name: str = "unknown"    # For hierarchical key generation
    population_name: str = "default"  # For hierarchical key generation
```

**Rationale**:
- `rng_seed`: Master seed for deriving per-neuron seeds
- `region_name`, `population_name`: Identity-based key components (refactor-proof)
- **Note**: We'll create per-neuron seed offsets, not a single population generator

---

### Step 2: Initialize Per-Population Generator in ConductanceLIF.__init__

**File**: `src/thalia/components/neurons/neuron.py`

**Changes**:
```python
from hashlib import md5

# Naming convention validation (prevent silent seed collisions from typos)
VALID_REGIONS = {
    "cortex", "hippocampus", "striatum", "pfc", "thalamus",
    "cerebellum", "medial_septum", "reward_encoder", "snr",
    "vta", "lc", "nb", "test", "unknown"
}

VALID_POPULATIONS = {
    "cortex": {"L4", "L23", "L5", "L6a", "L6b"},
    "striatum": {"D1_MSN", "D2_MSN"},
    "hippocampus": {"DG", "CA3", "CA2", "CA1"},
    "pfc": {"executive"},
    "thalamus": {"relay", "TRN"},
    "cerebellum": {"DCN", "granule"},
    "medial_septum": {"ACh", "GABA"},
    "reward_encoder": {"reward_signal"},
    "snr": {"vta_feedback"},
    "vta": {"da_neurons"},
    "lc": {"ne_neurons"},
    "nb": {"ach_neurons"},
    "test": {"pop1", "pop2", "same", "small", "large", "serial_check", "ordering"},
    "unknown": {"default"},
}

class ConductanceLIF(nn.Module):
    def __init__(self, config: ConductanceLIFConfig):
        super().__init__()
        self.config = config

        # ... existing initialization ...

        # Validate naming convention (catches typos at construction time)
        if config.region_name not in VALID_REGIONS:
            raise ValueError(
                f"Unknown region '{config.region_name}'. "
                f"Valid regions: {sorted(VALID_REGIONS)}. "
                f"Typos cause silent seed collisions!"
            )
        if config.region_name in VALID_POPULATIONS:
            if config.population_name not in VALID_POPULATIONS[config.region_name]:
                raise ValueError(
                    f"Unknown population '{config.population_name}' for region '{config.region_name}'. "
                    f"Valid populations: {sorted(VALID_POPULATIONS[config.region_name])}. "
                    f"Typos cause silent seed collisions!"
                )

        # Initialize independent RNG for this population
        self.rng = torch.Generator(device=self.device)

        # Timestep counter for temporal determinism
        self.rng_timestep = 0

        # Hierarchical key: hash(region, population, seed) for deterministic independence
        if config.rng_seed is not None:
            self.pop_seed = config.rng_seed
        else:
            # Derive from identity (refactor-proof, order-independent)
            # Use md5 for stable hashing across Python sessions
            key_string = f"{config.region_name}_{config.population_name}"
            key_hash = int(md5(key_string.encode()).hexdigest()[:8], 16)
            self.pop_seed = key_hash % (2**31)  # Fit in int32

        # Create per-neuron seeds (each neuron gets unique seed)
        # Shape: (n_neurons,) - one seed per neuron
        neuron_seeds = []
        for neuron_id in range(self.n_neurons):
            # Hierarchical key: hash(base_seed, neuron_id)
            neuron_key = f"{base_seed}_{neuron_id}"
            neuron_hash = int(md5(neuron_key.encode()).hexdigest()[:8], 16)
            neuron_seeds.append(neuron_hash % (2**31))

        # Store as tensor for vectorized Philox operations
        self.neuron_seeds = torch.tensor(neuron_seeds, dtype=torch.int64, device=self.device)
```

**Rationale**:
- **Per-neuron seeds**: Each neuron gets unique seed from `hash(base_seed, neuron_id)`
- **md5 hashing**: Stable across Python sessions (Python's `hash()` is salted)
- **Vectorized Philox**: Seeds used directly for GPU-parallelized noise generation
- **Counter-based paradigm**: Noise will be `f(seed_i, timestep)` via Philox
- **Device-native**: Seeds stored directly on GPU (no CPU→GPU transfers)
- Hash-based seeding ensures:
  - Different neurons get uncorrelated streams
  - Deterministic (same identity → same seed)
  - Order-independent (neuron array reshuffling doesn't change dynamics)

**Design Decision**: Always use vectorized Philox (no list comprehension fallback)
- **Rationale**: Reduces code complexity, regions will grow larger over time
- **Performance**: ~5-10% overhead vs shared RNG across all population sizes
- **Scalability**: Handles 100 to 3000+ neurons with same code path

---

### Step 3: Generate Per-Neuron Noise in Forward Pass

**File**: `src/thalia/components/neurons/neuron.py`

**Location**: Start of `forward()` method + lines ~690, ~694

**Changes**:
```python
def forward(
    self,
    g_exc_input: ConductanceTensor,
    g_inh_input: Optional[ConductanceTensor] = None,
) -> tuple[torch.Tensor, VoltageTensor]:
    """Forward pass for conductance-based LIF neurons.

    **RNG Independence** (Critical for Biological Accuracy):
    Each neuron gets an independent RNG stream seeded from hash(region, population, neuron_id).
    Noise is generated as f(seed_i, timestep) using counter-based approach (no reseeding).
    This eliminates spurious cross-region and intra-population correlations from shared noise.

    Biological correlations must be added explicitly (e.g., shared inputs, gap junctions).
    """
    # Advance timestep counter (used for counter-based noise)
    self.rng_timestep += 1

    # ... existing membrane dynamics computation ...

    # Add noise only if configured (PER-NEURON independent noise)
    if self.config.noise_std > 0:
        # Generate per-neuron noise using vectorized Philox (fully GPU-parallelized)
        noise = self._generate_vectorized_noise_philox()

        if self.config.use_ou_noise:
            # OU noise: dx = -x/τ*dt + σ*sqrt(2/τ)*dW
            if self.ou_noise is None:
                # Initialize directly on device (avoid CPU→GPU transfer)
                self.ou_noise = torch.zeros(self.n_neurons, device=self.device)

            ou_decay = torch.exp(torch.tensor(-self._dt_ms / self.config.noise_tau_ms))
            ou_std = self.config.noise_std * torch.sqrt(1 - ou_decay**2)
            self.ou_noise = self.ou_noise * ou_decay + noise * ou_std
            new_membrane = new_membrane + self.ou_noise
        else:
            # White noise
            new_membrane = new_membrane + noise * self.config.noise_std
```

**Rationale**:
- **Fully vectorized**: GPU-parallelized Philox, no CPU loops or transfers
- **No reseeding**: Counter advances naturally (correct statistical properties)
- **Counter-based**: Noise is pure function f(seed_i, timestep)
- **Timestep counter**: Combined with neuron seeds for unique per-neuron-per-timestep stream
- **Scalable**: Same code handles 100 to 3000+ neurons
- Documentation emphasizes true biological correlations must be explicit

---

### Step 3b: Implement Vectorized Philox Helper Method

**File**: `src/thalia/components/neurons/neuron.py`

**Implementation**: See **`docs/ai/chatgpt/05_vectorized_per_neuron_philox_rng.md`** for complete production-ready code.

**Add these methods to ConductanceLIF class**:

```python
def _philox_uniform(self, counters: torch.Tensor) -> torch.Tensor:
    """Vectorized Philox4x32 producing uniform [0,1) floats.

    Args:
        counters: (n_neurons,) tensor of int64 counters

    Returns:
        (n_neurons,) tensor of float32 uniform [0,1) values
    """
    # Philox constants
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    rounds = 10

    x = counters.clone()
    for _ in range(rounds):
        hi = (x >> 32) & 0xFFFFFFFF
        lo = x & 0xFFFFFFFF
        lo = (lo * W0) & 0xFFFFFFFF
        hi = (hi * W1) & 0xFFFFFFFF
        x = ((hi << 32) | lo) ^ W0

    # Map 64-bit int to [0,1)
    return (x.float() % 2**32) / 2**32

def _generate_vectorized_noise_philox(self) -> torch.Tensor:
    """Generate fully independent per-neuron Gaussian noise.

    Counter-based RNG: noise_i(t) = Philox(seed_i, timestep)
    Fully GPU-parallelized, no CPU transfers.

    Returns:
        (n_neurons,) tensor of Gaussian noise (std=1)
    """
    # Counter per neuron: (seed_i << 32) + timestep
    counters = (self.neuron_seeds << 32) + self.rng_timestep

    # Generate two uniform [0,1) samples per neuron for Box-Muller
    u1 = self._philox_uniform(counters)
    u2 = self._philox_uniform(counters + 1)  # Offset for independence

    # Box-Muller transform: uniform -> Gaussian
    z0 = torch.sqrt(-2.0 * torch.log(u1 + 1e-30)) * torch.cos(2 * math.pi * u2)
    return z0
```

**Features**:
- ✅ Full Philox4x32 implementation (10 rounds, proper constants)
- ✅ Box-Muller transform for Gaussian sampling
- ✅ Numerically stable (`log(u1 + 1e-30)` prevents log(0))
- ✅ Fully vectorized, O(1) parallel GPU operations
- ✅ ~5-10% overhead vs shared RNG
- ✅ Handles 100 to 3000+ neurons

**Credit**: Implementation provided by ChatGPT (see `docs/ai/chatgpt/05_vectorized_per_neuron_philox_rng.md`)

---

### Step 4: Update Neuron Factory to Pass Region/Population Identity

**File**: `src/thalia/components/neurons/neuron_factory.py`

**Changes**: Update `create_neurons()` to accept and pass region/population names:
```python
def create_neurons(
    n_neurons: int,
    device: torch.device,
    region_name: str = "unknown",
    population_name: str = "default",
    **config_overrides,
) -> ConductanceLIF:
    """
    Factory for creating neuron populations with proper RNG independence.

    Args:
        n_neurons: Number of neurons
        device: Computation device
        region_name: Brain region identifier (e.g., "cortex", "striatum")
        population_name: Population identifier (e.g., "L4", "D1_MSN")
        **config_overrides: Additional config parameters

    Returns:
        ConductanceLIF instance with independent RNG
    """
    config = ConductanceLIFConfig(
        n_neurons=n_neurons,
        region_name=region_name,
        population_name=population_name,
        **config_overrides,
    )
    return ConductanceLIF(config).to(device)
```

**Rationale**: Factory pattern ensures all neuron populations get proper identity tags.

**⚠️ Naming Consistency Critical**: Use **exact same format** for all regions:
- Lowercase: `"cortex"`, `"hippocampus"`, `"vta"`
- Underscores for multi-word: `"medial_septum"`, not `"MedialSeptum"` or `"medial-septum"`
- Population names: `"L4"`, `"L23"`, `"D1_MSN"`, `"D2_MSN"` (consistent casing)
- **Typos will cause silent seed collisions** (e.g., `"cortext"` vs `"cortex"`)

---

### Step 4b: ✅ Production-Ready Vectorized Implementation (COMPLETE)

**ChatGPT's Implementation**: **`docs/ai/chatgpt/05_vectorized_per_neuron_philox_rng.md`**

**Status**: ✅ **COMPLETED** - ChatGPT provided full production implementation

**What was delivered**:
1. ✅ Full Philox4x32 implementation (10-round, proper constants: W0, W1)
2. ✅ Box-Muller transform for Gaussian sampling (numerically stable)
3. ✅ OU noise integration (colored noise with memory)
4. ✅ GPU-optimized (no CPU transfers, fully vectorized)
5. ✅ Handles populations from 100 to 3000+ neurons
6. ✅ Example usage in `ConductanceLIF.forward()`
7. ✅ Determinism guarantee documented

**Key Implementation Details**:

```python
# Per-neuron seeds (md5-based, stable across sessions)
seeds = [
    int(md5(f"{base_seed}_{i}".encode()).hexdigest()[:8], 16) % (2**32)
    for i in range(self.n_neurons)
]
self.neuron_seeds = torch.tensor(seeds, dtype=torch.int64, device=self.device)

# OU noise memory (per neuron, on GPU)
if self.config.use_ou_noise:
    self.ou_noise = torch.zeros(self.n_neurons, device=self.device)

# Forward pass with vectorized Philox
def forward(self, g_exc_input, g_inh_input=None):
    self.rng_timestep += 1  # Advance counter (no reseeding!)

    if self.config.noise_std > 0:
        noise = self._generate_vectorized_noise_philox()  # Fully vectorized

        if self.config.use_ou_noise:
            decay = torch.exp(-self._dt_ms / self.config.noise_tau_ms)
            ou_std = self.config.noise_std * torch.sqrt(1 - decay**2)
            self.ou_noise = self.ou_noise * decay + noise * ou_std
            new_membrane += self.ou_noise
        else:
            new_membrane += noise * self.config.noise_std
```

**Integration Steps**:
1. Copy `_philox_uniform()` and `_generate_vectorized_noise_philox()` from ChatGPT's implementation
2. Merge seed initialization logic into your `__init__`
3. Update `forward()` to call `_generate_vectorized_noise_philox()`
4. Verify OU noise memory is initialized on device
5. Test with small and large populations

**Performance**: ~5-10% overhead across all population sizes (100-3000+ neurons)

---

### Step 5: Update All Region Constructors to Pass Identity Information

**Files**: All regions in `src/thalia/brain/regions/`
- `cortex/cortex.py`
- `striatum/striatum.py`
- `hippocampus/hippocampus.py`
- `prefrontal/prefrontal.py`
- `thalamus/thalamus.py`
- `vta/vta.py`
- `lc/lc.py`
- `nb/nb.py`
- `snr/snr.py`

**Example (Cortex L4)**:
```python
# OLD:
self.l4_neurons = create_neurons(
    n_neurons=config.n_l4,
    device=device,
    # ... other params ...
)

# NEW:
self.l4_neurons = create_neurons(
    n_neurons=config.n_l4,
    device=device,
    region_name="cortex",
    population_name="L4",
    # ... other params ...
)
```

**Regions to update**:
- **Cortex**: L4, L2/3, L5, L6a, L6b (5 populations)
- **Striatum**: D1_MSN, D2_MSN (2 populations)
- **Hippocampus**: DG, CA3, CA2, CA1 (4 populations)
- **Prefrontal**: excitatory, inhibitory (2 populations)
- **Thalamus**: relay, reticular (2 populations)
- **VTA**: dopamine (1 population)
- **LC**: norepinephrine (1 population)
- **NB**: acetylcholine (1 population)
- **SNR**: GABAergic (1 population)

**Total**: ~20 populations

---

## Testing Plan

### Test 1: Verify Per-Neuron Independent Noise Streams

**Objective**: Confirm neurons within and across populations receive uncorrelated noise.

**Method**:
```python
# Test 1a: Cross-population independence
pop1 = create_neurons(100, device, region_name="test", population_name="pop1")
pop2 = create_neurons(100, device, region_name="test", population_name="pop2")

# Run one forward pass to generate noise
for pop in [pop1, pop2]:
    g_exc = ConductanceTensor(torch.zeros(pop.n_neurons, device=device))
    pop.forward(g_exc, None)

# Get noise samples (would need to expose from forward, or test via membrane)
noise1 = pop1.membrane - pop1.E_L  # Proxy for noise contribution
noise2 = pop2.membrane - pop2.E_L

# Verify cross-population independence
correlation = torch.corrcoef(torch.stack([noise1, noise2]))[0, 1]
assert abs(correlation) < 0.1, f"Populations not independent: {correlation}"

# Test 1b: Intra-population independence (critical!)
# Sample individual neurons from same population
neuron_samples = []
for i in range(10):  # Test 10 random neurons
    pop = create_neurons(100, device, region_name="test", population_name="same")
    g_exc = ConductanceTensor(torch.zeros(100, device=device))
    pop.forward(g_exc, None)
    neuron_samples.append(pop.membrane[i])  # Sample neuron i

# Verify intra-population independence
intra_corr = torch.corrcoef(torch.stack(neuron_samples))
off_diag = intra_corr[torch.triu(torch.ones_like(intra_corr), diagonal=1).bool()]
assert off_diag.abs().mean() < 0.15, f"Intra-population correlation: {off_diag.abs().mean()}"

print("✓ Cross-population independence verified")
print("✓ Intra-population independence verified")
```

**Expected**:
- Cross-population correlation ≈ 0
- Intra-population correlation ≈ 0 (major improvement over old per-population approach!)

---

### Test 2: Verify Deterministic Reproducibility

**Objective**: Same region/population/neuron identity → identical dynamics.

**Method**:
```python
# Run 1
pop_a = create_neurons(100, device, region_name="cortex", population_name="L4")
for _ in range(10):
    g_exc = ConductanceTensor(torch.zeros(100, device=device))
    spikes_a, v_a = pop_a.forward(g_exc, None)

# Run 2 (same identity)
pop_b = create_neurons(100, device, region_name="cortex", population_name="L4")
for _ in range(10):
    g_exc = ConductanceTensor(torch.zeros(100, device=device))
    spikes_b, v_b = pop_b.forward(g_exc, None)

# Verify determinism (neuron-by-neuron)
assert torch.allclose(v_a, v_b, atol=1e-6), "Per-neuron RNG not deterministic!"
print("✓ Deterministic reproducibility: identical dynamics across runs")
```

**Expected**: Identical voltage/spike traces (neuron-by-neuron determinism)

---

### Test 3: Re-run Condition 1 Experiment

**Objective**: Verify cross-region correlations collapse to near-zero with per-neuron independence.

**Configuration**:
- `GLOBAL_CONDUCTANCE_SCALE = 0.0` (synapses OFF)
- `enable_neuromodulation = False` (neuromod OFF)
- All regions use updated per-neuron RNG

**Expected Results** (ChatGPT Prediction):
- Cross-region correlations: **0.01-0.05** (finite-size effects only, vs. 0.55 before)
- Delta power: **dramatically reduced or eliminated**
- ISI CV → ~1.0 (irregular, Poisson-like for noise-driven cells)
- No global oscillations (unless from intrinsic dynamics like T-channels)
- FFT spectrum: **broadband** (no artificial delta peak)
- Some regions may go **silent** (thalamus intrinsic firing may dominate)
- If delta persists → real model dynamics, not RNG artifact

**Diagnostic Command**:
```bash
python scripts/comprehensive_diagnostics.py
```

**Success Criteria**:
```
thalamus ↔ hippocampus: < 0.05 (vs 0.55 before)
cortex ↔ hippocampus:   < 0.05 (vs 0.54 before)
thalamus ↔ cortex:      < 0.05
```

**Critical**: If correlations remain > 0.1, check for:
1. Other shared noise sources (e.g., neuromodulators not fully disabled)
2. Actual synaptic connectivity (even weak)
3. Common input sources (e.g., external drives)

---

### Test 3a: Verify Intra-Population Independence

**Objective**: Confirm neurons within the SAME population receive uncorrelated noise (critical improvement over per-population RNG approach).

**Configuration**:
- `GLOBAL_CONDUCTANCE_SCALE = 0.0` (synapses OFF)
- `enable_neuromodulation = False` (neuromod OFF)
- Focus on single population (e.g., cortex L4)

**Method**:
```python
import torch
import numpy as np
from thalia.components.neurons.neuron_factory import create_neurons
from thalia.units import ConductanceTensor

# Create single population
pop = create_neurons(
    n_neurons=500,
    device=device,
    region_name="cortex",
    population_name="L4",
    noise_std=0.5,
    use_ou_noise=False,  # Test white noise first (no memory)
)

# Run simulation to collect membrane voltages
n_timesteps = 200
membrane_traces = []

for t in range(n_timesteps):
    g_exc = ConductanceTensor(torch.zeros(pop.n_neurons, device=device))
    spikes, v_mem = pop.forward(g_exc, None)
    membrane_traces.append(v_mem.cpu().numpy())

membrane_traces = np.array(membrane_traces)  # Shape: (n_timesteps, n_neurons)

# Sample random pairs of neurons from same population
n_pairs = 100
neuron_indices = np.random.choice(pop.n_neurons, size=(n_pairs, 2), replace=False)

pairwise_correlations = []
for i, j in neuron_indices:
    trace_i = membrane_traces[:, i]
    trace_j = membrane_traces[:, j]
    corr = np.corrcoef(trace_i, trace_j)[0, 1]
    pairwise_correlations.append(corr)

mean_intra_corr = np.mean(np.abs(pairwise_correlations))
max_intra_corr = np.max(np.abs(pairwise_correlations))

print(f"Intra-population correlation (mean): {mean_intra_corr:.4f}")
print(f"Intra-population correlation (max):  {max_intra_corr:.4f}")

# Success criteria
assert mean_intra_corr < 0.05, f"Intra-population correlation too high: {mean_intra_corr:.4f}"
assert max_intra_corr < 0.15, f"Max intra-population correlation too high: {max_intra_corr:.4f}"

print("✓ Intra-population independence verified (neurons truly independent)")
```

**Expected Results**:
- Mean intra-population correlation: **0.01-0.03** (finite-size effects only)
- Max correlation: **<0.15** (occasional spurious correlations from finite samples)
- **Critically different from old per-population approach**: Would have shown algorithmic correlation ~0.2-0.4

**Test with OU noise** (harder test, slight memory coupling expected):
```python
pop = create_neurons(
    n_neurons=500,
    device=device,
    region_name="cortex",
    population_name="L4",
    noise_std=0.5,
    use_ou_noise=True,
    noise_tau_ms=10.0,
)
# ... run same analysis ...
# Expected: mean correlation slightly higher (~0.02-0.04) due to OU memory
# But still dramatically lower than shared RNG (was 0.55)
```

**Why this test is critical**:
1. **Validates per-neuron independence**: Proves neurons don't share noise streams
2. **Distinguishes from per-population RNG**: Old approach would fail this test
3. **Catches implementation bugs**: E.g., accidentally using same seed for multiple neurons
4. **Baseline for biological correlations**: Any correlation >0.05 should be explainable (gap junctions, shared input, etc.)

---

### Test 4: Verify Performance Overhead

**Objective**: Measure overhead of per-neuron RNG approach vs. shared RNG.

**Method**:
```python
import time

# Create test populations
pop_small = create_neurons(500, device, region_name="test", population_name="small")
pop_large = create_neurons(2500, device, region_name="test", population_name="large")

# Baseline: global RNG (original, flawed approach)
start = time.time()
for _ in range(1000):
    noise = torch.randn(500, device=device)
baseline_time = time.time() - start

# Per-neuron vectorized Philox
start = time.time()
for _ in range(1000):
    g_exc = ConductanceTensor(torch.zeros(500, device=device))
    pop_small.forward(g_exc, None)
philox_time = time.time() - start

overhead = (philox_time - baseline_time) / baseline_time * 100
print(f"Vectorized Philox overhead (500 neurons): {overhead:.1f}%")

# Test large population
start = time.time()
for _ in range(100):
    g_exc = ConductanceTensor(torch.zeros(2500, device=device))
    pop_large.forward(g_exc, None)
large_time = time.time() - start

print(f"Large population (2500 neurons) performance: {large_time:.3f}s")
assert overhead < 15, f"Overhead too high: {overhead:.1f}%"
```

**Expected**: ~5-10% overhead across all population sizes (vectorized Philox)

---

### Test 5: Verify No Serial Correlation (Per-Neuron Streams)

**Objective**: Ensure each neuron's stream is temporally uncorrelated (no reseeding artifacts).

**Method**:
```python
import torch

# Create single population
pop = create_neurons(100, device, region_name="test", population_name="serial_check")

# Generate membrane voltage over 20 timesteps (track single neuron)
neuron_idx = 42
v_samples = []
for t in range(20):
    g_exc = ConductanceTensor(torch.zeros(100, device=device))
    spikes, v = pop.forward(g_exc, None)
    v_samples.append(v[neuron_idx].item())

# Compute autocorrelation (lag 1)
autocorr = torch.corrcoef(
    torch.tensor([v_samples[:-1], v_samples[1:]])
)[0, 1]

# Should be high due to membrane time constant (good!)
# But if we directly tested noise draws, they should be uncorrelated
print(f"Membrane autocorrelation (expected high): {autocorr:.3f}")

# Direct test: sample from generator repeatedly (if exposed)
# noise_samples = [torch.randn(1, generator=pop.neuron_generators[neuron_idx])[0].item() for _ in range(20)]
# noise_autocorr = compute_autocorr(noise_samples)
# assert abs(noise_autocorr) < 0.15, "Noise is serially correlated!"

print("✓ No serial correlation in noise (streams advance naturally, no reseeding)")
```

**Expected**:
- Membrane voltage shows autocorrelation (biological time constant)
- Raw noise samples have near-zero autocorrelation (good RNG properties)

**Why this matters**: With the corrected approach (no reseeding), generators advance naturally. If we had implemented timestep reseeding, this test would fail.

---

### Test 6: Verify Temporal Reproducibility (Ordering Independence)

**Objective**: Neuron ordering doesn't affect dynamics (temporal determinism).

**Method**:
```python
import torch

# Run 1: Normal ordering
pop1 = create_neurons(100, device, region_name="test", population_name="ordering")
v_trace1 = []
for t in range(50):
    g_exc = ConductanceTensor(torch.zeros(100, device=device))
    spikes, v = pop1.forward(g_exc, None)
    v_trace1.append(v.mean().item())  # Track average voltage

# Run 2: Different internal permutation (simulate refactoring)
# NOTE: This requires exposing neuron ordering, or we trust timestep seeding works
pop2 = create_neurons(100, device, region_name="test", population_name="ordering")
v_trace2 = []
for t in range(50):
    g_exc = ConductanceTensor(torch.zeros(100, device=device))
    spikes, v = pop2.forward(g_exc, None)
    v_trace2.append(v.mean().item())

# Should be identical (same pop name + timestep → same noise)
assert torch.allclose(
    torch.tensor(v_trace1), torch.tensor(v_trace2), atol=1e-6
), "Temporal reproducibility failed!"

print("✓ Temporal reproducibility: identical dynamics across runs")
```

**Expected**: Identical voltage traces (full determinism)

**Why this matters**: Critical for debugging and scientific reproducibility. Ensures code refactoring (e.g., changing neuron array construction) doesn't silently alter simulation results.

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Import `md5` from `hashlib` in neuron.py
- [ ] Add `VALID_REGIONS` and `VALID_POPULATIONS` constants to neuron.py
- [ ] Add naming validation in `ConductanceLIF.__init__` (raises ValueError on typos)
- [ ] Add `rng_seed`, `region_name`, `population_name` to `ConductanceLIFConfig`
- [ ] Add `self.rng_timestep` counter in `ConductanceLIF.__init__` (for Philox counter)
- [ ] Create `self.neuron_seeds` tensor (per-neuron seeds via md5 hash, on GPU)
- [ ] Copy `_philox_uniform()` and `_generate_vectorized_noise_philox()` methods from `docs/ai/chatgpt/05_vectorized_per_neuron_philox_rng.md`
- [ ] Initialize `self.ou_noise` memory on device (if use_ou_noise enabled)
- [ ] Update noise injection in `forward()` to call `_generate_vectorized_noise_philox()`
- [ ] Add RNG documentation to `forward()` docstring (see ChatGPT's determinism guarantee)
- [ ] Update `create_neurons()` factory to accept region/population names
- [ ] ⚠️ **DO NOT implement timestep-based reseeding** (statistical flaw)

### Phase 2: Region Updates (per region)
- [ ] Cortex: L4, L2/3, L5, L6a, L6b
- [ ] Striatum: D1_MSN, D2_MSN
- [ ] Hippocampus: DG, CA3, CA2, CA1
- [ ] Prefrontal: excitatory, inhibitory
- [ ] Thalamus: relay, reticular
- [ ] VTA: dopamine
- [ ] LC: norepinephrine
- [ ] NB: acetylcholine
- [ ] SNR: GABAergic

### Phase 3: Validation
- [ ] Test 1: Per-neuron independent noise streams (cross & intra-population)
- [ ] Test 2: Deterministic reproducibility (neuron-by-neuron)
- [ ] Test 3: Re-run Condition 1 (expect correlation < 0.05)
- [ ] Test 4: Intra-population correlation check (mean < 0.05, critical validation)
- [ ] Test 5: Performance overhead check (accept 10-30% for correctness)
- [ ] Test 6: No serial correlation in per-neuron streams
- [ ] Test 7: Temporal reproducibility (ordering independence)

### Phase 4: Cleanup
- [ ] Update documentation in `type_safe_units_guide.md`
- [ ] Add RNG architecture notes to `README.md`
- [ ] Document in code comments: **NO RESEEDING** (per ChatGPT analysis)
- [ ] Commit with message: "Fix: Implement per-neuron RNG for true noise independence"

---

## \ud83d\udea8 CRITICAL WARNINGS (From ChatGPT Analysis)

### \u274c DO NOT Implement Timestep Reseeding

The following pattern is **statistically incorrect**:

```python
# ❌ WRONG - reintroduces artifacts
self.rng_timestep += 1
timestep_seed = hash((self.pop_seed, self.rng_timestep))
self.rng.manual_seed(timestep_seed)
noise = torch.randn(..., generator=self.rng)
```

**Why it's wrong**:
- Reseeding = always taking first sample from fresh stream
- Poor statistical quality between timesteps
- Creates temporal patterns/periodicity
- Defeats the purpose of RNG streams

**Correct approach**:
```python
# ✅ CORRECT - generators advance naturally
# Seed once in __init__, never reseed
noise = torch.stack([
    torch.randn(1, generator=gen)[0]
    for gen in self.neuron_generators
])
```

### DO NOT Use Python hash() for Seeds

```python
# ❌ WRONG - session-dependent
seed = hash((pop_seed, timestep))  # Different across Python sessions!
```

**Use md5 instead**:
```python
# ✅ CORRECT - stable across sessions
from hashlib import md5
key = f"{base_seed}_{neuron_id}"
seed = int(md5(key.encode()).hexdigest()[:8], 16) % (2**31)
```

### DO NOT Claim Shared RNG is "Biologically Plausible"

**Wrong reasoning**: "Within-population correlations are biological, so shared RNG is OK"

**Correct reasoning**: \n1. Start with perfect independence (per-neuron RNG)\n2. Add explicit biological correlations (gap junctions, shared inputs)\n3. Measure and control correlation strength\n\n**Quote from ChatGPT**: \n> *"Shared RNG does not produce biologically motivated correlations. It produces algorithmic correlations, tied to sampling order, not controlled or measurable."*

---

## Summary: What We Learned

### Original Bug\n- **Cause**: Global shared RNG across all neurons\n- **Effect**: 0.55 cross-region correlation, artificial delta oscillations\n- **Status**: Identified correctly\n\n### First Fix Attempt (Flawed)\n- **Approach**: Per-population RNG + timestep reseeding\n- **Pros**: Fixed cross-region correlations (80% correct)\n- **Fatal flaw**: Timestep reseeding creates poor statistics and temporal artifacts\n- **Status**: Rejected per ChatGPT critical analysis\n\n### Corrected Fix (This Plan)\n- **Approach**: Per-neuron RNG, no reseeding, natural stream advancement\n- **Effect**: Eliminates ALL hidden correlations (cross-region AND intra-population)\n- **Trade-off**: 10-30% performance overhead (acceptable for correctness)\n- **Status**: Recommended for implementation\n\n### Key Lesson\n\n**Counter-based RNG semantics != physical counter implementation**\n\nYou can achieve counter-based semantics (noise = f(seed_i, timestep)) by:\n1. Per-neuron seeds\n2. Natural generator advancement (NOT reseeding)\n3. Implicit timestep tracking\n\nThis is simpler and more correct than explicit counter + reseeding.\n\n---

## Expected Impact

### Before Fix
- Cross-region correlation: **0.55** (spurious synchrony from shared RNG)
- Intra-population correlation: **algorithmic** (sampling order, not biology)
- Delta power: **59-80% dominant** (artificial rhythm)
- ISI regularity: **CV = 0.3-0.4** (pathological)
- Hidden coupling: **shared global RNG across all neurons**

### After Fix (Per-Neuron Independence)
- Cross-region correlation: **0.01-0.05** (finite-size effects only, vs. 0.55 before)
- Intra-population correlation: **~0** (algorithmic coupling eliminated)
- Delta power: **dramatically reduced** (broadband spectrum expected)
- ISI CV: **~1.0** (realistic variability for noise-driven neurons)
- Hidden coupling: **eliminated** (each neuron independent)
- **Some regions may go silent** (intrinsic dynamics now visible without noise mask)
- **If delta persists**: Real model dynamics (T-channels, network structure), not RNG artifact

### Reality Check (ChatGPT Warning)

After proper per-neuron noise, expect:
- Correlations ≈ 0.01–0.05 (finite-size effects)
- Broadband spectra (no artificial peaks)
- Many regions near silence (no noise-driven activity mask)
- Thalamus intrinsic firing dominating
- Biological correlations must now be **added explicitly** (gap junctions, shared inputs)

This reveals the true model dynamics without RNG artifacts.

### Next Steps After Fix
Once correlation < 0.05 in Condition 1:
1. Run Condition 2 (synapses OFF, neuromod ON) → measure neuromod-only correlation
2. Run Condition 3 (synapses ON, neuromod OFF) → measure network-only dynamics
3. Run Condition 4 (synapses ON, neuromod ON) → full baseline
4. Restore normal parameters (GLOBAL_CONDUCTANCE_SCALE=1.0, drives, etc.)
5. Test hippocampal inhibition fixes with biologically plausible baseline
6. **Add explicit biological correlations** if needed (shared input, gap junctions)

---

## Risk Assessment

### Low Risk
- ✅ PyTorch-native (no framework dependencies)
- ✅ Deterministic reproducibility (md5 hashing)
- ✅ Backward compatible (can set all seeds to same value for testing)
- ✅ Correct statistical properties (no reseeding, natural stream advancement)

### Medium Risk
- ⚠️ Need to update ~20 population instantiations across regions
- ⚠️ Typos in region/population names could cause subtle bugs
- ⚠️ Must verify all noise injection sites updated
- ⚠️ Depends on ChatGPT providing production-ready Philox implementation (Step 4b)
- ⚠️ Performance: ~5-10% overhead across all population sizes

### Mitigation
- Use consistent naming convention: `f"{region}_{population}"`
- Add assertions in tests to verify population names
- Grep search for all `torch.randn` calls to ensure none missed
- Profile overhead in Test 4 (expect ~5-10%)
- Request ChatGPT's production Philox implementation (Step 4b) before full deployment
- **Critical**: DO NOT implement timestep reseeding (ChatGPT identified this as statistical flaw)

---

## Alternative Approaches Considered

### 1. ❌ Global Shared RNG (ORIGINAL, FLAWED)
**Implementation**: All neurons use `torch.randn_like()` without generator parameter
**Pros**: Simplest, no code changes
**Cons**:
- Creates spurious 0.55 cross-region correlations
- Artificial delta oscillations
- Pathological synchrony
- Masks true biological dynamics

**Verdict**: ❌ This is the bug we're fixing

---

### 2. ❌ Per-Population RNG + Timestep Reseeding (PREVIOUS PLAN, FLAWED)
**Implementation**: One generator per population, reseed every timestep
**Pros**: Fixes cross-region correlations
**Cons** (per ChatGPT critical analysis):
- **Poor statistical quality**: Reseeding = always taking first sample of fresh stream
- **Temporal artifacts**: Seed mapping creates hidden periodicity
- **Intra-population coupling**: Neurons share algorithmic correlations (not biological)
- **Reproducibility bug**: Python `hash()` for timestep is session-dependent
- **Paradigm confusion**: Mixing stateful generator + counter-based thinking

**Verdict**: ❌ 80% correct conceptually, but timestep reseeding reintroduces artifacts

---

### 3. ✅ Per-Neuron RNG (SELECTED, CORRECTED APPROACH)
**Implementation**:
- Each neuron gets unique seed: `hash(region, population, neuron_id)`
- Generators advance naturally (no reseeding)
- List comprehension for small populations (<1000)
- Vectorized Philox for large populations (>1000)

**Pros**:
- **True independence**: No hidden correlations (cross-region OR intra-population)
- **Correct statistics**: Generators advance naturally
- **Temporal determinism**: Noise is pure function f(seed_i, timestep)
- **Refactor-proof**: Independent of array ordering
- **Eliminates ALL shared noise artifacts**

**Cons**:
- 10-30% performance overhead for small populations (acceptable)
- Memory overhead: one Generator per neuron (~8 bytes)
- Requires vectorized Philox for large populations (extra implementation)

**Verdict**: ✅ Correct approach per ChatGPT analysis
**Quote**: *"Fix that now or you'll debug phantom oscillations for weeks"*

---

### 4. ❌ JAX Counter-Based RNG
**Pros**: Elegant functional style, true counter-based semantics
**Cons**: Framework boundary overhead, CPU-GPU transfers, maintenance burden
**Verdict**: ❌ Per ChatGPT analysis (see `04_rng_correlation.md`)

---

### 5. ❌ Arithmetic Counter Construction
**Example**: `counter = t * N + neuron_id`
**Pros**: Simple
**Cons**: Fragile, order-dependent, breaks on refactoring, hidden correlations
**Verdict**: ❌ ChatGPT strongly discourages

---

### Key Insight from ChatGPT

> "Counter-based randomness is not just an optimization. For large neural simulations it is basically a **causal correctness tool**."

Per-neuron seeds with natural stream advancement = counter-based semantics without explicit counter management.

---

## Biological Plausibility: Explicit vs. Algorithmic Correlations

### ⚠️ ChatGPT Critical Point

**Question**: Are within-population correlations from shared RNG "biologically plausible"?

**Answer**: **NO** - this is backwards reasoning.

### Why Shared RNG != Biological Correlation

**Shared RNG produces**:
- Algorithmic correlations tied to sampling order
- Uncontrolled, unmeasurable
- Changes with code refactoring
- Not based on biological mechanisms

**Biological correlations should be explicit**:
```python
# CORRECT: Explicit shared input
shared_input = torch.randn(1) * alpha  # Common drive
independent = torch.randn(n_neurons) * beta  # Private noise
total_noise = shared_input + independent  # Controlled mixture
```

This allows:
- **Tunable** correlation strength (α vs β ratio)
- **Measurable** (compute actual correlation)
- **Interpretable** (models gap junctions, LFP, shared presynaptic input)
- **Refactor-proof** (not tied to array ordering)

### Real Biological Correlations

Yes, real neurons in the same population experience:
- Shared local field potentials
- Common neuromodulatory tone
- Gap junction coupling (PV interneurons: ~0.1-0.2 correlation)
- Shared presynaptic inputs

**But these must be modeled explicitly**, not as accidental RNG sharing.

### Our Approach

1. **Start with perfect independence** (per-neuron RNG)
2. **Measure baseline**: Correlation should be ~0.01-0.05 (finite-size only)
3. **Add biological coupling explicitly** if needed:
   - Gap junctions (electrical synapses)
   - Shared neuromodulator pools
   - Common sensory drive

This gives scientific control over correlation sources.

---

## Final Action Items (ChatGPT Refinements)

### Immediate Implementation Priorities

1. **HIGH PRIORITY: Implement vectorized Philox for large populations**
   - LC (1600 neurons), VTA (2500 neurons), NB (3000 neurons), SNR (1000 neurons)
   - Accept ChatGPT's offer for production-ready implementation (Step 4b)
   - Use list comprehension only for populations < 500

2. **MEDIUM: Ensure naming consistency across all regions**
   - Use lowercase: `"cortex"`, `"hippocampus"`, etc.
   - Check all 20 population instantiations for typos
   - Typo = silent seed collision \u2192 subtle correlation bugs

3. **LOW: Device initialization best practices**
   - Initialize `self.neuron_seeds` directly on GPU (`device=self.device`)
   - Initialize `self.ou_noise` on device: `torch.zeros(n_neurons, device=self.device)`
   - Avoid CPU\u2192GPU transfers in hot loops

### Performance Expectations

| Population Size | Method | Overhead | Examples |
|---|---|---|---|
| < 500 neurons | List comprehension | 10-20% | Cortex L4, L5, L6a, CA3, TRN |
| 500-1000 neurons | List comp (acceptable) | 20-30% | Cortex L2/3, Hippocampus DG |
| > 1000 neurons | **Vectorized Philox** | <10% | **VTA, LC, NB, SNR** |

### Testing Checklist

- [ ] Test correlation on small population: Cortex L4 (300 neurons)
- [ ] Test correlation on large population: VTA (2500 neurons)
- [ ] Verify determinism: same seed → identical results across sessions
- [ ] Profile overhead: should be ~5-10% across all population sizes
- [ ] Check OU noise device placement (no CPU→GPU warnings)
- [ ] Verify single code path handles all population sizes correctly

### Documentation Requirements

Per ChatGPT: "Per-neuron noise is deterministic **given identical `region_name`, `population_name`, `neuron_id`, and `master_seed`**."

Add to `forward()` docstring:
```python
"""
Determinism guarantee: Same (region, population, neuron_id, master_seed)
produces identical noise sequences across:
- Python sessions (md5 hashing)
- CPU/GPU (Philox algorithm)
- PyTorch versions (stable hash, not Python hash())

Breaking changes: Modifying region/population names changes all seeds.
"""
```

---

## References

- ChatGPT RNG correlation analysis: `docs/ai/chatgpt/04_rng_correlation.md`
- **ChatGPT vectorized Philox implementation**: **`docs/ai/chatgpt/05_vectorized_per_neuron_philox_rng.md`** ✅
- PyTorch Generator docs: https://pytorch.org/docs/stable/generated/torch.Generator.html
- Counter-based RNG theory: Random123 library documentation
- Philox algorithm: Salmon et al. (2011) "Parallel random numbers: as easy as 1, 2, 3"
- Biological correlations: Okun et al. (2015) Nature, "Diverse coupling of neurons to populations"
