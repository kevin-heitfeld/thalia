# TODO

---

- 🟡 Missing gap junctions and electrical synapses
- 🟡 Limited glial influence on learning

---

- Delay precision: Are delays accurate to 0.1ms? Axons show sub-millisecond variability
- Synapse morphology: Are different synapse types (mossy, climbing, etc.) modeled?
- Dendritic computation: Are dendritic NMDA spikes and plateau potentials included?
- Axonal branching: Do axons branch and target multiple regions with different delays?
- Myelination dynamics: Do delays change with activity (myelin plasticity)?

---

- ✅ **REMOVED `gamma_n_slots` parameter** - Design flaw fixed (Dec 23, 2025)
  - **Problem**: Hardcoded slot assignment was biologically implausible
    - Old: `neuron[i] % 7 = slot[i]` - explicit, non-emergent
    - Biology: Phase preference emerges from synaptic delays + STDP
  - **Why 7 is still correct**: Lisman & Jensen (2013) - 7±2 gamma cycles per theta
  - **Solution Implemented**: Phase coding now EMERGES from:
    1. ✅ Phase diversity initialization (`phase_jitter_std_ms=5.0`)
       - Adds timing jitter to CA3 recurrent weights (~±15% variation)
       - Simulates effect of different axonal path lengths
    2. ✅ Gamma amplitude modulation (not slot gating!)
       - Neurons more responsive during high gamma amplitude
       - Creates temporal windows WITHOUT explicit neuron→slot mapping
    3. ✅ STDP naturally strengthens phase-appropriate connections
    4. ✅ Dendritic integration (~15ms) naturally filters by timing
  - **Result**: Working memory capacity = gamma_freq/theta_freq (~5-7 items)
  - **Tests**: `test_phase_coding_emergence.py` validates:
    - No `_ca3_slot_assignment` attribute
    - Phase diversity in weight initialization
    - Emergent phase selectivity through learning
    - STDP strengthens phase preferences
  - **Files Modified**:
    - `config.py`: Replaced `gamma_n_slots` with `phase_diversity_init`
    - `trisynaptic.py`: Removed slot gating (lines 1141-1182)
    - `checkpoint_manager.py`: Removed slot_assignment save/load
  - **Status**: ✅ COMPLETE - More biologically plausible!
- `docs\design\circuit_modeling_plan.md`
- Consciousness / Self-Awareness:
  - https://www.youtube.com/watch?v=OlnioeAtloY
  - How to: Mirror Test?

---

## Architecture & Infrastructure

- [ ] Sleep/Wake System 🟢 **LOW PRIORITY**
  - Already partially handled by oscillator frequency modulation
  - Could extract if needed for specific tasks
  - Not urgent given current capabilities
  - See: `docs/architecture/CENTRALIZED_SYSTEMS.md` for details
