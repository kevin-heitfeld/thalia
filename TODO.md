# TODO

---

- 🟡 Missing gap junctions and electrical synapses
- 🟡 Limited glial influence on learning

---

- **Oscillation Emergence Analysis** (see below for details):
  - ✅ **Beta/Delta**: Keep explicit (systems-level coordination)
  - ⚠️ **Gamma (40Hz)**: SHOULD emerge from L6→TRN→Thalamus loop (~25ms latency)
    - Currently failing: `test_gamma_oscillation_emergence` shows L6 inactive
    - Action: Fix L6 activation, validate gamma emergence from feedback timing
  - ⚠️ **Theta (8Hz)**: Should PARTIALLY emerge from CA3 recurrence
    - Keep global coordinator for multi-region sync
    - Validate hippocampus shows theta even without explicit oscillator
  - ⚠️ **Alpha (10Hz)**: Could emerge from thalamic T-currents (if implemented)
  - 🔄 **Theta-Gamma Coupling**: Should emerge from hippocampus-cortex anatomy
  - **Validation Strategy**: FFT analysis of population activity, autocorrelation, loop timing measurements
  - **Next Steps**: (1) Fix L6 activity, (2) Test gamma emergence, (3) Add CA3 theta test
- `docs\design\circuit_modeling_plan.md`
- Sparse computations throughout?
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
