# TODO

---

- 🟡 Missing gap junctions and electrical synapses
- 🟡 Limited glial influence on learning
- 🟡 Thalamus-cortex-TRN feedback loop incomplete
- 🟡 Cerebellum implementation could be more detailed

---

- Add spillover/volume transmission?
  - Spatial organization required?
  - Integrate into weight matrix of receiving components rather than spike output of transmitting components (because we are using binary spikes, see ADR-004)
  - Performance implications?
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
