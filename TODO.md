# TODO

---

    # Note: LC and NB currently don't have settable internal state
    # Future: Add state restoration for LC and NB when needed

---

- optional: add noise to the whole brain (processing (weights) + temporal (spikes))
  - enable during curriculum training? (see curriculum strategy)
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
