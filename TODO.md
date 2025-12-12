# TODO

---

> The DG→CA3→CA1 circuit is a single biological computation that must execute
> within one theta cycle (~100-150ms). Splitting would require passing ~20
> intermediate tensors between files and break the narrative flow of the
> biological computation.

---

## Architecture & Infrastructure

- TODO's in the codebase
- `PredictiveCortex` not registered with `@register_region`
- **Nothing** registered with `@register_module`
- See `PathwayConfig.bidirectional`, should those be split into two pathways (with own weight matrices)?
- Review public brain API (`new_trial()`, etc. necessary?)
- Add learning mechanisms to `CrossModalGammaBinding`

- [ ] Sleep/Wake System 🟢 **LOW PRIORITY**
  - Already partially handled by oscillator frequency modulation
  - Could extract if needed for specific tasks
  - Not urgent given current capabilities
  - See: `docs/architecture/CENTRALIZATION_ANALYSIS.md` for analysis

### Future Enhancements (Lower Priority)
- [ ] Adaptive coupling strength (learn optimal coupling per task)
- [ ] Region-specific coupling (different strengths per region)
- [ ] Oscillatory pathology detection (detect abnormal patterns)
- [ ] Cross-region phase synchrony metrics
