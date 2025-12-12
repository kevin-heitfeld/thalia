# TODO

## Architecture & Infrastructure

- Remove `RegionConfig`
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
