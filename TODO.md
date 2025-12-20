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

- **Oscillation Emergence Analysis** (see `docs/architecture/OSCILLATION_EMERGENCE_ANALYSIS.md` for full analysis):
  - ✅ **Beta/Delta**: Keep explicit (systems-level coordination, not circuit-specific)
  - ✅ **Gamma (40Hz)**: ✅ EMERGES from L6→TRN→Thalamus loop - EXPLICIT OSCILLATOR DISABLED BY DEFAULT
    - **STATUS**: ✅ Implementation complete (Dec 20, 2025)
    - **L6 Activity**: Confirmed active (1318 total spikes over 50ms)
    - **Gamma Default**: DISABLED in `OscillatorManager.__init__()` (line 617)
    - **Tests**: ✅ 10/10 passing (integration + unit tests validate emergence)
    - **FFT Validation**: ✅ Measures 25 Hz (40ms period) - low gamma band
      - **Analysis**: Loop delay ~40ms vs expected ~30ms
      - **Root cause**: Missing explicit axonal delays (`l6_to_trn_delay_ms=0`, should be 10ms)
      - **Membrane integration**: ~12-15ms added by neuron time constants
      - **Solution**: Add explicit delays for faster oscillation
    - **L6a/L6b Implementation**: ✅ APPROVED - Will implement dual pathways
      - **Decision**: Implement L6a→TRN and L6b→relay pathways (Dec 20, 2025)
      - **Rationale**: Biological completeness, enables dual gamma bands (25Hz + 66Hz)
      - **Benefits**: Low gamma (L6a) + high gamma (L6b), fast sensory modulation
      - **Complexity**: ~150 lines code, two pathways to tune
      - **Plan**: See `OSCILLATION_EMERGENCE_ANALYSIS.md` section 1.6 for implementation
      - **Status**: 🔵 TO IMPLEMENT (next sprint)
    - **Cross-Area Gamma**: Different cortical areas show different frequencies (30-80 Hz)
      - **Current**: 25 Hz (matches visual/auditory cortex)
      - **Future**: Test when we have multiple brain presets (visual, motor, PFC)
      - **Implementation**: Use different delays per region + L6a/L6b ratio
    - **Tools Created**: `diagnostics/oscillation_detection.py` (FFT, autocorrelation)
    - **Enable if needed**: `brain.oscillators.enable_oscillator('gamma', True)`
  - ✅ **Theta (8Hz)**: REQUIRES central coordinator (OscillatorManager = biological septum) ✅ CORRECT APPROACH
    - **Reality**: CA3 recurrence WITHOUT septum gives 10-20 Hz (too fast)
    - **Biology**: Medial septum coordinates theta across entire brain
    - **Decision**: Keep OscillatorManager (functionally equivalent to septum)
    - **Optional**: Test CA3 intrinsic dynamics (may show 15-20 Hz without coordination)
  - ❌ **Alpha (10Hz)**: Keep explicit oscillator (T-currents not worth implementation cost)
    - **Reality**: Requires T-type calcium channels (~500+ lines new neuron model)
    - **Decision**: Explicit oscillator sufficient for current needs
    - **Future**: Only implement if adding sleep/wake states (burst vs tonic modes)
  - ✅ **Theta-Gamma Coupling**: Already centralized in OscillatorManager ✅ NO ACTION NEEDED
    - **Status**: `get_coupled_amplitude()` handles phase-amplitude coupling
    - **Note**: Anatomical coupling (hippocampus→cortex) also exists naturally
  - **Key Insight**: LOCAL circuits (gamma) emerge, DISTRIBUTED networks (theta) need coordinator
  - **Validation Tools**: ✅ Created `oscillation_detection.py` with FFT/autocorrelation
  - **Immediate Actions**: ✅ Complete - All validation done

---

- Add growth() support to ConductanceLIF
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
