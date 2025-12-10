# Architecture Documentation

High-level architectural documentation for the Thalia framework.

## Key Documents

### Oscillator System (Complete)
- **[Oscillator Architecture](oscillator-architecture.md)** - Complete centralized oscillator implementation
  - All 5 brain oscillators (delta, theta, alpha, beta, gamma)
  - Cross-frequency coupling via weighted averaging
  - ~300 lines, <0.001% overhead
  - Key insight: Implementation simpler than expected
  
- **[Oscillator Integration Complete](OSCILLATOR_INTEGRATION_COMPLETE.md)** - Achievement summary
  - All 5 biologically-motivated couplings working
  - 80/80 tests passing
  - Proof that oscillators are a "free lunch" for biological realism

## Contents

This directory contains:
- System architecture overviews
- Component interaction diagrams
- Integration patterns
- Cross-region communication models

## Related Documentation

- **[Design Docs](../design/)** - Detailed design specifications
- **[Patterns](../patterns/)** - Common implementation patterns
- **[Decisions](../decisions/)** - Architecture decision records (ADRs)

## Status

✅ Oscillator system documentation complete and up-to-date (Dec 2025)

