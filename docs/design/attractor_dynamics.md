# Attractor Dynamics

## Overview

Attractors are stable patterns of neural activity that the network settles into. They form the basis for:
- Working memory (sustained activity)
- Pattern completion
- Decision making
- Thought representation

## Types of Attractors

### Point Attractors
Fixed patterns of activity. Used for discrete memories and decisions.

### Line Attractors
Continuous manifolds of stable states. Used for analog working memory.

### Limit Cycles
Periodic activity patterns. Could represent temporal sequences.

### Chaotic Attractors
Complex dynamics that may support creativity and exploration.

## Implementation Strategy

1. **Excitatory-Inhibitory Balance**: Critical for stable dynamics
2. **Recurrent Connections**: Create basins of attraction
3. **Synaptic Facilitation**: Strengthen frequently-visited states
4. **STDP + Homeostasis**: Learn new attractors while maintaining stability

## Key Equations

Energy function for Hopfield-like networks:

$$E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j + \sum_i \theta_i s_i$$

Network dynamics minimize this energy, converging to attractors.

## References

- Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities.
- Amit, D.J. (1989). Modeling Brain Function.
