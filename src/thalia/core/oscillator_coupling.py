"""
Oscillator Coupling Manager - Broadcasts oscillator phases to brain regions.

This module provides a centralized manager for distributing oscillator signals
(phases, amplitudes, slots) to all brain regions that need them. This reduces
boilerplate in Brain and makes coupling configuration explicit and tunable.

Design Philosophy:
==================
- Single source of truth for oscillator-region coupling
- Explicit configuration of which regions receive which oscillators
- Configurable coupling strengths per region-oscillator pair
- Follows same broadcast pattern as dopamine system

Usage:
======
    # In EventDrivenBrain __init__:
    self.oscillator_coupling = OscillatorCouplingManager(
        oscillators=self.oscillators,
        regions={
            "cortex": self.cortex.impl,
            "hippocampus": self.hippocampus.impl,
            "pfc": self.pfc.impl,
        },
        couplings={
            "cortex:theta": 1.0,
            "cortex:gamma": 0.8,
            "hippocampus:theta": 1.0,
            "pfc:theta": 0.6,
            "pfc:alpha": 0.5,
        }
    )

    # Each timestep (after oscillators.advance()):
    self.oscillator_coupling.broadcast()

Benefits:
=========
- Reduces boilerplate (no manual hasattr checks)
- Makes coupling configuration explicit and tunable
- Enables oscillator experiments without modifying Brain
- Consistent with dopamine broadcast architecture
- Easier debugging (centralized logging point)

Author: Thalia Team
Date: December 11, 2025
"""

from typing import Dict, Optional, Any
from thalia.core.oscillator import OscillatorManager


class OscillatorCouplingManager:
    """Manages oscillator-region coupling with configurable strength.

    Broadcasts oscillator phases, signals, and slots to regions that need them.
    Each region-oscillator pair has an optional coupling strength that modulates
    the effective amplitude of the oscillator for that region.

    Attributes:
        oscillators: The OscillatorManager providing phases/signals
        regions: Dictionary mapping region names to region implementations
        couplings: Dictionary mapping "region:oscillator" to coupling strength
        n_theta_slots: Number of slots for theta phase coding (default: 7)
    """

    def __init__(
        self,
        oscillators: OscillatorManager,
        regions: Dict[str, Any],
        couplings: Optional[Dict[str, float]] = None,
        n_theta_slots: int = 7,
    ):
        """Initialize oscillator coupling manager.

        Args:
            oscillators: OscillatorManager instance providing phases/signals
            regions: Dictionary mapping region name → region implementation
                Example: {"cortex": brain.cortex.impl, "hippocampus": brain.hippocampus.impl}
            couplings: Dictionary mapping "region:oscillator" → coupling strength (0-1)
                Example: {"cortex:theta": 1.0, "hippocampus:gamma": 0.5}
                Regions not listed receive signals with strength 1.0 (full coupling)
            n_theta_slots: Number of slots for theta phase coding (default: 7)
        """
        self.oscillators = oscillators
        self.regions = regions
        self.couplings = couplings or {}
        self.n_theta_slots = n_theta_slots

        # Cache region names that support oscillator phases
        self._capable_regions = {
            name: region
            for name, region in regions.items()
            if hasattr(region, 'set_oscillator_phases')
        }

    def broadcast(self) -> None:
        """Broadcast current oscillator state to all coupled regions.

        Gets current phases, signals, theta slot, and effective amplitudes from
        the OscillatorManager and distributes them to all regions that implement
        set_oscillator_phases().

        Regions can ignore oscillators they don't need - this just makes them
        available. Coupling strengths are applied via effective_amplitudes.

        Called every timestep after oscillators.advance().
        """
        # Get oscillator state from manager
        phases = self.oscillators.get_phases()
        signals = self.oscillators.get_signals()
        effective_amplitudes = self.oscillators.get_effective_amplitudes()
        theta_slot = self.oscillators.get_theta_slot(n_slots=self.n_theta_slots)

        # Apply region-specific coupling strengths
        for region_name, region in self._capable_regions.items():
            # Create region-specific effective amplitudes
            region_amplitudes = self._compute_region_amplitudes(
                region_name, effective_amplitudes
            )

            # Broadcast to region
            region.set_oscillator_phases(
                phases=phases,
                signals=signals,
                theta_slot=theta_slot,
                effective_amplitudes=region_amplitudes,
            )

    def _compute_region_amplitudes(
        self,
        region_name: str,
        base_amplitudes: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute region-specific effective amplitudes.

        Takes base effective amplitudes (already computed with cross-frequency
        coupling) and applies region-specific coupling strengths.

        Args:
            region_name: Name of the region
            base_amplitudes: Base effective amplitudes from OscillatorManager

        Returns:
            Dictionary mapping oscillator name → region-specific amplitude
        """
        region_amplitudes = {}

        for osc_name, base_amp in base_amplitudes.items():
            # Check for region-specific coupling strength
            coupling_key = f"{region_name}:{osc_name}"
            strength = self.couplings.get(coupling_key, 1.0)  # Default: full coupling

            # Apply coupling strength
            region_amplitudes[osc_name] = base_amp * strength

        return region_amplitudes

    def set_coupling_strength(
        self,
        region: str,
        oscillator: str,
        strength: float
    ) -> None:
        """Dynamically adjust coupling strength for a region-oscillator pair.

        This allows runtime modification of coupling (e.g., increasing theta
        coupling during encoding, decreasing during retrieval).

        Args:
            region: Name of the region (e.g., "cortex")
            oscillator: Name of the oscillator (e.g., "theta")
            strength: Coupling strength (0.0 = no coupling, 1.0 = full coupling)

        Example:
            # Strengthen hippocampal theta during encoding
            manager.set_coupling_strength("hippocampus", "theta", 1.0)

            # Weaken cortical alpha during attention task
            manager.set_coupling_strength("cortex", "alpha", 0.3)
        """
        coupling_key = f"{region}:{oscillator}"
        self.couplings[coupling_key] = max(0.0, min(1.0, strength))  # Clamp 0-1

    def get_coupling_strength(self, region: str, oscillator: str) -> float:
        """Get current coupling strength for a region-oscillator pair.

        Args:
            region: Name of the region
            oscillator: Name of the oscillator

        Returns:
            Current coupling strength (0.0 to 1.0)
        """
        coupling_key = f"{region}:{oscillator}"
        return self.couplings.get(coupling_key, 1.0)

    def get_coupling_config(self) -> Dict[str, float]:
        """Get complete coupling configuration.

        Returns:
            Dictionary of all configured couplings
        """
        return self.couplings.copy()

    def add_region(self, name: str, region: Any) -> None:
        """Add a new region to coupling manager.

        Useful for dynamic brain construction or adding regions after
        manager initialization.

        Args:
            name: Name of the region
            region: Region implementation (must have set_oscillator_phases method)
        """
        self.regions[name] = region

        # Update capable regions cache if method exists
        if hasattr(region, 'set_oscillator_phases'):
            self._capable_regions[name] = region

    def remove_region(self, name: str) -> None:
        """Remove a region from coupling manager.

        Args:
            name: Name of the region to remove
        """
        self.regions.pop(name, None)
        self._capable_regions.pop(name, None)

        # Remove any couplings for this region
        keys_to_remove = [k for k in self.couplings.keys() if k.startswith(f"{name}:")]
        for key in keys_to_remove:
            del self.couplings[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about oscillator coupling.

        Useful for monitoring and debugging.

        Returns:
            Dictionary with:
            - n_regions: Number of regions
            - n_capable_regions: Number implementing set_oscillator_phases
            - n_couplings: Number of configured couplings
            - regions: List of region names
            - capable_regions: List of regions receiving oscillators
            - couplings: Current coupling configuration
        """
        return {
            "n_regions": len(self.regions),
            "n_capable_regions": len(self._capable_regions),
            "n_couplings": len(self.couplings),
            "regions": list(self.regions.keys()),
            "capable_regions": list(self._capable_regions.keys()),
            "couplings": self.get_coupling_config(),
        }
