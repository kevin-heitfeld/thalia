"""STP Equilibrium Calculator.

Wraps :class:`~thalia.brain.synapses.stp.STPConfig` to provide convenient
analytical inspection of short-term-plasticity (STP) at steady-state firing
rates — no simulation required.

At a constant presynaptic firing rate *r* (ms⁻¹ = rate_hz / 1000):

    u_ss = U · (1 + r · τ_f) / (1 + U · r · τ_f)      [utilisation]
    x_ss = 1 / (1 + u_ss · r · τ_d)                    [resource fraction]
    eff  = u_ss · x_ss                                  [effective weight scale]

One common mistake during calibration is setting STP parameters on a pathway
and forgetting that—at the population's equilibrium firing rate—the synapse is
severely depleted, so what looks like an adequate weight actually drives the
post-synaptic neuron at a fraction of the intended conductance.

Usage::

    from thalia.diagnostics.stp_calculator import stp_eq, stp_table

    # Single query
    r = stp_eq(U=0.50, tau_d=800, tau_f=20, rate_hz=20.0)
    r.print()
    # → u=0.615, x=0.068, eff=0.042  ← very depleted

    # Sweep across rates
    stp_table(U=0.25, tau_d=200, tau_f=50)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from thalia.brain.synapses.stp import STPConfig


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class STPResult:
    """Steady-state STP values at a single presynaptic firing rate.

    Attributes
    ----------
    U :
        Baseline release probability (STPConfig.U).
    tau_d :
        Depression recovery time constant (ms).
    tau_f :
        Facilitation decay time constant (ms).
    rate_hz :
        Queried presynaptic firing rate (Hz).
    u_ss :
        Steady-state utilisation (effective release probability).
    x_ss :
        Steady-state resource fraction (vesicle availability).
    eff :
        Combined efficacy *u_ss × x_ss* — multiply base weight by this.
    """

    U: float
    tau_d: float
    tau_f: float
    rate_hz: float
    u_ss: float
    x_ss: float
    eff: float

    def print(self) -> None:
        """Print a compact summary of the STP state."""
        w = 56
        print(f"\n{'─' * w}")
        print("  STP Equilibrium")
        print(f"{'─' * w}")
        print(f"  U={self.U:.3f}  tau_d={self.tau_d:.0f}ms  tau_f={self.tau_f:.0f}ms")
        print(f"  Presynaptic rate : {self.rate_hz:.1f} Hz")
        print(f"  u_ss (utilisation) : {self.u_ss:.4f}")
        print(f"  x_ss (resources)   : {self.x_ss:.4f}")
        eff_flag = ""
        if self.eff < 0.10:
            eff_flag = "  ⚠ SEVERELY DEPLETED"
        elif self.eff < 0.30:
            eff_flag = "  ⚠ depleted"
        elif self.eff > 0.90:
            eff_flag = "  ✓ facilitated / near-baseline"
        else:
            eff_flag = "  ✓"
        print(f"  eff = u×x          : {self.eff:.4f}{eff_flag}")
        print(f"{'─' * w}\n")


# ---------------------------------------------------------------------------
# stp_eq — point query
# ---------------------------------------------------------------------------

def stp_eq(
    U: float,
    tau_d: float,
    tau_f: float,
    rate_hz: float,
) -> STPResult:
    """Return steady-state STP values at a single rate.

    Parameters
    ----------
    U :
        Baseline utilisation (release probability).
    tau_d :
        Depression recovery time constant (ms).
    tau_f :
        Facilitation decay time constant (ms).
    rate_hz :
        Presynaptic firing rate (Hz).

    Returns
    -------
    STPResult
    """
    stp = STPConfig(U=U, tau_d=tau_d, tau_f=tau_f)
    u_ss, x_ss = stp.steady_state_ux(rate_hz)
    return STPResult(
        U=U,
        tau_d=tau_d,
        tau_f=tau_f,
        rate_hz=rate_hz,
        u_ss=float(u_ss),
        x_ss=float(x_ss),
        eff=float(u_ss * x_ss),
    )


# ---------------------------------------------------------------------------
# stp_table — rate sweep
# ---------------------------------------------------------------------------

def stp_table(
    U: float,
    tau_d: float,
    tau_f: float,
    rates_hz: Optional[List[float]] = None,
) -> None:
    """Print a table of STP values across a sweep of firing rates.

    Parameters
    ----------
    U :
        Baseline utilisation.
    tau_d :
        Depression recovery time constant (ms).
    tau_f :
        Facilitation decay time constant (ms).
    rates_hz :
        List of presynaptic firing rates (Hz).
        Defaults to ``[0.5, 1, 2, 5, 10, 20, 30, 50, 80, 100]``.
    """
    if rates_hz is None:
        rates_hz = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0]

    stp = STPConfig(U=U, tau_d=tau_d, tau_f=tau_f)

    print(f"\n  STP Sweep  U={U}  tau_d={tau_d}ms  tau_f={tau_f}ms")
    print(f"  {'Rate(Hz)':>9} {'u_ss':>8} {'x_ss':>8} {'eff':>8}  note")
    print(f"  {'-'*9} {'-'*8} {'-'*8} {'-'*8}  {'─'*20}")

    for r in rates_hz:
        u_ss, x_ss = stp.steady_state_ux(r)
        eff = u_ss * x_ss
        note = ""
        if eff < 0.10:
            note = "⚠ severely depleted"
        elif eff < 0.30:
            note = "⚠ depleted"
        elif eff > 1.20 * U:
            note = "↑ facilitated"
        print(f"  {r:>9.1f} {float(u_ss):>8.4f} {float(x_ss):>8.4f} {float(eff):>8.4f}  {note}")

    print()
