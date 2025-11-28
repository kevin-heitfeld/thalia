#!/usr/bin/env python3
"""
Example: STDP Learning Demonstration

Shows how STDP modifies synaptic weights based on spike timing.
"""

import torch
from thalia.learning.stdp import STDP, STDPConfig


def main():
    print("STDP Learning Demonstration")
    print("=" * 50)

    # Configure STDP with visible learning rates
    config = STDPConfig(
        tau_plus=20.0,
        tau_minus=20.0,
        a_plus=0.1,
        a_minus=0.1,
        w_min=0.0,
        w_max=1.0
    )

    # Simple 1-to-1 synapse to visualize STDP
    stdp = STDP(n_pre=1, n_post=1, config=config)

    print("\nSTDP Parameters:")
    print(f"  τ+ = {config.tau_plus} ms (potentiation time constant)")
    print(f"  τ- = {config.tau_minus} ms (depression time constant)")
    print(f"  A+ = {config.a_plus} (potentiation amplitude)")
    print(f"  A- = {config.a_minus} (depression amplitude)")

    # Case 1: Post after Pre (LTP - potentiation)
    print("\n" + "-" * 50)
    print("Case 1: Pre spike → Post spike (LTP)")
    print("-" * 50)

    stdp.reset_traces(batch_size=1)

    # Pre spike at t=0
    dw = stdp(torch.ones(1, 1), torch.zeros(1, 1))
    print(f"t=0:  Pre spike  | Pre trace: {stdp.trace_pre.item():.4f} | ΔW: {dw.item():+.4f}")

    # Decay for 5ms
    for t in range(1, 5):
        dw = stdp(torch.zeros(1, 1), torch.zeros(1, 1))
        print(f"t={t}:  (decay)    | Pre trace: {stdp.trace_pre.item():.4f} | ΔW: {dw.item():+.4f}")

    # Post spike at t=5ms
    dw = stdp(torch.zeros(1, 1), torch.ones(1, 1))
    print(f"t=5:  Post spike | Pre trace: {stdp.trace_pre.item():.4f} | ΔW: {dw.item():+.4f} ← LTP!")

    # Case 2: Pre after Post (LTD - depression)
    print("\n" + "-" * 50)
    print("Case 2: Post spike → Pre spike (LTD)")
    print("-" * 50)

    stdp.reset_traces(batch_size=1)

    # Post spike at t=0
    dw = stdp(torch.zeros(1, 1), torch.ones(1, 1))
    print(f"t=0:  Post spike | Post trace: {stdp.trace_post.item():.4f} | ΔW: {dw.item():+.4f}")

    # Decay for 5ms
    for t in range(1, 5):
        dw = stdp(torch.zeros(1, 1), torch.zeros(1, 1))
        print(f"t={t}:  (decay)    | Post trace: {stdp.trace_post.item():.4f} | ΔW: {dw.item():+.4f}")

    # Pre spike at t=5ms
    dw = stdp(torch.ones(1, 1), torch.zeros(1, 1))
    print(f"t=5:  Pre spike  | Post trace: {stdp.trace_post.item():.4f} | ΔW: {dw.item():+.4f} ← LTD!")

    # Show timing window effect
    print("\n" + "-" * 50)
    print("STDP Timing Window")
    print("-" * 50)
    print("\nΔt = t_post - t_pre")
    print("Δt > 0: Pre before Post → Potentiation (LTP)")
    print("Δt < 0: Post before Pre → Depression (LTD)")
    print("\n  Δt (ms) │  ΔW")
    print("  ────────┼────────")

    for delta_t in [-20, -10, -5, -2, 2, 5, 10, 20]:
        stdp.reset_traces(batch_size=1)

        if delta_t > 0:
            # Pre then post
            stdp(torch.ones(1, 1), torch.zeros(1, 1))
            for _ in range(abs(delta_t) - 1):
                stdp(torch.zeros(1, 1), torch.zeros(1, 1))
            dw = stdp(torch.zeros(1, 1), torch.ones(1, 1))
        else:
            # Post then pre
            stdp(torch.zeros(1, 1), torch.ones(1, 1))
            for _ in range(abs(delta_t) - 1):
                stdp(torch.zeros(1, 1), torch.zeros(1, 1))
            dw = stdp(torch.ones(1, 1), torch.zeros(1, 1))

        bar_width = int(abs(dw.item()) * 50)
        if dw.item() > 0:
            bar = "+" + "█" * bar_width
        else:
            bar = "-" + "█" * bar_width
        print(f"  {delta_t:+4d}    │ {dw.item():+.4f}  {bar}")

    print("\nDone!")


if __name__ == "__main__":
    main()
