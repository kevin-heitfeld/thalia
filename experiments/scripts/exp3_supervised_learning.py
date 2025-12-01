#!/usr/bin/env python3
"""
Experiment 3: Supervised Learning with Dopamine-Modulated Plasticity

This experiment demonstrates how a spiking neural network can learn ARBITRARY
input-output mappings using a dopamine-based teaching signal, in contrast to
exp2 which learns only topographic (diagonal) mappings via unsupervised Hebbian
learning.

Key concepts:

THREE-FACTOR LEARNING: dW = pre * post * dopamine
  - pre: Presynaptic activity (input spike)
  - post: Postsynaptic activity (output spike)
  - dopamine: Teaching signal (positive = reward, negative = punishment)

ELIGIBILITY TRACES: Bridge timing between activity and reward
  - Synaptic activity creates a transient "eligibility" (~50-100ms)
  - When dopamine arrives, only eligible synapses are modified
  - This allows delayed reward to credit correct synapses

SUPERVISED vs UNSUPERVISED:
  - exp2 (unsupervised): Network discovers structure via correlations
    - Can only learn mappings implied by initial bias + input statistics
    - Diagonal bias -> learns diagonal mapping
  - exp3 (supervised): Teacher provides correct answer
    - Can learn ANY arbitrary mapping, even anti-topographic
    - Shuffled, reversed, or completely random mappings

Biological basis:
- Dopamine encodes "reward prediction error" (Schultz et al., 1997)
- VTA dopamine neurons: burst for unexpected reward, pause for omission
- Eligibility traces = calcium/CaMKII cascades in dendrites (~50-100ms)
- Three-factor rules found in striatum, prefrontal cortex, hippocampus

Success criteria:
1. Learn default mapping (same as exp2): >80% accuracy
2. Learn SHUFFLED mapping (arbitrary permutation): >80% accuracy
3. Learn REVERSED mapping (anti-topographic): >80% accuracy
4. Generalization: correct response to novel timing/input patterns
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

# Thalia library imports
from thalia.dynamics import (
    NetworkState,
    forward_timestep_with_stp,
    forward_pattern,
    create_temporal_pattern,
)
from thalia.learning import (
    EligibilityTraces,
    DopamineSystem,
    TargetMapping,
    create_default_mapping,
    create_shuffled_mapping,
    create_reversed_mapping,
    apply_dopamine_modulated_update,
    update_homeostatic_conductance_bidirectional,
    update_bcm_threshold,
    PredictiveCoding,
    hebbian_update,
)
from thalia.evaluation import (
    compute_paired_diagonal_score,
    analyze_recurrent_structure,
    print_recurrent_analysis,
)
from thalia.visualization import create_training_summary_figure
from thalia.diagnostics import DiagnosticConfig, ExperimentDiagnostics, MechanismConfig

# Experiment utilities
from experiments.scripts.exp_utils import (
    select_device,
    create_weight_matrix,
    create_recurrent_weights,
    create_output_neurons,
    create_network_config,
    create_default_synaptic_mechanisms,
    create_recurrent_stdp,
    add_common_experiment_args,
    add_mechanism_ablation_args,
    create_mechanism_config,
    print_experiment_header,
    print_success_criteria,
    compute_w_max,
    compute_hebbian_learning_rate,
    get_results_dir,
    DEFAULT_DT,
    DEFAULT_V_THRESHOLD,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Experiment 3: Supervised Learning with Dopamine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add common arguments from exp_utils
    add_common_experiment_args(parser)
    add_mechanism_ablation_args(parser)

    # Supervised learning specific args
    parser.add_argument("--mapping_type", type=str, default="shuffled",
                        choices=["default", "shuffled", "reversed"],
                        help="Target mapping type to learn")
    parser.add_argument("--shuffle_seed", type=int, default=42,
                        help="Random seed for shuffled mapping")

    # Dopamine parameters
    parser.add_argument("--dopamine_burst", type=float, default=1.0,
                        help="Dopamine level for correct response (reward)")
    parser.add_argument("--dopamine_dip", type=float, default=-0.5,
                        help="Dopamine level for incorrect response (punishment)")
    parser.add_argument("--dopamine_tau_ms", type=float, default=20.0,
                        help="Dopamine decay time constant (ms)")

    # Eligibility trace parameters
    # Biological value: 500-2000ms (Yagishita et al., 2014)
    # Needs to be long enough that all phases have similar eligibility at cycle end
    parser.add_argument("--eligibility_tau_ms", type=float, default=1000.0,
                        help="Eligibility trace decay (ms)")

    # Three-factor learning rate
    parser.add_argument("--supervised_lr", type=float, default=0.02,
                        help="Learning rate for dopamine-modulated updates")

    # Note: --diagonal_bias is in common args - for exp3 it's applied as a RANDOM
    # PERMUTATION (not strict diagonal) to provide temporal differentiation while
    # being equally distant from any target mapping on average.

    # Exploration noise for bootstrapping learning from uniform weights
    # Higher noise (20-30%) helps different neurons win randomly, enabling exploration
    # This is biologically plausible: cortical neurons show 10-25% subthreshold fluctuations
    # and noise increases during attention/learning
    parser.add_argument("--exploration_noise", type=float, default=0.3,
                        help="Membrane noise for exploration (0.3 = 30%% of threshold)")

    # Note: warmup_cycles is already defined in add_common_experiment_args

    # Compare supervised vs unsupervised
    parser.add_argument("--compare_unsupervised", action="store_true",
                        help="Also run unsupervised learning for comparison")

    return parser.parse_args()


# =============================================================================
# SUPERVISED TRAINING LOOP
# =============================================================================

def run_supervised_training(
    args,
    target_mapping: TargetMapping,
    device: torch.device,
    mech_config: MechanismConfig,
    verbose: bool = True,
) -> dict:
    """Run supervised training with dopamine modulation.

    Args:
        args: Parsed command line arguments
        target_mapping: The mapping to learn
        device: Torch device
        mech_config: Mechanism ablation config
        verbose: Print progress

    Returns:
        Dictionary with training results
    """
    # Network dimensions
    n_input = args.n_input
    n_output = args.n_output
    n_cycles = args.max_cycles
    dt = DEFAULT_DT

    # Weight parameters
    w_max = compute_w_max()
    hebbian_lr = compute_hebbian_learning_rate()

    # Timing parameters
    cycle_duration_ms = n_input * 8.0  # 160ms
    gap_duration_ms = args.gap_duration_ms if args.pattern_type == "gapped" else 0.0
    effective_cycle_ms = cycle_duration_ms + gap_duration_ms
    total_duration_ms = n_cycles * effective_cycle_ms

    cycle_duration = int(cycle_duration_ms / dt)
    effective_cycle = int(effective_cycle_ms / dt)
    total_duration = int(total_duration_ms / dt)

    # IMPORTANT: n_input=20 but we have n_output=10 logical phases
    # Each logical phase has 2 inputs (inputs_per_phase = n_input // n_output)
    # phase_duration should be per LOGICAL phase, not per input
    n_logical_phases = n_output
    phase_duration = cycle_duration // n_logical_phases  # 16ms per logical phase (not 8ms)

    if verbose:
        print(f"\n  Timing: {cycle_duration_ms}ms cycle + {gap_duration_ms}ms gap = {effective_cycle_ms}ms")
        print(f"  Total: {total_duration_ms}ms ({n_cycles} cycles)")

    # Create output neurons with exploration noise for symmetry breaking
    # Higher noise helps bootstrap learning from uniform weights
    output_neurons = create_output_neurons(
        n_output=n_output,
        neuron_model=args.neuron_model,
        device=device,
        n_input=n_input,
        dt=dt,
        noise_std=args.exploration_noise,
    )
    if verbose:
        print(f"  Neuron model: {args.neuron_model}")

    # ==========================================================================
    # WEIGHT INITIALIZATION with Random Permutation Bias
    # ==========================================================================
    # For supervised learning, we need temporal differentiation (different neurons
    # prefer different phases) so dopamine can differentially reward/punish.
    #
    # We use a RANDOM PERMUTATION (not strict diagonal) so the initial mapping
    # is equally distant from any target mapping on average. This provides a
    # fair test of the learning algorithm.
    #
    # The bias strength (diagonal_bias) controls how strong the initial preference is.

    # Start with base lognormal weights
    weights = create_weight_matrix(
        n_output=n_output,
        n_input=n_input,
        w_max=w_max,
        device=device,
        initial_fraction=0.10,  # Base weight level
        diagonal_bias_strength=0.0,  # We'll add our own random permutation bias
        distribution="lognormal",
    )

    # Apply random permutation bias for temporal differentiation
    # Each neuron gets a unique preferred phase (1-to-1 mapping via permutation)
    bias_strength = args.diagonal_bias if hasattr(args, 'diagonal_bias') else 0.3

    if bias_strength > 0:
        # Random permutation ensures each neuron prefers a DIFFERENT phase
        random_phases = torch.randperm(n_output, device=device)  # [7, 3, 0, 9, 2, ...]

        for neuron_idx in range(n_output):
            preferred_phase = random_phases[neuron_idx].item()
            input_start = int(preferred_phase * 2)  # 2 inputs per phase
            input_end = input_start + 2

            # Boost preferred inputs (strength determines how much)
            # bias_strength=0.3 → 1 + 3*0.3 = 1.9x boost
            # bias_strength=1.0 → 1 + 3*1.0 = 4x boost
            boost_factor = 1.0 + 3.0 * bias_strength
            weights[neuron_idx, input_start:input_end] *= boost_factor

            # Reduce non-preferred inputs to create contrast
            mask = torch.ones(n_input, device=device, dtype=torch.bool)
            mask[input_start:input_end] = False
            suppress_factor = 1.0 - 0.5 * bias_strength  # 0.3 → 0.85x, 1.0 → 0.5x
            weights[neuron_idx, mask] *= suppress_factor

        if verbose:
            print(f"  Random permutation bias: strength={bias_strength:.1f}")
            print(f"    Boost={1.0 + 3.0 * bias_strength:.1f}x, Suppress={1.0 - 0.5 * bias_strength:.2f}x")
            print(f"    Phase assignments: {random_phases.tolist()}")

    initial_weights = weights.clone()

    if verbose:
        print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Recurrent weights (learned via predictive coding, same as exp2)
    recurrent_weights = create_recurrent_weights(n_output, device)
    recurrent_stdp = create_recurrent_stdp(n_output, device)

    # Network config - use UNIFORM theta for supervised learning
    # Uniform theta provides global rhythmic gating (biologically realistic)
    # without biasing specific neurons. All neurons become more/less excitable
    # together, and dopamine signals guide which neuron wins.
    #
    # Note: theta_mode defaults to "uniform" in create_network_config, so we
    # just need to enable theta_modulation_strength.
    #
    # Reduce lateral inhibition for supervised learning to allow more exploration.
    # Default shunting=0.6 creates very strong winner suppression.
    # We want weaker inhibition so dopamine can shape competition over time.
    net_config, derived = create_network_config(
        n_input=n_input,
        n_output=n_output,
        device=device,
        dt=dt,
        neuron_model=args.neuron_model,
        theta_modulation_strength=args.theta_modulation_strength,  # Uniform theta enabled
        theta_mode="uniform",  # All neurons get same theta modulation
        sigma_inhibition=args.sigma_inhibition,
        shunting_relative_strength=0.3,  # Reduced from 0.6 - weaker lateral inhibition
        blanket_inhibition_strength=0.2,  # Reduced from 0.5 - weaker global inhibition
    )
    net_config.effective_cycle = effective_cycle  # Update for gapped pattern

    # Synaptic mechanisms
    synaptic_mechanisms = create_default_synaptic_mechanisms(
        n_input=n_input,
        n_output=n_output,
        device=device,
        enable_stp=not args.disable_stp if hasattr(args, 'disable_stp') else True,
        enable_nmda=not args.disable_nmda if hasattr(args, 'disable_nmda') else True,
    )

    # ==========================================================================
    # SUPERVISED LEARNING COMPONENTS
    # ==========================================================================

    # Eligibility traces: bridge timing between spikes and reward
    eligibility = EligibilityTraces(
        n_pre=n_input,
        n_post=n_output,
        tau_ms=args.eligibility_tau_ms,
        device=device,
    )

    # Dopamine system: provides teaching signal
    dopamine_system = DopamineSystem(
        target_mapping=target_mapping,
        burst_magnitude=args.dopamine_burst,
        dip_magnitude=args.dopamine_dip,
        tau_ms=args.dopamine_tau_ms,
        device=device,
    )

    if verbose:
        print(f"\n  Dopamine: burst={args.dopamine_burst}, dip={args.dopamine_dip}")
        print(f"  Eligibility tau: {args.eligibility_tau_ms}ms")
        print(f"  Supervised learning rate: {args.supervised_lr}")

    # Network state
    train_state = NetworkState.create(
        n_output=n_output,
        recurrent_delay=derived["cycle_duration"] // 16,
        interneuron_delay=20,
        v_threshold=DEFAULT_V_THRESHOLD,
        target_rate=0.02,
        device=device,
        n_input=n_input,
        max_ff_delay=derived["max_ff_delay"],
    )
    train_state.skip_spike_tracking = True

    # CRITICAL: Set input eligibility trace tau based on phase duration!
    # This prevents late-phase bias where all neurons learn to respond to late inputs.
    # With 16ms phases (160 timesteps at dt=0.1), tau must be >= 160 timesteps.
    # Using 1.5x phase_duration gives good coverage of the entire phase.
    eligibility_tau_timesteps = phase_duration * 1.5
    train_state.input_eligibility_tau = eligibility_tau_timesteps

    # Diagnostics for supervised learning
    diag_config = DiagnosticConfig.from_level(args.diagnostics if hasattr(args, 'diagnostics') else "summary")
    diag_config.collect_eligibility = True  # Enable eligibility tracking
    diag_config.collect_dopamine = True     # Enable dopamine tracking
    diagnostics = ExperimentDiagnostics(
        config=diag_config,
        n_neurons=n_output,
        n_phases=n_input,
        n_inputs=n_input,
        device=device,
    )

    # Predictive coding for recurrent learning
    gamma_period = derived["gamma_period"]
    predictive_coding = PredictiveCoding(
        n_output=n_output,
        gamma_period=gamma_period,
        learning_phase=gamma_period - 10,
        start_cycle=args.max_cycles // 2,  # Start recurrent after feedforward stabilizes
        base_lr=0.05,
        device=device,
    )

    # Create training pattern
    pattern = create_temporal_pattern(
        n_input, total_duration_ms, args.pattern_type,
        gap_duration_ms=gap_duration_ms, dt=dt
    ).to(device)

    # Homeostasis parameters (needed for warmup and training)
    target_firing_rate_hz = args.target_firing_rate_hz
    homeostatic_tau = 20.0
    homeostatic_strength_hz = 0.01

    # ==========================================================================
    # WARMUP PHASE (like exp2)
    # ==========================================================================
    # Run network without learning to:
    # 1. Stabilize homeostatic mechanisms (g_tonic, g_inh_tonic)
    # 2. Let STP vesicle dynamics reach steady-state
    # 3. Establish baseline firing patterns
    # 4. Prevent early learning artifacts

    warmup_cycles = args.warmup_cycles

    # Initialize g_tonic and g_inh_tonic for conductance-based neurons
    if args.neuron_model in ("conductance", "dendritic"):
        train_state.g_tonic = torch.full((1, n_output), 0.1, device=device)
        train_state.g_inh_tonic = torch.full((1, n_output), 0.2, device=device)

    # Set input eligibility tau for proper trace tracking
    train_state.input_eligibility_tau = int(12.0 / dt)  # 12ms like exp2

    # Only run warmup if warmup_cycles > 0
    if warmup_cycles > 0:
        warmup_duration = warmup_cycles * effective_cycle
        warmup_pattern = create_temporal_pattern(
            n_input, warmup_cycles * effective_cycle_ms, args.pattern_type,
            gap_duration_ms=gap_duration_ms, dt=dt
        ).to(device)

        if verbose:
            print(f"\n  Warmup phase: {warmup_cycles} cycles ({warmup_cycles * effective_cycle_ms}ms) without learning...")

        # Run warmup without learning
        warmup_spikes = 0
        for t in range(warmup_duration):
            input_spikes = warmup_pattern[t].unsqueeze(0)
            output_spikes, _ = forward_timestep_with_stp(
                t, input_spikes, train_state, net_config,
                weights, recurrent_weights, output_neurons,
                synaptic_mechanisms=synaptic_mechanisms
            )
            warmup_spikes += output_spikes.sum().item()

            # Update homeostasis during warmup (but no weight learning)
            if (t + 1) % effective_cycle == 0:
                cycle_num = (t + 1) // effective_cycle
                neuron_spikes_warmup = output_spikes  # Simplified tracking
                current_rate_hz = neuron_spikes_warmup * (1000.0 / effective_cycle_ms)
                if args.neuron_model in ("conductance", "dendritic"):
                    train_state.avg_firing_rate, train_state.g_tonic, train_state.g_inh_tonic = \
                        update_homeostatic_conductance_bidirectional(
                            current_rate=current_rate_hz,
                            avg_firing_rate=train_state.avg_firing_rate,
                            g_tonic=train_state.g_tonic,
                            g_inh_tonic=train_state.g_inh_tonic,
                            target_rate=target_firing_rate_hz,
                            tau=homeostatic_tau,
                            strength=homeostatic_strength_hz * 0.01,
                            exc_bounds=(0.0, 0.5),
                            inh_bounds=(0.0, 5.0),
                        )

        if verbose:
            print(f"    Warmup spikes: {warmup_spikes:.0f}")
            if train_state.g_tonic is not None:
                print(f"    Post-warmup g_tonic: [{train_state.g_tonic.min():.3f}, {train_state.g_tonic.max():.3f}]")
                print(f"    Post-warmup g_inh_tonic: [{train_state.g_inh_tonic.min():.3f}, {train_state.g_inh_tonic.max():.3f}]")

        # Reset for training - clear transient state but keep slow variables
        #
        # RESET (transient dynamics that shouldn't carry over):
        # - STP vesicle pools → full vesicles for fair start
        # - Refractory counters → no lingering inhibition
        #
        # KEEP (slow homeostatic variables we just stabilized):
        # - g_tonic, g_inh_tonic → conductance-based homeostasis
        # - avg_firing_rate → rate estimation
        #
        # DON'T RESET (mostly harmless either way):
        # - Membrane potential → will equilibrate quickly anyway

        # Reset STP - fresh vesicles for training
        if synaptic_mechanisms is not None and "stp" in synaptic_mechanisms:
            synaptic_mechanisms["stp"].reset()

        # Reset refractory state only (keep membrane, it equilibrates fast)
        if hasattr(output_neurons, 'refractory'):
            output_neurons.refractory = torch.zeros_like(output_neurons.refractory)
    else:
        if verbose:
            print(f"\n  Skipping warmup (warmup_cycles=0)")

    if verbose:
        print(f"\n  Training for {n_cycles} cycles...")

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    # Tracking
    accuracy_history = []
    weight_history = [weights.clone().cpu().numpy()]
    dopamine_history = []

    # BCM threshold
    bcm_threshold = torch.ones(1, n_output, device=device) * 0.5

    # Cycle counters
    current_cycle_spikes = torch.tensor(0.0, device=device)
    neuron_spikes = torch.zeros(1, n_output, device=device)
    correct_responses = 0
    total_responses = 0

    start_time = time.perf_counter()

    for t in range(total_duration):
        # Cycle tracking
        if t % effective_cycle == 0:
            cycle_num = t // effective_cycle + 1

            # Record eligibility before resetting (for diagnostics)
            if cycle_num > 1:
                target_for_diag = target_mapping.get_target_for_phase(0)  # Use first phase target
                diagnostics.record_eligibility(eligibility.get(), target_neuron=target_for_diag)
                diagnostics.end_cycle(cycle_num - 1)

            eligibility.reset()  # Fresh traces each cycle
            dopamine_system.reset()
            predictive_coding.reset()
            diagnostics.start_cycle(cycle_num)

            if cycle_num > 1:
                # Record accuracy for previous cycle
                accuracy = correct_responses / max(total_responses, 1)
                accuracy_history.append(accuracy)

                if cycle_num % 20 == 0 and verbose:
                    print(f"    Cycle {cycle_num}: accuracy={accuracy:.1%}, "
                          f"spikes={current_cycle_spikes.item():.0f}")

            # Reset cycle counters
            current_cycle_spikes = torch.tensor(0.0, device=device)
            neuron_spikes = torch.zeros(1, n_output, device=device)
            correct_responses = 0
            total_responses = 0

        # Get input
        input_spikes = pattern[t].unsqueeze(0)

        # Forward pass
        output_spikes, effective_weights = forward_timestep_with_stp(
            t, input_spikes, train_state, net_config,
            weights, recurrent_weights, output_neurons,
            synaptic_mechanisms=synaptic_mechanisms
        )

        # Track spikes
        neuron_spikes = neuron_spikes + output_spikes
        current_cycle_spikes = current_cycle_spikes + output_spikes.sum()

        # Phase tracking
        cycle_position = t % effective_cycle
        in_gap = args.pattern_type == "gapped" and cycle_position >= cycle_duration
        current_phase = (cycle_position % cycle_duration) // phase_duration if not in_gap else -1

        # =================================================================
        # SUPERVISED LEARNING: Hebbian + Dopamine Modulation
        # =================================================================
        #
        # Biological basis (three-factor rule):
        # 1. HEBBIAN LEARNING (always active): Δw ∝ pre × post
        #    - NMDA-mediated LTP when pre and post are co-active
        #    - This is the foundation that creates input selectivity
        #
        # 2. DOPAMINE MODULATION of Hebbian learning:
        #    - Dopamine burst (reward) → ENHANCE learning rate
        #    - Dopamine dip (punishment) → REDUCE or REVERSE learning
        #    - Baseline dopamine → normal Hebbian learning continues
        #
        # 3. ELIGIBILITY TRACES for credit assignment:
        #    - Bridge timing between activity and delayed reward
        #    - Allow dopamine to retroactively credit active synapses
        #
        # KEY INSIGHT: Dopamine doesn't replace Hebbian learning, it modulates it!
        # =================================================================

        # Update eligibility traces for credit assignment
        eligibility.update(input_spikes, output_spikes, dt)

        # Subthreshold eligibility: neurons close to threshold build partial eligibility
        if hasattr(output_neurons, 'membrane') and output_neurons.membrane is not None:
            eligibility.update_subthreshold(
                input_spikes,
                output_neurons.membrane,
                v_threshold=net_config.v_threshold,
                dt=dt,
                subthreshold_scale=0.3,
            )

        # Compute dopamine signal based on response correctness
        dopamine_level = dopamine_system.compute(
            output_spikes, current_phase, dt, in_gap
        )

        # Track response accuracy and diagnostics
        if output_spikes.sum() > 0 and not in_gap:
            total_responses += 1
            winner = int(output_spikes.squeeze().argmax().item())
            target = target_mapping.get_target_for_phase(current_phase)
            if winner == target:
                correct_responses += 1

            # Record winner for diagnostics (consolidated winner tracking)
            diagnostics.record_winner(int(current_phase), winner)
            diagnostics.record_phase_spikes(int(current_phase), output_spikes.squeeze())

            # Record dopamine event for diagnostics
            diagnostics.record_dopamine(
                timestep=t,
                dopamine_level=dopamine_level,
                was_correct=(winner == target),
                winner=winner,
                target=target,
            )

        # Get STP resources for learning gating
        # Prevents early-phase inputs from dominating learning
        stp_resources = None
        if synaptic_mechanisms is not None and "stp" in synaptic_mechanisms:
            stp_resources = synaptic_mechanisms["stp"].resources  # (n_post, n_pre)

        # Get input eligibility trace (like exp2)
        input_trace = train_state.input_eligibility
        if input_trace is None:
            input_trace = input_spikes.squeeze()

        # Normalize eligibility trace to prevent late-phase bias
        trace_sum = input_trace.sum()
        if trace_sum > 0:
            input_trace = input_trace / trace_sum

        # =================================================================
        # THREE-FACTOR LEARNING (Striatal Model)
        # =================================================================
        # Pure three-factor rule: Δw = eligibility × dopamine
        #
        # Biological basis (Yagishita et al., 2014; Izhikevich, 2007):
        # - In striatum, there is NO unsupervised Hebbian learning
        # - Pre-post activity creates eligibility traces (molecular tags)
        # - Eligibility alone does NOT cause weight change
        # - Dopamine arriving later converts eligibility to plasticity:
        #   * DA burst (reward) → eligibility becomes LTP
        #   * DA dip (punishment) → eligibility becomes LTD
        #   * No DA → eligibility decays away, NO learning
        #
        # This is fundamentally different from cortical Hebbian learning
        # where plasticity happens automatically based on activity.
        # =================================================================

        # Apply three-factor update: eligibility × dopamine
        # Only when there's a dopamine signal (no baseline learning!)
        if mech_config.enable_feedforward_learning and abs(dopamine_level) > 0.01:
            old_weights = weights.detach().clone()
            weights = apply_dopamine_modulated_update(
                weights=weights,
                eligibility=eligibility.get(),
                dopamine=dopamine_level,
                learning_rate=args.supervised_lr,
                w_min=0.0,
                w_max=w_max,
                soft_bounds=True,
                stp_resources=stp_resources,
            )
            # Track weight changes
            if output_spikes.sum() > 0:
                diagnostics.record_weight_change("three_factor", old_weights, weights.detach())
            dopamine_history.append((t, dopamine_level))

        # Recurrent learning (same as exp2)
        if not in_gap:
            predictive_coding.accumulate_spikes(output_spikes)

        current_cycle = t // effective_cycle
        recurrent_weights, _ = predictive_coding.update_recurrent(
            t, recurrent_weights, current_cycle,
            w_min=0.0, w_max=1.5
        )

        # End of cycle updates
        if (t + 1) % effective_cycle == 0:
            cycle_num = (t + 1) // effective_cycle

            # Homeostasis
            current_rate_hz = neuron_spikes * (1000.0 / effective_cycle_ms)
            if args.neuron_model in ("conductance", "dendritic"):
                train_state.avg_firing_rate, train_state.g_tonic, train_state.g_inh_tonic = \
                    update_homeostatic_conductance_bidirectional(
                        current_rate=current_rate_hz,
                        avg_firing_rate=train_state.avg_firing_rate,
                        g_tonic=train_state.g_tonic,
                        g_inh_tonic=train_state.g_inh_tonic,
                        target_rate=target_firing_rate_hz,
                        tau=homeostatic_tau,
                        strength=homeostatic_strength_hz * 0.01,
                        exc_bounds=(0.0, 0.5),
                        inh_bounds=(0.0, 5.0),
                    )

            # BCM
            avg_activity_hz = float((neuron_spikes.sum() / n_output) * 1000.0 / effective_cycle_ms)
            bcm_threshold = update_bcm_threshold(
                bcm_threshold, avg_activity_hz, target_firing_rate_hz,
                tau=200.0, min_threshold=0.01, max_threshold=2.0,
            )

            # Save weight snapshot
            if cycle_num % 10 == 0:
                weight_history.append(weights.detach().clone().cpu().numpy())

    # End final cycle diagnostics
    diagnostics.record_eligibility(eligibility.get())
    diagnostics.end_cycle(n_cycles)

    training_time = time.perf_counter() - start_time

    # Final accuracy
    final_accuracy = np.mean(accuracy_history[-10:]) if accuracy_history else 0.0

    # Build expected winners array from target mapping for diagnostics
    # For each phase, which neuron SHOULD win?
    expected_winners = np.array([
        target_mapping.get_target_for_phase(phase) for phase in range(n_input)
    ])

    if verbose:
        print(f"\n  Training complete in {training_time:.1f}s")
        print(f"  Final accuracy (last 10 cycles): {final_accuracy:.1%}")

        # Print diagnostic summary with target mapping
        diagnostics.print_final_summary(expected_winners=expected_winners)

    # Analyze learned mapping
    diagonal_score, _, max_indices, _ = compute_paired_diagonal_score(weights)

    # Compute mapping accuracy (different from diagonal score!)
    # This checks if learned mapping matches TARGET mapping
    mapping_correct = 0
    for out_idx in range(n_output):
        learned_input = max_indices[out_idx]
        target_output = target_mapping.get_target_output(learned_input)
        if target_output == out_idx:
            mapping_correct += 1

    if verbose:
        print(f"  Mapping accuracy: {mapping_correct}/{n_output}")
        print(f"  Learned peaks: {max_indices}")

    # Recurrent analysis
    recurrent_analysis = analyze_recurrent_structure(recurrent_weights, args.pattern_type)

    return {
        "weights": weights,
        "initial_weights": initial_weights,
        "recurrent_weights": recurrent_weights,
        "accuracy_history": accuracy_history,
        "weight_history": weight_history,
        "final_accuracy": final_accuracy,
        "mapping_correct": mapping_correct,
        "max_indices": max_indices,
        "recurrent_analysis": recurrent_analysis,
        "training_time": training_time,
        "output_neurons": output_neurons,
        "net_config": net_config,
        "train_state": train_state,
        "synaptic_mechanisms": synaptic_mechanisms,
        "w_max": w_max,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(args=None) -> bool:
    """Run the supervised learning experiment."""
    if args is None:
        args = parse_args()

    # Device selection
    device = select_device(args.n_input + args.n_output, verbose=True)

    # Create mechanism config
    mech_config = create_mechanism_config(args)

    # Print header
    print_experiment_header(
        title="Experiment 3: Supervised Learning with Dopamine",
        extra_info={
            "Mapping type": args.mapping_type,
            "Dopamine burst/dip": f"{args.dopamine_burst}/{args.dopamine_dip}",
            "Eligibility tau": f"{args.eligibility_tau_ms}ms",
            "Supervised LR": args.supervised_lr,
            "Random permutation bias": args.diagonal_bias,
        }
    )

    # Create target mapping
    if args.mapping_type == "default":
        target_mapping = create_default_mapping(args.n_input, args.n_output)
    elif args.mapping_type == "shuffled":
        target_mapping = create_shuffled_mapping(args.n_input, args.n_output, args.shuffle_seed)
    else:  # reversed
        target_mapping = create_reversed_mapping(args.n_input, args.n_output)

    print(f"\nTarget mapping ({target_mapping.name}):")
    print("  Phase -> Target output:")
    for phase in range(min(10, args.n_output)):
        target = target_mapping.get_target_for_phase(phase)
        print(f"    Phase {phase} (inputs {phase*2},{phase*2+1}) -> Output {target}")

    # Run supervised training
    print("\n" + "=" * 60)
    print("SUPERVISED TRAINING")
    print("=" * 60)

    results = run_supervised_training(
        args=args,
        target_mapping=target_mapping,
        device=device,
        mech_config=mech_config,
        verbose=True,
    )

    # Recurrent analysis
    print("\nRecurrent Structure (Learned):")
    print_recurrent_analysis(results["recurrent_analysis"])

    # Test generalization
    print("\n" + "=" * 60)
    print("GENERALIZATION TEST")
    print("=" * 60)

    # Test on a single cycle to check learned mapping
    test_duration_ms = args.n_input * 8.0
    test_pattern = create_temporal_pattern(
        args.n_input, test_duration_ms, args.pattern_type,
        dt=DEFAULT_DT
    ).to(device)

    results["train_state"].reset_tracking()
    test_spikes, test_winners = forward_pattern(
        test_pattern, results["train_state"], results["net_config"],
        results["weights"], results["recurrent_weights"],
        results["output_neurons"],
        synaptic_mechanisms=results["synaptic_mechanisms"]
    )

    # Check test accuracy
    phase_duration = int(test_duration_ms / args.n_input / DEFAULT_DT)
    test_correct = 0
    test_total = 0

    for phase in range(args.n_input):
        t_start = phase * phase_duration
        t_end = (phase + 1) * phase_duration
        phase_spikes = test_spikes[t_start:t_end]

        if phase_spikes.ndim > 1:
            spike_counts = phase_spikes.sum(axis=0)
            if spike_counts.sum() > 0:
                winner = np.argmax(spike_counts)
                target = target_mapping.get_target_for_phase(phase)
                test_total += 1
                if winner == target:
                    test_correct += 1

    test_accuracy = test_correct / max(test_total, 1)
    print(f"  Test accuracy: {test_correct}/{test_total} = {test_accuracy:.1%}")

    # Success criteria
    criteria = [
        (f"Final training accuracy > 80% ({results['final_accuracy']:.1%})",
         results['final_accuracy'] > 0.80),
        (f"Test accuracy > 70% ({test_accuracy:.1%})",
         test_accuracy > 0.70),
        (f"Mapping learned ({results['mapping_correct']}/{args.n_output} correct)",
         results['mapping_correct'] >= args.n_output * 0.8),
        (f"Recurrent chain learned ({results['recurrent_analysis'].correct_count}/5)",
         results['recurrent_analysis'].correct_count >= 3),
    ]

    all_passed = print_success_criteria(criteria)

    # Visualization
    if not args.no_plot:
        # Create expected mapping based on target
        expected_inputs = []
        for out_idx in range(args.n_output):
            # Find which input this output should respond to
            for inp in range(args.n_input):
                if target_mapping.get_target_output(inp) == out_idx:
                    expected_inputs.append(inp)
                    break
            else:
                expected_inputs.append(-1)

        fig, axes = create_training_summary_figure(
            initial_weights=results["initial_weights"],
            final_weights=results["weights"],
            recurrent_weights=results["recurrent_weights"],
            spike_counts=[int(a * 20) for a in results["accuracy_history"]],  # Scale for viz
            weight_history=results["weight_history"],
            title=f"Exp3: Supervised Learning ({args.mapping_type} mapping)",
            expected_mapping=np.array(expected_inputs),
        )

        # Save figure
        output_dir = get_results_dir()
        output_path = output_dir / f"exp3_supervised_{args.mapping_type}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        plt.show()

    return all_passed


if __name__ == "__main__":
    args = parse_args()
    success = run_experiment(args)
    exit(0 if success else 1)
