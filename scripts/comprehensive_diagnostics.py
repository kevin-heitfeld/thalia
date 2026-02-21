"""
Comprehensive Brain Diagnostics

This script consolidates all diagnostic analyses into one comprehensive run:
1. Brain-wide activity patterns and spike flow
2. Oscillation analysis - frequency spectrum and autocorrelation
3. Septum pacemaker rhythm analysis
4. Neuron-level vs population-level frequency
5. Hippocampus inhibition verification
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from thalia.brain import BrainBuilder, DynamicBrain
from thalia.components.synapses import WeightInitializer
from thalia.diagnostics import BrainActivityAnalyzer, BrainActivityReport
from thalia.typing import BrainOutput


# NOTE: Enable line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)


class ComprehensiveDiagnostics:
    """Comprehensive brain diagnostics during a single simulation run.

    Uses BrainActivityAnalyzer for brain-wide metrics + callbacks for specialized tracking.
    """

    def __init__(self, brain: DynamicBrain, timesteps: int, input_pattern: str = "random"):
        self.brain = brain
        self.timesteps = timesteps
        self.input_pattern = input_pattern

        # Use existing BrainActivityAnalyzer for comprehensive brain-wide analysis
        self.brain_analyzer = BrainActivityAnalyzer(brain=brain, timesteps=timesteps)
        self.brain_report: Optional[BrainActivityReport] = None

        # Septum-specific tracking
        septum = brain.regions["medial_septum"]
        if septum:
            self.septum_ach_trace = np.zeros(timesteps)
            self.septum_gaba_trace = np.zeros(timesteps)
            self.gaba_spike_times: List[List[int]] = [[] for _ in range(septum.gaba_size)]
        else:
            self.septum_ach_trace = None
            self.septum_gaba_trace = None
            self.gaba_spike_times = None

        # Hippocampus-specific tracking
        hippocampus = brain.regions["hippocampus"]
        if hippocampus:
            self.ca3_firing_rates: List[float] = []
        else:
            self.ca3_firing_rates = None

        # SNR-specific tracking (basal ganglia output)
        substantia_nigra = brain.regions["substantia_nigra"]
        if substantia_nigra:
            self.snr_firing_trace = np.zeros(timesteps)
            self.snr_membrane_trace = np.zeros((timesteps, substantia_nigra.vta_feedback_size))
            self.snr_d1_input_trace = np.zeros(timesteps)
            self.snr_d2_input_trace = np.zeros(timesteps)
        else:
            self.snr_firing_trace = None
            self.snr_membrane_trace = None
            self.snr_d1_input_trace = None
            self.snr_d2_input_trace = None

        # Striatum pathway tracking
        striatum = brain.regions["striatum"]
        if striatum:
            self.d1_firing_trace = np.zeros(timesteps)
            self.d2_firing_trace = np.zeros(timesteps)
        else:
            self.d1_firing_trace = None
            self.d2_firing_trace = None

        # Thalamus T-channel diagnostics (CRITICAL for debugging alpha oscillations)
        thalamus = brain.regions["thalamus"]
        if thalamus and hasattr(thalamus, "relay_neurons"):
            # Sample 10 relay neurons for detailed traces
            self.n_relay_samples = min(10, thalamus.relay_size)
            self.relay_voltage_traces = np.zeros((timesteps, self.n_relay_samples))
            self.relay_h_T_traces = np.zeros((timesteps, self.n_relay_samples))
            self.relay_spikes_traces = np.zeros((timesteps, self.n_relay_samples), dtype=bool)
            self.relay_g_inh_traces = np.zeros((timesteps, self.n_relay_samples))
            self.relay_g_exc_traces = np.zeros((timesteps, self.n_relay_samples))
            # Sample 5 TRN neurons
            self.n_trn_samples = min(5, thalamus.trn_size)
            self.trn_voltage_traces = np.zeros((timesteps, self.n_trn_samples))
            self.trn_spikes_traces = np.zeros((timesteps, self.n_trn_samples), dtype=bool)
        else:
            self.n_relay_samples = 0
            self.relay_voltage_traces = None
            self.relay_h_T_traces = None
            self.relay_spikes_traces = None
            self.relay_g_inh_traces = None
            self.relay_g_exc_traces = None
            self.n_trn_samples = 0
            self.trn_voltage_traces = None
            self.trn_spikes_traces = None

        # Cortex L4 inhibitory network tracking
        cortex = brain.regions["cortex"]
        if cortex and hasattr(cortex, "l4_inhibitory"):
            self.l4_pyr_firing_trace = np.zeros(timesteps)
            self.l4_pv_firing_trace = np.zeros(timesteps)
        else:
            self.l4_pyr_firing_trace = None
            self.l4_pv_firing_trace = None

        # Homeostatic gain tracking (sample every 10ms)
        self.gain_traces: Dict[str, List[float]] = defaultdict(list)

        # Delta synchrony investigation: Per-neuron spike time tracking for ISI analysis
        self.spike_times: Dict[str, Dict[str, List[List[int]]]] = {}  # region -> pop -> neuron -> [times]

        # Population firing rates (10ms bins) for FFT analysis
        self.bin_size_ms = 10.0
        self.n_bins = int((timesteps * brain.dt_ms) / self.bin_size_ms)
        self.population_rates: Dict[str, np.ndarray] = {}  # region:pop -> [n_bins]

        # Conductance samples for phase diagram analysis
        self.conductance_samples: Dict[str, List[Dict[str, float]]] = {}  # region:pop -> [{g_exc, g_inh, ...}]

        # Cross-correlation tracking
        self.region_spike_counts = np.zeros((timesteps, len(brain.regions)))  # [time, region]
        self.region_names = list(brain.regions.keys())

    def _timestep_callback(self, timestep: int, outputs: BrainOutput) -> None:
        """Callback invoked during simulation to collect specialized diagnostics."""
        # Track per-neuron spike times for ISI analysis
        for region_idx, (region_name, region_outputs) in enumerate(outputs.items()):
            if region_name not in self.spike_times:
                self.spike_times[region_name] = {}

            for pop_name, pop_spikes in region_outputs.items():
                if pop_name not in self.spike_times[region_name]:
                    # Initialize spike time lists for this population
                    pop_size = pop_spikes.numel()
                    self.spike_times[region_name][pop_name] = [[] for _ in range(pop_size)]

                # Record spike times
                spiking_neurons = torch.where(pop_spikes)[0]
                for neuron_idx in spiking_neurons.tolist():
                    self.spike_times[region_name][pop_name][neuron_idx].append(timestep)

            # Track region-level spike counts for cross-correlation
            total_spikes = sum(pop_spikes.sum().item() for pop_spikes in region_outputs.values())
            self.region_spike_counts[timestep, region_idx] = total_spikes

        # Track population firing rates in 10ms bins for FFT analysis
        bin_idx = int(timestep // (self.bin_size_ms / self.brain.dt_ms))
        if bin_idx < self.n_bins:
            for region_name, region_outputs in outputs.items():
                for pop_name, pop_spikes in region_outputs.items():
                    key = f"{region_name}:{pop_name}"
                    if key not in self.population_rates:
                        self.population_rates[key] = np.zeros(self.n_bins)
                    self.population_rates[key][bin_idx] += pop_spikes.sum().item()

        # Sample conductances every 50ms for phase diagram analysis
        if timestep % 50 == 0:
            # Sample relay neurons from thalamus
            if "thalamus" in self.brain.regions:
                thalamus = self.brain.regions["thalamus"]
                if hasattr(thalamus, "relay_neurons"):
                    relay = thalamus.relay_neurons
                    if relay.g_E is not None and relay.g_I is not None:
                        key = "thalamus:relay"
                        if key not in self.conductance_samples:
                            self.conductance_samples[key] = []
                        self.conductance_samples[key].append({
                            "g_exc": float(relay.g_E.mean().item()),
                            "g_inh": float(relay.g_I.mean().item()),
                            "g_nmda": float(relay.g_nmda.mean().item()) if relay.g_nmda is not None else 0.0,
                            "v_mem": float(relay.membrane.mean().item()) if relay.membrane is not None else 0.0,
                        })

            # Sample cortical L4 neurons
            if "cortex" in self.brain.regions:
                cortex = self.brain.regions["cortex"]
                if hasattr(cortex, "l4_neurons"):
                    l4 = cortex.l4_neurons
                    if l4.g_E is not None and l4.g_I is not None:
                        key = "cortex:l4"
                        if key not in self.conductance_samples:
                            self.conductance_samples[key] = []
                        self.conductance_samples[key].append({
                            "g_exc": float(l4.g_E.mean().item()),
                            "g_inh": float(l4.g_I.mean().item()),
                            "g_nmda": float(l4.g_nmda.mean().item()) if l4.g_nmda is not None else 0.0,
                            "v_mem": float(l4.membrane.mean().item()) if l4.membrane is not None else 0.0,
                        })

        # Track septum activity (original tracking code continues below)
        if self.septum_ach_trace is not None:
            septum_output = outputs.get("medial_septum", {})
            septum_ach = septum_output.get("ach", torch.zeros(200, dtype=torch.bool))
            septum_gaba = septum_output.get("gaba", torch.zeros(200, dtype=torch.bool))

            self.septum_ach_trace[timestep] = septum_ach.sum().item()
            self.septum_gaba_trace[timestep] = septum_gaba.sum().item()

            # Track individual GABA neuron spike times
            spiking_neurons = torch.where(septum_gaba)[0]
            for neuron_idx in spiking_neurons.tolist():
                self.gaba_spike_times[neuron_idx].append(timestep)

        # Track hippocampus CA3 firing rate
        if self.ca3_firing_rates is not None:
            hippocampus = self.brain.regions["hippocampus"]
            if hippocampus:
                ca3_spikes = hippocampus._ca3_spike_buffer.read(0)
                self.ca3_firing_rates.append(ca3_spikes.float().mean().item())

        # Track SNR activity (basal ganglia output for TD learning)
        if self.snr_firing_trace is not None:
            substantia_nigra = self.brain.regions["substantia_nigra"]
            snr_output = outputs.get("substantia_nigra", {})
            snr_spikes = snr_output.get("vta_feedback", torch.zeros(substantia_nigra.vta_feedback_size, dtype=torch.bool))

            self.snr_firing_trace[timestep] = snr_spikes.sum().item()

            # Get membrane potentials if available
            if substantia_nigra._current_output is not None:
                _, membrane = substantia_nigra._current_output
                self.snr_membrane_trace[timestep, :] = membrane.detach().cpu().numpy()

            # Track striatal inputs (from previous timestep outputs)
            striatum_output = outputs.get("striatum", {})
            d1_spikes = striatum_output.get("d1", torch.zeros(200, dtype=torch.bool))
            d2_spikes = striatum_output.get("d2", torch.zeros(200, dtype=torch.bool))
            self.snr_d1_input_trace[timestep] = d1_spikes.sum().item()
            self.snr_d2_input_trace[timestep] = d2_spikes.sum().item()

        # Track striatum D1/D2 pathways
        if self.d1_firing_trace is not None:
            striatum_output = outputs.get("striatum", {})
            d1_spikes = striatum_output.get("d1", torch.zeros(200, dtype=torch.bool))
            d2_spikes = striatum_output.get("d2", torch.zeros(200, dtype=torch.bool))
            self.d1_firing_trace[timestep] = d1_spikes.sum().item()
            self.d2_firing_trace[timestep] = d2_spikes.sum().item()

        # Track cortex L4 inhibitory network
        if self.l4_pyr_firing_trace is not None:
            cortex = self.brain.regions["cortex"]
            cortex_output = outputs.get("cortex", {})
            l4_spikes = cortex_output.get("l4", torch.zeros(cortex.l4_pyr_size, dtype=torch.bool))
            self.l4_pyr_firing_trace[timestep] = l4_spikes.sum().item()

            # Get PV interneuron activity (from inhibitory network if available)
            if hasattr(cortex, "_l4_pv_spikes"):
                self.l4_pv_firing_trace[timestep] = cortex._l4_pv_spikes.sum().item()

        # Track thalamus T-channel state (CRITICAL for debugging alpha oscillations!)
        if self.relay_voltage_traces is not None:
            thalamus = self.brain.regions["thalamus"]
            # Sample relay neurons (first N neurons)
            relay_neurons = thalamus.relay_neurons
            if relay_neurons.membrane is not None:
                self.relay_voltage_traces[timestep, :] = relay_neurons.membrane[:self.n_relay_samples].detach().cpu().numpy()

            # Track h_T de-inactivation state
            if hasattr(relay_neurons, 'h_T') and relay_neurons.h_T is not None:
                self.relay_h_T_traces[timestep, :] = relay_neurons.h_T[:self.n_relay_samples].detach().cpu().numpy()

            # Track relay spikes
            thalamus_output = outputs.get("thalamus", {})
            relay_spikes = thalamus_output.get("relay", torch.zeros(thalamus.relay_size, dtype=torch.bool))
            self.relay_spikes_traces[timestep, :] = relay_spikes[:self.n_relay_samples].cpu().numpy()

            # Track conductances (excitatory and inhibitory)
            if relay_neurons.g_E is not None:
                self.relay_g_exc_traces[timestep, :] = relay_neurons.g_E[:self.n_relay_samples].detach().cpu().numpy()
            if relay_neurons.g_I is not None:
                self.relay_g_inh_traces[timestep, :] = relay_neurons.g_I[:self.n_relay_samples].detach().cpu().numpy()

            # Sample TRN neurons
            trn_neurons = thalamus.trn_neurons
            if trn_neurons.membrane is not None:
                self.trn_voltage_traces[timestep, :] = trn_neurons.membrane[:self.n_trn_samples].detach().cpu().numpy()
            trn_spikes = thalamus_output.get("trn", torch.zeros(thalamus.trn_size, dtype=torch.bool))
            self.trn_spikes_traces[timestep, :] = trn_spikes[:self.n_trn_samples].cpu().numpy()

        # Sample homeostatic gains (g_L_scale) every 10ms
        if timestep % 10 == 0:
            # Cortex layers
            if 'cortex' in self.brain.regions:
                cortex = self.brain.regions['cortex']
                if hasattr(cortex, 'l23_neurons'):
                    self.gain_traces['cortex:l23'].append(float(cortex.l23_neurons.g_L_scale.mean().item()))
                if hasattr(cortex, 'l4_neurons'):
                    self.gain_traces['cortex:l4'].append(float(cortex.l4_neurons.g_L_scale.mean().item()))
                if hasattr(cortex, 'l5_neurons'):
                    self.gain_traces['cortex:l5'].append(float(cortex.l5_neurons.g_L_scale.mean().item()))
                if hasattr(cortex, 'l6a_neurons'):
                    self.gain_traces['cortex:l6a'].append(float(cortex.l6a_neurons.g_L_scale.mean().item()))
                if hasattr(cortex, 'l6b_neurons'):
                    self.gain_traces['cortex:l6b'].append(float(cortex.l6b_neurons.g_L_scale.mean().item()))

            # Thalamus
            if 'thalamus' in self.brain.regions:
                thalamus = self.brain.regions['thalamus']
                if hasattr(thalamus, 'relay_neurons'):
                    self.gain_traces['thalamus:relay'].append(float(thalamus.relay_neurons.g_L_scale.mean().item()))
                if hasattr(thalamus, 'trn_neurons'):
                    self.gain_traces['thalamus:trn'].append(float(thalamus.trn_neurons.g_L_scale.mean().item()))

            # Hippocampus
            if 'hippocampus' in self.brain.regions:
                hippocampus = self.brain.regions['hippocampus']
                if hasattr(hippocampus, 'dg_neurons'):
                    self.gain_traces['hippocampus:dg'].append(float(hippocampus.dg_neurons.g_L_scale.mean().item()))
                if hasattr(hippocampus, 'ca3_neurons'):
                    self.gain_traces['hippocampus:ca3'].append(float(hippocampus.ca3_neurons.g_L_scale.mean().item()))
                if hasattr(hippocampus, 'ca1_neurons'):
                    self.gain_traces['hippocampus:ca1'].append(float(hippocampus.ca1_neurons.g_L_scale.mean().item()))

            # Striatum
            if 'striatum' in self.brain.regions:
                striatum = self.brain.regions['striatum']
                if hasattr(striatum, 'd1_pathway') and hasattr(striatum.d1_pathway, 'neurons'):
                    self.gain_traces['striatum:d1'].append(float(striatum.d1_pathway.neurons.g_L_scale.mean().item()))
                if hasattr(striatum, 'd2_pathway') and hasattr(striatum.d2_pathway, 'neurons'):
                    self.gain_traces['striatum:d2'].append(float(striatum.d2_pathway.neurons.g_L_scale.mean().item()))

            # PFC
            if 'prefrontal' in self.brain.regions:
                prefrontal = self.brain.regions['prefrontal']
                if hasattr(prefrontal, 'neurons'):
                    self.gain_traces['prefrontal:executive'].append(float(prefrontal.neurons.g_L_scale.mean().item()))

    def run_simulation_with_callbacks(self) -> None:
        """Run single simulation with BrainActivityAnalyzer plus specialized tracking via callbacks."""
        print("\n" + "="*80)
        print(f"RUNNING {self.timesteps}ms SIMULATION WITH COMPREHENSIVE DIAGNOSTICS")
        print("="*80)
        print(f"\nTracking {len(self.brain.regions)} regions with {self.timesteps} timesteps")
        print()

        # Monkey-patch the analyzer to call our callback during simulation
        original_run_sim = self.brain_analyzer.run_simulation

        def run_with_callback(input_pattern=None, verbose=True):
            """Wrapper around run_simulation that injects our callback."""
            if verbose:
                print(f"\n{'='*80}")
                print(f"BRAIN ACTIVITY ANALYZER - Simulating {self.brain_analyzer.timesteps} timesteps")
                print(f"{'='*80}\n")

            self.brain_analyzer.spike_history = []
            self.brain_analyzer.region_activity = defaultdict(list)

            start_time = time.time()

            for t in range(self.brain_analyzer.timesteps):
                # Generate input
                input_spikes = self.brain_analyzer._generate_input(t, input_pattern)

                # Run brain forward pass
                outputs = self.brain.forward(input_spikes)

                # Record outputs (original behavior)
                self.brain_analyzer.spike_history.append(outputs)
                for region_name, region_outputs in outputs.items():
                    self.brain_analyzer.region_activity[region_name].append(region_outputs)

                # Track gains and weights if enabled
                if self.brain_analyzer.track_dynamics:
                    self.brain_analyzer._record_timestep_diagnostics(t)

                # INJECT OUR CALLBACK HERE
                self._timestep_callback(t, outputs)

                # Progress updates
                if verbose and (t + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (t + 1) / elapsed
                    print(f"  Timestep {t+1}/{self.brain_analyzer.timesteps} ({rate:.1f} steps/sec)")

            if verbose:
                elapsed = time.time() - start_time
                print(f"\n✓ Simulation complete in {elapsed:.2f}s ({self.brain_analyzer.timesteps/elapsed:.1f} steps/sec)")

        # Replace method temporarily
        self.brain_analyzer.run_simulation = run_with_callback

        # Run the analysis (now with our callbacks)
        print(f"Running brain-wide activity analysis with specialized tracking...")
        print(f"Input pattern: {self.input_pattern}")
        self.brain_report = self.brain_analyzer.analyze_full_brain(
            input_pattern=self.input_pattern,
            verbose=True
        )

        # Restore original method
        self.brain_analyzer.run_simulation = original_run_sim

        print("\n✓ Comprehensive analysis complete")

    def print_brain_activity_report(self) -> None:
        """Print the comprehensive brain activity report from BrainActivityAnalyzer."""
        if self.brain_report is None:
            print("\n⚠️  No brain activity report available")
            return

        self.brain_analyzer.print_report(self.brain_report, detailed=True)

    def analyze_septum_rhythm(self) -> None:
        """Analyze medial septum pacemaker rhythm."""
        if self.septum_ach_trace is None:
            print("\n⚠️  Medial septum not found - skipping septum analysis")
            return

        print("\n" + "="*80)
        print("MEDIAL SEPTUM PACEMAKER ANALYSIS")
        print("="*80)

        # ACh activity
        print(f"\nACh cholinergic activity:")
        print(f"  Mean spikes/ms: {self.septum_ach_trace.mean():.2f}")
        print(f"  Max spikes/ms: {self.septum_ach_trace.max():.0f}")

        # GABA activity
        print(f"\nGABA rhythmic activity:")
        print(f"  Mean spikes/ms: {self.septum_gaba_trace.mean():.2f}")
        print(f"  Max spikes/ms: {self.septum_gaba_trace.max():.0f}")

        # Population-level frequency (FFT on spike counts)
        gaba_fft = np.fft.rfft(self.septum_gaba_trace - self.septum_gaba_trace.mean())
        freqs = np.fft.rfftfreq(len(self.septum_gaba_trace), d=0.001)
        gaba_power = np.abs(gaba_fft) ** 2
        gaba_peak_idx = np.argmax(gaba_power[1:]) + 1
        gaba_population_freq = freqs[gaba_peak_idx]

        print(f"\n  Population-level oscillation: {gaba_population_freq:.1f} Hz")

        # Individual neuron ISI analysis
        neuron_isis = []
        neuron_frequencies = []

        septum = self.brain.regions["medial_septum"]
        for neuron_idx in range(septum.gaba_size):
            spike_times = self.gaba_spike_times[neuron_idx]

            if len(spike_times) >= 3:
                spike_times_array = np.array(spike_times)
                isis = np.diff(spike_times_array)
                neuron_isis.extend(isis.tolist())

                mean_isi = isis.mean()
                if mean_isi > 0:
                    neuron_freq = 1000.0 / mean_isi
                    neuron_frequencies.append(neuron_freq)

        if len(neuron_frequencies) > 0:
            neuron_isis_array = np.array(neuron_isis)
            neuron_frequencies_array = np.array(neuron_frequencies)

            print(f"\n  Individual neuron analysis:")
            print(f"    Neurons with ≥3 spikes: {len(neuron_frequencies)} / {septum.gaba_size}")
            print(f"    Mean ISI: {neuron_isis_array.mean():.1f} ms")
            print(f"    Mean firing freq: {neuron_frequencies_array.mean():.1f} Hz")
            print(f"    Frequency range: [{neuron_frequencies_array.min():.1f}, {neuron_frequencies_array.max():.1f}] Hz")

            # Diagnosis
            print(f"\n  Pacemaker rhythm assessment:")
            if neuron_frequencies_array.mean() < 12:  # Within 8 Hz ± 50%
                print(f"    ✓ Individual neurons fire at ~{neuron_frequencies_array.mean():.1f} Hz (close to 8 Hz target)")
                print(f"    ✓ Population shows {gaba_population_freq:.1f} Hz (measurement artifact from phase distribution)")
                print(f"    ✓ Pacemaker functioning correctly")
            else:
                print(f"    ⚠️ Individual neurons fire at ~{neuron_frequencies_array.mean():.1f} Hz (too fast!)")
                print(f"    ⚠️ Expected ~8 Hz for theta pacemaker")

    def analyze_hippocampus_inhibition(self) -> None:
        """Analyze hippocampus inhibitory network strength."""
        if self.ca3_firing_rates is None:
            print("\n⚠️  Hippocampus not found - skipping inhibition analysis")
            return

        print("\n" + "="*80)
        print("HIPPOCAMPUS INHIBITION ANALYSIS")
        print("="*80)

        hippocampus = self.brain.regions["hippocampus"]

        # CA3 firing rate
        mean_ca3_rate = np.mean(self.ca3_firing_rates) * 100
        print(f"\nCA3 pyramidal activity:")
        print(f"  Mean firing rate: {mean_ca3_rate:.2f}%")

        # Inhibitory network structure
        print(f"\nInhibitory network structure:")
        print(f"  CA3 pyramidal neurons: {hippocampus.ca3_size}")
        print(f"  CA3 PV neurons: {hippocampus.ca3_inhibitory.n_pv}")
        print(f"  CA3 OLM neurons: {hippocampus.ca3_inhibitory.n_olm}")
        print(f"  CA3 Bistratified: {hippocampus.ca3_inhibitory.n_bistratified}")

        # Inhibitory weights
        pv_to_pyr = hippocampus.ca3_inhibitory.pv_to_pyr
        pyr_to_pv = hippocampus.ca3_inhibitory.pyr_to_pv

        print(f"\nInhibitory weight statistics:")
        print(f"  PV → Pyramidal:")
        print(f"    Mean: {pv_to_pyr.mean():.4f}")
        print(f"    Max: {pv_to_pyr.max():.4f}")
        print(f"    Connectivity: {(pv_to_pyr > 0).float().mean():.2%}")
        print(f"  Pyramidal → PV:")
        print(f"    Mean: {pyr_to_pv.mean():.4f}")
        print(f"    Max: {pyr_to_pv.max():.4f}")
        print(f"    Connectivity: {(pyr_to_pv > 0).float().mean():.2%}")

        # Diagnosis
        print(f"\n  Inhibition strength assessment:")
        if pv_to_pyr.mean() < 0.03:
            print(f"    ⚠️ PV→Pyr weights very weak (< 0.03)")
            print(f"    ⚠️ Insufficient inhibitory conductance")
        elif mean_ca3_rate > 10:
            print(f"    ⚠️ CA3 firing rate > 10% indicates hyperactivity")
            print(f"    ⚠️ Check if inhibitory neurons are firing")
        else:
            print(f"    ✓ Inhibitory weights appear adequate")
            print(f"    ✓ CA3 activity within expected range")

    def analyze_snr_basal_ganglia(self) -> None:
        """Analyze SNR (substantia nigra reticulata) basal ganglia output."""
        if self.snr_firing_trace is None:
            print("\n⚠️  SNR not found - skipping basal ganglia analysis")
            return

        print("\n" + "="*80)
        print("SNR BASAL GANGLIA OUTPUT ANALYSIS")
        print("="*80)

        substantia_nigra = self.brain.regions["substantia_nigra"]

        # SNR firing statistics
        total_snr_spikes = self.snr_firing_trace.sum()
        mean_snr_rate = (total_snr_spikes / (substantia_nigra.vta_feedback_size * self.timesteps)) * 100
        mean_spikes_per_ms = self.snr_firing_trace.mean()

        print(f"\nSNR tonic activity:")
        print(f"  Mean firing rate: {mean_snr_rate:.2f}%")
        print(f"  Mean spikes/ms: {mean_spikes_per_ms:.2f}")
        print(f"  Total spikes: {int(total_snr_spikes)}")
        print(f"  Max spikes/ms: {self.snr_firing_trace.max():.0f}")

        # Striatal input statistics
        mean_d1_input = self.snr_d1_input_trace.mean()
        mean_d2_input = self.snr_d2_input_trace.mean()

        print(f"\nStriatal input activity:")
        print(f"  D1 (inhibitory) mean spikes/ms: {mean_d1_input:.2f}")
        print(f"  D2 (excitatory) mean spikes/ms: {mean_d2_input:.2f}")
        print(f"  D1/D2 ratio: {mean_d1_input / mean_d2_input if mean_d2_input > 0 else float('inf'):.2f}")

        # Membrane potential statistics (if available)
        if self.snr_membrane_trace is not None and self.snr_membrane_trace.any():
            mean_v_mem = np.mean(self.snr_membrane_trace, axis=1)
            print(f"\nMembrane dynamics:")
            print(f"  Mean V_mem: {mean_v_mem.mean():.3f}")
            print(f"  V_mem std: {mean_v_mem.std():.3f}")
            print(f"  V_mem range: [{mean_v_mem.min():.3f}, {mean_v_mem.max():.3f}]")

        # Diagnosis
        print(f"\n  SNR health assessment:")
        if mean_snr_rate < 0.1:
            print(f"    🔴 CRITICAL: SNR essentially silent ({mean_snr_rate:.3f}%)")
            print(f"    🔴 No value signal for TD learning")
            print(f"    🔴 Check: baseline_drive too weak or inhibition too strong")
        elif mean_snr_rate < 3.0:
            print(f"    ⚠️  SNR firing low ({mean_snr_rate:.2f}%, expected 5-7%)")
            print(f"    ⚠️  Striatal inhibition may be too strong")
        elif mean_snr_rate > 10.0:
            print(f"    ⚠️  SNR firing high ({mean_snr_rate:.2f}%, expected 5-7%)")
            print(f"    ⚠️  Baseline drive or D2 excitation may be too strong")
        else:
            print(f"    ✓ SNR tonic firing in biological range (5-7%)")
            print(f"    ✓ Value signal available for VTA")

        # Check striatal modulation
        if mean_d1_input > mean_d2_input * 2:
            print(f"    ⚠️  D1 pathway dominant - SNR should be suppressed")
        elif mean_d2_input > mean_d1_input * 2:
            print(f"    ⚠️  D2 pathway dominant - SNR should be elevated")
        else:
            print(f"    ✓ Balanced D1/D2 pathway activity")

    def analyze_striatum_pathways(self) -> None:
        """Analyze striatum D1/D2 pathway balance."""
        if self.d1_firing_trace is None:
            print("\n⚠️  Striatum not found - skipping pathway analysis")
            return

        print("\n" + "="*80)
        print("STRIATUM D1/D2 PATHWAY ANALYSIS")
        print("="*80)

        # D1 pathway (direct, "Go")
        d1_mean_rate = (self.d1_firing_trace.sum() / (200 * self.timesteps)) * 100
        d1_mean_spikes = self.d1_firing_trace.mean()

        print(f"\nD1 pathway (direct, Go):")
        print(f"  Mean firing rate: {d1_mean_rate:.2f}%")
        print(f"  Mean spikes/ms: {d1_mean_spikes:.2f}")

        # D2 pathway (indirect, "NoGo")
        d2_mean_rate = (self.d2_firing_trace.sum() / (200 * self.timesteps)) * 100
        d2_mean_spikes = self.d2_firing_trace.mean()

        print(f"\nD2 pathway (indirect, NoGo):")
        print(f"  Mean firing rate: {d2_mean_rate:.2f}%")
        print(f"  Mean spikes/ms: {d2_mean_spikes:.2f}")

        # Pathway balance
        d1_d2_ratio = d1_mean_rate / d2_mean_rate if d2_mean_rate > 0 else float('inf')
        print(f"\nPathway balance:")
        print(f"  D1/D2 ratio: {d1_d2_ratio:.2f}")

        # Diagnosis
        print(f"\n  Pathway health assessment:")
        if d1_mean_rate < 1.0 or d2_mean_rate < 1.0:
            print(f"    🔴 CRITICAL: Striatal pathway(s) nearly silent")
            print(f"    🔴 No action selection signal")
        elif abs(d1_d2_ratio - 1.0) > 0.5:
            print(f"    ⚠️  Pathway imbalance (D1/D2 = {d1_d2_ratio:.2f})")
            print(f"    ⚠️  Expected approximately balanced (~1.0)")
        else:
            print(f"    ✓ Balanced D1/D2 pathway activity")
            print(f"    ✓ Both pathways active for action selection")

    def analyze_cortex_l4_inhibition(self) -> None:
        """Analyze cortex L4 inhibitory network."""
        if self.l4_pyr_firing_trace is None:
            print("\n⚠️  Cortex L4 not found - skipping L4 inhibition analysis")
            return

        print("\n" + "="*80)
        print("CORTEX L4 INHIBITORY NETWORK ANALYSIS")
        print("="*80)

        cortex = self.brain.regions["cortex"]

        # L4 pyramidal activity
        l4_pyr_rate = (self.l4_pyr_firing_trace.sum() / (cortex.l4_pyr_size * self.timesteps)) * 100
        l4_pyr_spikes_per_ms = self.l4_pyr_firing_trace.mean()

        print(f"\nL4 pyramidal neurons:")
        print(f"  Mean firing rate: {l4_pyr_rate:.2f}%")
        print(f"  Mean spikes/ms: {l4_pyr_spikes_per_ms:.2f}")
        print(f"  Total neurons: {cortex.l4_pyr_size}")

        # PV interneuron activity (if available)
        if self.l4_pv_firing_trace is not None and self.l4_pv_firing_trace.sum() > 0:
            l4_pv_size = cortex.l4_inhibitory.pv_size
            l4_pv_rate = (self.l4_pv_firing_trace.sum() / (l4_pv_size * self.timesteps)) * 100
            l4_pv_spikes_per_ms = self.l4_pv_firing_trace.mean()

            print(f"\nL4 PV interneurons:")
            print(f"  Mean firing rate: {l4_pv_rate:.2f}%")
            print(f"  Mean spikes/ms: {l4_pv_spikes_per_ms:.2f}")
            print(f"  Total PV neurons: {l4_pv_size}")

            # E/I ratio
            e_i_ratio = l4_pyr_spikes_per_ms / l4_pv_spikes_per_ms if l4_pv_spikes_per_ms > 0 else float('inf')
            print(f"\nExcitatory/Inhibitory balance:")
            print(f"  E/I ratio: {e_i_ratio:.2f}")
        else:
            print(f"\n⚠️  PV interneuron activity not tracked")

        # Diagnosis
        print(f"\n  L4 health assessment:")
        if l4_pyr_rate > 10.0:
            print(f"    🔴 CRITICAL: L4 hyperactive ({l4_pyr_rate:.2f}%, expected 1-3%)")
            print(f"    🔴 Check: input weights too strong or inhibition too weak")
            if self.l4_pv_firing_trace is None or self.l4_pv_firing_trace.sum() == 0:
                print(f"    🔴 PV interneurons may not be active")
        elif l4_pyr_rate > 5.0:
            print(f"    ⚠️  L4 elevated ({l4_pyr_rate:.2f}%, expected 1-3%)")
            print(f"    ⚠️  Consider: increasing threshold or strengthening inhibition")
        else:
            print(f"    ✓ L4 firing rate in biological range (1-3%)")
            print(f"    ✓ Sparse sensory representation")

    def analyze_thalamus_t_channels(self) -> None:
        """Analyze thalamic T-channel dynamics and rebound bursting."""
        if self.relay_voltage_traces is None:
            print("\n⚠️  Thalamus T-channel tracking not enabled - skipping")
            return

        print("\n" + "="*80)
        print("THALAMUS T-CHANNEL & REBOUND BURST ANALYSIS")
        print("="*80)

        print(f"\n  Tracking {self.n_relay_samples} sample relay neurons + {self.n_trn_samples} TRN neurons")

        # Voltage statistics for relay neurons
        v_mean = self.relay_voltage_traces.mean()
        v_std = self.relay_voltage_traces.std()
        v_min = self.relay_voltage_traces.min()
        v_max = self.relay_voltage_traces.max()

        print(f"\nRelay neuron voltage dynamics:")
        print(f"  Mean V: {v_mean:.3f}")
        print(f"  Std V: {v_std:.3f}")
        print(f"  Range: [{v_min:.3f}, {v_max:.3f}]")
        print(f"  V_half_h_T threshold: -0.3 (hyperpolarization needed for T-channel de-inactivation)")

        # Check if neurons reach hyperpolarization threshold
        hyper_threshold = -0.3
        relay_v_flat = self.relay_voltage_traces.flatten()
        hyperpolarized_fraction = (relay_v_flat < hyper_threshold).sum() / len(relay_v_flat)
        print(f"  Time below V_half_h_T: {hyperpolarized_fraction*100:.1f}%")

        if hyperpolarized_fraction < 0.01:
            print(f"  🔴 CRITICAL: Relay neurons NOT reaching hyperpolarization threshold!")
            print(f"  🔴 T-channels cannot de-inactivate → no rebound bursts")
            print(f"  🔴 Solution: Increase TRN→relay inhibition or check inhibitory routing")
        elif hyperpolarized_fraction < 0.10:
            print(f"  ⚠️  Relay neurons rarely hyperpolarized (<10% of time)")
            print(f"  ⚠️  T-channel rebound bursts will be weak/infrequent")
        else:
            print(f"  ✓ Relay neurons reaching hyperpolarization (>10% of time)")

        # h_T de-inactivation statistics
        if self.relay_h_T_traces is not None and self.relay_h_T_traces.any():
            h_T_mean = self.relay_h_T_traces.mean()
            h_T_std = self.relay_h_T_traces.std()
            h_T_max = self.relay_h_T_traces.max()

            print(f"\nT-channel de-inactivation (h_T):")
            print(f"  Mean h_T: {h_T_mean:.3f}")
            print(f"  Std h_T: {h_T_std:.3f}")
            print(f"  Max h_T: {h_T_max:.3f}")
            print(f"  Note: h_T ≈ 1.0 = fully de-inactivated (ready for burst), h_T ≈ 0.0 = inactivated")

            # Check if h_T is responding to voltage
            if h_T_std < 0.05:
                print(f"  🔴 CRITICAL: h_T not varying (std={h_T_std:.3f})!")
                print(f"  🔴 T-channels may not be updating properly")
            elif h_T_mean < 0.2:
                print(f"  ⚠️  h_T very low (mean={h_T_mean:.3f}) - channels mostly inactivated")
                print(f"  ⚠️  Neurons may not be hyperpolarizing enough")
            elif h_T_mean > 0.8:
                print(f"  ⚠️  h_T very high (mean={h_T_mean:.3f}) - channels always de-inactivated")
                print(f"  ⚠️  May indicate neurons are too hyperpolarized (not spiking)")
            else:
                print(f"  ✓ h_T in reasonable range (0.2-0.8) - cycling between states")

        # Conductance balance (excitation vs inhibition)
        g_exc_mean = self.relay_g_exc_traces.mean()
        g_inh_mean = self.relay_g_inh_traces.mean()
        g_exc_max = self.relay_g_exc_traces.max()
        g_inh_max = self.relay_g_inh_traces.max()

        print(f"\nRelay conductance balance:")
        print(f"  Mean g_exc: {g_exc_mean:.3f}")
        print(f"  Mean g_inh: {g_inh_mean:.3f}")
        print(f"  Max g_exc: {g_exc_max:.3f}")
        print(f"  Max g_inh: {g_inh_max:.3f}")
        print(f"  g_inh/g_exc ratio: {g_inh_mean/g_exc_mean if g_exc_mean > 0 else 0:.2f}")

        if g_inh_mean < 0.05:
            print(f"  🔴 CRITICAL: Almost no inhibitory input (g_inh={g_inh_mean:.3f})!")
            print(f"  🔴 TRN→relay connection may be broken or weights too weak")
        elif g_inh_mean < g_exc_mean * 0.3:
            print(f"  ⚠️  Weak inhibition relative to excitation")
            print(f"  ⚠️  May not create sufficient hyperpolarization for T-channels")
        else:
            print(f"  ✓ Inhibition present and balanced with excitation")

        # Spike pattern analysis
        relay_spike_rates = self.relay_spikes_traces.mean(axis=0) * 1000  # Convert to Hz
        relay_fr_mean = relay_spike_rates.mean()
        relay_fr_std = relay_spike_rates.std()

        print(f"\nRelay neuron firing:")
        print(f"  Mean FR: {relay_fr_mean:.1f} Hz")
        print(f"  Std FR: {relay_fr_std:.1f} Hz")

        # Check for burst detection (simple: 2+ spikes within 10ms window)
        burst_count = 0
        for neuron_idx in range(self.n_relay_samples):
            spikes = self.relay_spikes_traces[:, neuron_idx]
            spike_times = np.where(spikes)[0]
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                bursts = (isis <= 10).sum()  # ISI <= 10ms = burst
                burst_count += bursts

        print(f"  Burst events detected: {burst_count} (ISI ≤ 10ms)")
        if burst_count == 0:
            print(f"  ⚠️  No burst firing detected - T-channel rebound may not be working")

        # TRN-relay oscillation analysis
        print(f"\nTRN-relay oscillation dynamics:")
        trn_spike_rate = self.trn_spikes_traces.mean() * 1000  # Hz
        print(f"  TRN firing rate: {trn_spike_rate:.1f} Hz")

        # Cross-correlation between TRN and relay activity
        trn_pop_activity = self.trn_spikes_traces.sum(axis=1).astype(float)
        relay_pop_activity = self.relay_spikes_traces.sum(axis=1).astype(float)

        if trn_pop_activity.sum() > 0 and relay_pop_activity.sum() > 0:
            # Normalize
            trn_norm = (trn_pop_activity - trn_pop_activity.mean()) / (trn_pop_activity.std() + 1e-10)
            relay_norm = (relay_pop_activity - relay_pop_activity.mean()) / (relay_pop_activity.std() + 1e-10)

            # Cross-correlation (lag=0 to 50ms)
            max_lag = 50
            xcorr = np.correlate(relay_norm, trn_norm, mode='full')
            lags = np.arange(-len(trn_norm)+1, len(relay_norm))
            center_idx = len(lags) // 2

            # Focus on positive lags (TRN→relay causality)
            relevant_lags = lags[center_idx:center_idx+max_lag]
            relevant_xcorr = xcorr[center_idx:center_idx+max_lag]

            peak_lag_idx = np.argmax(np.abs(relevant_xcorr))
            peak_lag = relevant_lags[peak_lag_idx]
            peak_corr = relevant_xcorr[peak_lag_idx]

            print(f"  TRN→relay peak correlation: {peak_corr:.3f} at lag {peak_lag}ms")

            if abs(peak_corr) < 0.1:
                print(f"  ⚠️  Very weak TRN-relay coupling (correlation<0.1)")
                print(f"  ⚠️  Oscillatory mechanism may not be functional")
            elif peak_corr < 0:
                print(f"  ✓ Negative correlation (inhibitory coupling) at lag {peak_lag}ms")
            else:
                print(f"  ⚠️  Positive correlation at lag {peak_lag}ms (expected negative for inhibition)")

    def analyze_frequency_spectrum(self) -> None:
        """Analyze frequency spectrum of population activity using FFT."""
        print("\n" + "="*80)
        print("FREQUENCY SPECTRUM ANALYSIS (FFT)")
        print("="*80)

        # Analyze a few key populations
        sample_populations = [
            ("thalamus:relay", "Thalamus relay"),
            ("cortex:l4", "Cortex L4"),
            ("cortex:l23", "Cortex L2/3"),
            ("hippocampus:ca1", "Hippocampus CA1"),
        ]

        for key, display_name in sample_populations:
            if key not in self.population_rates:
                continue

            rates = self.population_rates[key]
            if rates.sum() == 0:
                print(f"\n{display_name}: No activity")
                continue

            # Detrend (remove DC component)
            rates_detrended = rates - rates.mean()

            # Compute FFT
            fft_result = np.fft.rfft(rates_detrended)
            freqs = np.fft.rfftfreq(len(rates), d=self.bin_size_ms / 1000.0)  # Convert to seconds
            power = np.abs(fft_result) ** 2

            # Normalize power
            power = power / power.sum()

            # Find dominant frequencies
            freq_bands = {
                "delta (1-4 Hz)": (1, 4),
                "theta (4-8 Hz)": (4, 8),
                "alpha (8-12 Hz)": (8, 12),
                "beta (12-30 Hz)": (12, 30),
                "gamma (30-100 Hz)": (30, 100),
            }

            print(f"\n{display_name}:")
            print(f"  Mean firing rate: {rates.mean():.2f} spikes per {self.bin_size_ms}ms bin")

            for band_name, (f_min, f_max) in freq_bands.items():
                mask = (freqs >= f_min) & (freqs < f_max)
                if mask.any():
                    band_power = power[mask].sum()
                    peak_idx = mask.nonzero()[0][power[mask].argmax()] if power[mask].max() > 0 else None
                    peak_freq = freqs[peak_idx] if peak_idx is not None else 0

                    print(f"  {band_name:20s}: {band_power*100:5.1f}% power", end="")
                    if band_power > 0.3:  # Dominant band
                        print(f" ⚠️  DOMINANT (peak at {peak_freq:.1f} Hz)")
                    elif band_power > 0.15:
                        print(f" (peak at {peak_freq:.1f} Hz)")
                    else:
                        print()

            # Overall assessment
            delta_power = power[(freqs >= 1) & (freqs < 4)].sum()
            if delta_power > 0.5:
                print(f"  🔴 CRITICAL: Delta band dominates ({delta_power*100:.1f}% power)")
                print(f"  🔴 Indicates slow oscillations or pathological synchrony")
            elif delta_power > 0.3:
                print(f"  ⚠️  WARNING: Elevated delta power ({delta_power*100:.1f}%)")
            else:
                print(f"  ✓ Healthy frequency distribution (delta={delta_power*100:.1f}%)")

    def analyze_isi_distributions(self) -> None:
        """Analyze inter-spike interval distributions for key populations."""
        print("\n" + "="*80)
        print("INTER-SPIKE INTERVAL (ISI) ANALYSIS")
        print("="*80)

        sample_populations = [
            ("thalamus", "relay", "Thalamus relay"),
            ("cortex", "l4", "Cortex L4"),
            ("cortex", "l23", "Cortex L2/3"),
            ("hippocampus", "ca1", "Hippocampus CA1"),
        ]

        for region_name, pop_name, display_name in sample_populations:
            if region_name not in self.spike_times:
                continue
            if pop_name not in self.spike_times[region_name]:
                continue

            # Collect ISIs from all neurons
            all_isis = []
            for neuron_times in self.spike_times[region_name][pop_name]:
                if len(neuron_times) < 2:
                    continue
                isis = np.diff(neuron_times)
                all_isis.extend(isis.tolist())

            if len(all_isis) == 0:
                print(f"\n{display_name}: No spikes")
                continue

            all_isis = np.array(all_isis) * self.brain.dt_ms  # Convert to ms

            # Compute statistics
            mean_isi = np.mean(all_isis)
            std_isi = np.std(all_isis)
            cv_isi = std_isi / mean_isi if mean_isi > 0 else 0.0

            print(f"\n{display_name}:")
            print(f"  Mean ISI: {mean_isi:.1f} ms")
            print(f"  Std ISI: {std_isi:.1f} ms")
            print(f"  CV (coefficient of variation): {cv_isi:.3f}")

            # Diagnosis
            if cv_isi < 0.5:
                print(f"  🔴 Regular firing (CV < 0.5) - pathological synchrony")
            elif cv_isi > 1.5:
                print(f"  ⚠️  Bursting (CV > 1.5)")
            else:
                print(f"  ✓ Irregular firing (CV ≈ 1.0) - healthy AI state")

    def analyze_cross_correlations(self) -> None:
        """Analyze cross-regional correlations."""
        print("\n" + "="*80)
        print("CROSS-REGIONAL CORRELATION ANALYSIS")
        print("="*80)

        n_regions = len(self.region_names)
        corr_matrix = np.zeros((n_regions, n_regions))

        for i in range(n_regions):
            for j in range(i, n_regions):
                if self.region_spike_counts[:, i].sum() == 0 or self.region_spike_counts[:, j].sum() == 0:
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0
                else:
                    corr = np.corrcoef(self.region_spike_counts[:, i], self.region_spike_counts[:, j])[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        # Print highly correlated pairs
        print("\nHighly correlated region pairs (|corr| > 0.5):")
        found_high_corr = False
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.5:
                    found_high_corr = True
                    print(f"  {self.region_names[i]} ↔ {self.region_names[j]}: {corr:.3f}")
                    if corr > 0.7:
                        print(f"    🔴 Strong synchronization!")

        if not found_high_corr:
            print("  ✓ No strong correlations (all < 0.5) - healthy asynchronous state")

    def analyze_conductance_dynamics(self) -> None:
        """Analyze conductance dynamics and phase diagram metrics."""
        print("\n" + "="*80)
        print("CONDUCTANCE DYNAMICS & PHASE DIAGRAM")
        print("="*80)

        print("\nTarget AI Regime (Mean-Field Theory):")
        print("  R_g (conductance ratio):  2 - 8")
        print("  R_s (slow feedback):      3 - 15")
        print("  R_EI (E/I balance):       0.8 - 1.2")
        print("  S (stability):            1 - 3")

        for key, samples in self.conductance_samples.items():
            if len(samples) == 0:
                continue

            # Extract conductances
            g_exc = np.array([s["g_exc"] for s in samples])
            g_inh = np.array([s["g_inh"] for s in samples])
            g_nmda = np.array([s.get("g_nmda", 0.0) for s in samples])

            # Compute means
            mean_g_exc = np.mean(g_exc)
            mean_g_inh = np.mean(g_inh)
            mean_g_nmda = np.mean(g_nmda)
            mean_g_total = mean_g_exc + mean_g_inh + mean_g_nmda

            # Phase diagram metrics
            g_L = 1.0  # Normalized units
            R_g = mean_g_total / g_L

            tau_mem = 20.0  # ms
            tau_slow = 100.0  # ms (NMDA)
            tau_eff = tau_mem / (1.0 + R_g)
            R_s = tau_slow / tau_eff if tau_eff > 0 else float('inf')

            R_EI = mean_g_exc / mean_g_inh if mean_g_inh > 0 else float('inf')

            G_rec = R_EI
            S = G_rec * R_s

            print(f"\n{key}:")
            print(f"  Mean g_exc: {mean_g_exc:.4f}")
            print(f"  Mean g_inh: {mean_g_inh:.4f}")
            print(f"  Mean g_nmda: {mean_g_nmda:.4f}")
            print(f"  Mean g_total: {mean_g_total:.4f}")

            print(f"\n  Phase Diagram Metrics:")
            print(f"    R_g  = {R_g:7.2f}  (target: 2-8) ", end="")
            if 2 <= R_g <= 8:
                print("✓")
            elif R_g < 2:
                print("⚠️  Under-driven")
            else:
                print("⚠️  High conductance")

            print(f"    τ_eff = {tau_eff:6.2f} ms")
            print(f"    R_s  = {R_s:7.2f}  (target: 3-15) ", end="")
            if 3 <= R_s <= 15:
                print("✓")
            elif R_s > 50:
                print("❌ Slow oscillations")
            else:
                print("⚠️")

            print(f"    R_EI = {R_EI:7.2f}  (target: 0.8-1.2) ", end="")
            if 0.8 <= R_EI <= 1.2:
                print("✓")
            else:
                print("⚠️  Imbalanced")

            print(f"    S    = {S:7.2f}  (target: 1-3) ", end="")
            if 1 <= S <= 3:
                print("✓ Stable AI")
            elif S > 10:
                print("❌ Oscillatory")
            else:
                print("⚠️")

    def analyze_homeostatic_gains(self) -> None:
        """Analyze homeostatic g_L_scale (leak conductance scaling) trajectories."""
        print("\n" + "="*80)
        print("HOMEOSTATIC EXCITABILITY ANALYSIS (g_L_scale)")
        print("="*80)

        if not self.gain_traces:
            print("\n  No g_L_scale traces collected")
            return

        print(f"\ng_L_scale trajectories (initial → final):")
        print(f"  Note: g_L_scale range [0.1, 2.0] - lower = more excitable, higher = less excitable")
        print()
        for gain_name in sorted(self.gain_traces.keys()):
            gains = self.gain_traces[gain_name]
            if len(gains) > 0:
                initial = gains[0]
                final = gains[-1]
                change_pct = ((final - initial) / initial) * 100 if initial > 0 else 0

                # Healthy range is typically [0.3, 2.0]
                status = "⚠️" if final < 0.3 or final > 2.0 else "✓"
                print(f"  {gain_name:<25s}: {initial:.3f} → {final:.3f} ({change_pct:+6.1f}%) {status}")

                if final < 0.3:
                    print(f"      ⚠️ COLLAPSED (< 0.3 threshold)")
                elif final > 2.0:
                    print(f"      ⚠️ CLAMPED AT MAXIMUM (> 2.0)")

    def generate_health_report(self) -> None:
        """Generate overall health assessment combining BrainActivityAnalyzer + specialized checks."""
        print("\n" + "="*80)
        print("OVERALL HEALTH ASSESSMENT")
        print("="*80)

        # Start with health assessment from BrainActivityAnalyzer
        if self.brain_report and self.brain_report.health:
            print("\nFrom brain-wide analysis:")
            if self.brain_report.health.is_healthy:
                print("  ✅ Brain activity patterns: HEALTHY")
            else:
                print("  ⚠️  Brain activity patterns: ISSUES DETECTED")
                if self.brain_report.health.critical_issues:
                    print("\n  Critical issues:")
                    for issue in self.brain_report.health.critical_issues:
                        print(f"    • {issue}")
                if self.brain_report.health.warnings:
                    print("\n  Warnings:")
                    for warning in self.brain_report.health.warnings[:5]:
                        print(f"    • {warning}")

        # Add specialized checks
        specialized_issues = []
        specialized_warnings = []

        # Check g_L_scale values
        for gain_name, gains in self.gain_traces.items():
            if len(gains) > 0:
                final_g_L = gains[-1]
                if final_g_L < 0.3:
                    specialized_issues.append(f"Collapsed g_L_scale: {gain_name} = {final_g_L:.3f}")
                elif final_g_L > 2.0:
                    specialized_warnings.append(f"Maximum g_L_scale: {gain_name} = {final_g_L:.3f}")

        # Check septum pacemaker (if available)
        if self.septum_gaba_trace is not None and self.septum_gaba_trace.mean() < 0.5:
            specialized_warnings.append(f"Septum pacemaker activity low: {self.septum_gaba_trace.mean():.2f} spikes/ms")

        if specialized_issues or specialized_warnings:
            print("\nFrom specialized analyses:")
            if specialized_issues:
                print("  Critical issues:")
                for issue in specialized_issues:
                    print(f"    • {issue}")
            if specialized_warnings:
                print("  Warnings:")
                for warning in specialized_warnings:
                    print(f"    • {warning}")

        # Overall summary
        all_healthy = (
            (self.brain_report and self.brain_report.health and self.brain_report.health.is_healthy) and
            not specialized_issues and
            not specialized_warnings
        )

        if all_healthy:
            print("\n" + "="*80)
            print("✅ ALL SYSTEMS NOMINAL")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("⚠️  ATTENTION REQUIRED - See issues above")
            print("="*80)

        print()

    def save_detailed_report(self, output_dir: str) -> None:
        """Save comprehensive diagnostic data to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save brain activity report data
        if self.brain_report:
            report_data: Dict[str, Any] = {
                'is_healthy': self.brain_report.health.is_healthy if self.brain_report.health else None,
                'critical_issues': self.brain_report.health.critical_issues if self.brain_report.health else [],
                'warnings': self.brain_report.health.warnings if self.brain_report.health else [],
            }

            with open(f"{output_dir}/comprehensive_diagnostics_health.json", 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)

            print(f"✓ Saved health report to {output_dir}/comprehensive_diagnostics_health.json")

        # Save specialized traces as numpy arrays
        trace_data = {}
        if self.septum_ach_trace is not None:
            trace_data['septum_ach'] = self.septum_ach_trace
            trace_data['septum_gaba'] = self.septum_gaba_trace

        if trace_data:
            np.savez(
                f"{output_dir}/comprehensive_diagnostics_traces.npz",
                **trace_data
            )
            print(f"✓ Saved specialized traces to {output_dir}/comprehensive_diagnostics_traces.npz")


def print_neuron_populations(brain: DynamicBrain) -> None:
    """Print neuron population sizes for each region."""
    print("\n" + "="*80)
    print("NEURON POPULATION SIZES")
    print("="*80)

    for region_name, region in brain.regions.items():
        print(f"- {region_name}:")
        region_neuron_count = 0
        for pop_name, population in region.neuron_populations.items():
            print(f"    {pop_name}: {population.n_neurons} neurons ({population.__class__.__name__})")
            region_neuron_count += population.n_neurons
        print(f"    Total neurons in {region_name}: {region_neuron_count}")


def analyze_synaptic_weights(brain: DynamicBrain) -> None:
    """Analyze synaptic weight distributions across brain regions."""
    print("\n" + "="*80)
    print("SYNAPTIC WEIGHTS ANALYSIS")
    print("="*80)

    for region in brain.regions.values():
        for synapse_id, weights in region._synaptic_weights.items():
            mean_weight = weights.mean()
            min_weight = weights.min()
            max_weight = weights.max()
            print(f"  {region.__class__.__name__}:{synapse_id} - Mean: {mean_weight:.4f}, Min: {min_weight:.4f}, Max: {max_weight:.4f}")


def main():
    """Run comprehensive diagnostics."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run comprehensive brain diagnostics with FFT frequency analysis, ISI distributions, and phase diagram metrics."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of simulation timesteps (default: 1000ms)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/diagnostics",
        help="Directory to save diagnostic output files (default: data/diagnostics)"
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default="none",
        choices=["random", "rhythmic", "burst", "none"],
        help="Sensory input pattern: 'random' (continuous Poisson), 'rhythmic' (8Hz theta), 'burst' (single burst at 100ms), 'none' (no input) (default: random)"
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("THALIA COMPREHENSIVE BRAIN DIAGNOSTICS")
    print("="*80)

    # Build brain
    print("\nBuilding brain with default architecture...")
    start_time = time.time()
    brain = BrainBuilder.preset("default")
    build_time = time.time() - start_time

    print(f"✓ Brain built in {build_time:.2f}s")
    print(f"  WeightInitializer.GLOBAL_WEIGHT_SCALE: {WeightInitializer.GLOBAL_WEIGHT_SCALE}")
    print(f"  Axonal tracts: {len(brain.axonal_tracts)}")
    print(f"  Regions: {len(brain.regions)}")
    for region_name, region in brain.regions.items():
        print(f"  - {region_name}:")
        print(f"    Baseline noise conductance: {region.config.baseline_noise_conductance}")
        print(f"    Neuromodulation enabled: {region.config.enable_neuromodulation}")
        if hasattr(region.config, "baseline_drive"):
            print(f"    Baseline drive: {region.config.baseline_drive}")
        if hasattr(region.config, "ca3_persistent_gain"):
            print(f"    CA3 persistent gain: {region.config.ca3_persistent_gain}")

    print_neuron_populations(brain)
    analyze_synaptic_weights(brain)

    # Create diagnostics runner
    diagnostics = ComprehensiveDiagnostics(brain, timesteps=args.timesteps, input_pattern=args.input_pattern)

    # Run SINGLE simulation with callbacks for specialized tracking
    diagnostics.run_simulation_with_callbacks()

    analyze_synaptic_weights(brain)

    # Print brain-wide activity report
    diagnostics.print_brain_activity_report()

    # Run specialized analyses
    diagnostics.analyze_septum_rhythm()
    diagnostics.analyze_thalamus_t_channels()
    diagnostics.analyze_snr_basal_ganglia()
    diagnostics.analyze_cortex_l4_inhibition()
    diagnostics.analyze_hippocampus_inhibition()
    diagnostics.analyze_striatum_pathways()
    diagnostics.analyze_homeostatic_gains()

    # Run delta synchrony investigation analyses
    diagnostics.analyze_frequency_spectrum()
    diagnostics.analyze_isi_distributions()
    diagnostics.analyze_cross_correlations()
    diagnostics.analyze_conductance_dynamics()

    # Generate combined health report
    diagnostics.generate_health_report()

    # Save detailed data
    diagnostics.save_detailed_report(output_dir=args.output_dir)

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
