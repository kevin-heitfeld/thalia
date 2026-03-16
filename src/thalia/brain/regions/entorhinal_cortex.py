"""Entorhinal Cortex (EC) — Hippocampal Gateway Region.

The entorhinal cortex is the principal interface between the neocortex and
the hippocampal formation.  Sensory and associative cortical signals enter via
layers II and III; hippocampal outputs are relayed back to neocortex via
layer V.

Biological Background:
======================
**Anatomy:**
- Located at the medial temporal lobe, adjacent to hippocampus
- Provides ~85 % of hippocampal afferents
- Layers II and III project *in*, layer V projects *out*
- ~55,000 neurons in rat; ~5 × 10⁶ in human

**Three Functional Sub-populations:**

1. **EC_II (Layer II stellate cells — Perforant Path)**
   - Contains grid cells, border cells, head-direction cells
   - Axons cross the hippocampal fissure → perforant path
   - Project to DG outer molecular layer AND CA3 stratum lacunosum-moleculare
   - Facilitating STP — stronger signal with repeated cortical inputs
   - Primary *what/where* input to hippocampus

2. **EC_III (Layer III pyramidal cells — Temporoammonic Path)**
   - "Time cells" — ramp-like temporal coding on 100–200 ms scale
   - Project directly to CA1 stratum lacunosum-moleculare (bypassing DG/CA3)
   - Depressing STP — strong initial pulse, then adapts (novelty emphasis)
   - Critical for disambiguating retrieved memories from current perception

3. **EC_V (Layer V pyramidal cells — Back-projection)**
   - Receive CA1 / subiculum output (via synaptic projections)
   - Relay condensed memory index back to association and sensory cortices
   - Moderate sustained activity — short-term memory buffer over theta cycles

**Inputs (expected SynapseId patterns):**
- ``cortex_sensory`` / ``CortexPopulation.L5_PYR``  → EC_II (spatial context)
- ``cortex_association`` / ``CortexPopulation.L23_PYR``  → EC_II, EC_III (semantic context)
- ``hippocampus`` / ``HippocampusPopulation.CA1``  → EC_V (memory readout)

**Outputs:**
- ``EntorhinalCortexPopulation.EC_II``  — perforant path input to DG and CA3
- ``EntorhinalCortexPopulation.EC_III`` — temporoammonic input to CA1
- ``EntorhinalCortexPopulation.EC_V``   — back-projection to neocortex
"""

from __future__ import annotations

from typing import ClassVar, List, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import EntorhinalCortexConfig, EntorhinalCortexPopulationConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    ConductanceLIF,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    split_excitatory_conductance,
)
from thalia.brain.synapses import STPConfig, WeightInitializer
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import CircularDelayBuffer

from .neural_region import NeuralRegion
from .population_names import EntorhinalCortexPopulation
from .region_registry import register_region


@register_region(
    "entorhinal_cortex",
    aliases=["ec", "entorhinal", "hippocampal_gateway"],
    description="Entorhinal cortex — hippocampal gateway (perforant + temporoammonic paths)",
)
class EntorhinalCortex(NeuralRegion[EntorhinalCortexConfig]):
    """Entorhinal Cortex — Hippocampal Gateway.

    Routes cortical information into the hippocampal trisynaptic circuit via
    the perforant path (EC_II → DG, CA3) and temporoammonic direct path
    (EC_III → CA1), and relays hippocampal output back to neocortex via
    layer V (EC_V).

    Input Populations (via SynapseId routing):
    ------------------------------------------
    - Cortical inputs → EC_II (perforant path driver)
    - Cortical inputs → EC_III (temporoammonic path driver)
    - Hippocampal CA1 → EC_V (memory index back-projection)

    Output Populations:
    -------------------
    - ``EntorhinalCortexPopulation.EC_II``  — perforant path spikes (→ hippocampus DG, CA3)
    - ``EntorhinalCortexPopulation.EC_III`` — temporoammonic spikes (→ hippocampus CA1)
    - ``EntorhinalCortexPopulation.EC_V``   — back-projection spikes (→ cortex)
    """

    # EC does not output neuromodulators; it is a structural relay region.
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = []

    def __init__(
        self,
        config: EntorhinalCortexConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialise entorhinal cortex populations.

        Args:
            config: ``EntorhinalCortexConfig`` with layer-specific neuron params.
            population_sizes: Mapping of ``EntorhinalCortexPopulation`` keys to neuron counts.
            region_name: Unique region name string (e.g. ``"entorhinal_cortex"``).
        """
        super().__init__(config, population_sizes, region_name, device=device)

        self.ec_ii_size: int = population_sizes[EntorhinalCortexPopulation.EC_II]
        self.ec_iii_size: int = population_sizes[EntorhinalCortexPopulation.EC_III]
        self.ec_v_size: int = population_sizes[EntorhinalCortexPopulation.EC_V]
        self.ec_inh_size: int = population_sizes[EntorhinalCortexPopulation.EC_INHIBITORY]

        ec_ii_overrides: EntorhinalCortexPopulationConfig = config.population_overrides[EntorhinalCortexPopulation.EC_II]
        ec_iii_overrides: EntorhinalCortexPopulationConfig = config.population_overrides[EntorhinalCortexPopulation.EC_III]
        ec_v_overrides: EntorhinalCortexPopulationConfig = config.population_overrides[EntorhinalCortexPopulation.EC_V]
        ec_inh_overrides: EntorhinalCortexPopulationConfig = config.population_overrides[EntorhinalCortexPopulation.EC_INHIBITORY]

        # ── EC_II: Layer II stellate cells (Grid / place cells) ──────────────
        self.ec_ii_neurons: ConductanceLIF
        self.ec_ii_neurons = self._create_and_register_neuron_population(
            population_name=EntorhinalCortexPopulation.EC_II,
            n_neurons=self.ec_ii_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                g_L=heterogeneous_g_L(0.05, self.ec_ii_size, device),
                tau_E=5.0,
                tau_I=10.0,
                v_reset=-0.08,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                v_threshold=heterogeneous_v_threshold(ec_ii_overrides.v_threshold, self.ec_ii_size, device),
                adapt_increment=heterogeneous_adapt_increment(ec_ii_overrides.adapt_increment, self.ec_ii_size, device),
                tau_adapt=ec_ii_overrides.adapt_tau_ms,
                tau_mem_ms=heterogeneous_tau_mem(ec_ii_overrides.tau_mem_ms, self.ec_ii_size, device),
            ),
        )

        # ── EC_III: Layer III pyramidal cells (Time cells) ───────────────────
        self.ec_iii_neurons: ConductanceLIF
        self.ec_iii_neurons = self._create_and_register_neuron_population(
            population_name=EntorhinalCortexPopulation.EC_III,
            n_neurons=self.ec_iii_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                g_L=heterogeneous_g_L(0.05, self.ec_iii_size, device),
                tau_E=5.0,
                tau_I=10.0,
                v_reset=-0.08,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                v_threshold=heterogeneous_v_threshold(ec_iii_overrides.v_threshold, self.ec_iii_size, device),
                adapt_increment=heterogeneous_adapt_increment(ec_iii_overrides.adapt_increment, self.ec_iii_size, device),
                tau_adapt=ec_iii_overrides.adapt_tau_ms,
                tau_mem_ms=heterogeneous_tau_mem(ec_iii_overrides.tau_mem_ms, self.ec_iii_size, device),
            ),
        )

        # ── EC_V: Layer V pyramidal cells (Back-projection to cortex) ────────
        self.ec_v_neurons: ConductanceLIF
        self.ec_v_neurons = self._create_and_register_neuron_population(
            population_name=EntorhinalCortexPopulation.EC_V,
            n_neurons=self.ec_v_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                g_L=heterogeneous_g_L(0.05, self.ec_v_size, device),
                tau_E=5.0,
                tau_I=10.0,
                v_reset=-0.08,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                v_threshold=heterogeneous_v_threshold(ec_v_overrides.v_threshold, self.ec_v_size, device),
                adapt_increment=heterogeneous_adapt_increment(ec_v_overrides.adapt_increment, self.ec_v_size, device),
                tau_adapt=ec_v_overrides.adapt_tau_ms,
                tau_mem_ms=heterogeneous_tau_mem(ec_v_overrides.tau_mem_ms, self.ec_v_size, device),
            ),
        )

        # ── EC_INHIBITORY: Layer II/III PV basket cells ────────────────────────
        # Fast-spiking PV interneurons that provide feedforward/feedback inhibition
        # to EC_II and EC_III.  They are driven by the same cortical input as
        # the principal cells and fire before them (lower threshold), clamping
        # activity to the biological 1–10 Hz range and preventing runaway
        # excitation during periods of strong cortical drive.
        self.ec_inh_neurons: ConductanceLIF
        self.ec_inh_neurons = self._create_and_register_neuron_population(
            population_name=EntorhinalCortexPopulation.EC_INHIBITORY,
            n_neurons=self.ec_inh_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=3.0,
                tau_ref=2.5,
                g_L=heterogeneous_g_L(0.10, self.ec_inh_size, device, cv=0.08),
                v_threshold=heterogeneous_v_threshold(ec_inh_overrides.v_threshold, self.ec_inh_size, device, cv=0.06),
                adapt_increment=heterogeneous_adapt_increment(ec_inh_overrides.adapt_increment, self.ec_inh_size, device),
                tau_adapt=ec_inh_overrides.adapt_tau_ms,
                tau_mem_ms=heterogeneous_tau_mem(ec_inh_overrides.tau_mem_ms, self.ec_inh_size, device=device, cv=0.10),
            ),
        )

        # ── Internal connectivity: E→I and I→E ────────────────────────────
        # EC_II → EC_INHIBITORY (feedforward excitation: AMPA, mild depression).
        # Weight std scales with 1/n_src matching cortical E→PV formula (40/n_pyr
        # gives 10-70 Hz; 10/n gives ~5-10 Hz with adaptation, clamped to 1-10 Hz target).
        # STP: U=0.25, τd=150ms (mild depression) — less depressing than E→E so
        # interneurons stay responsive during sustained cortical drive.
        self._add_internal_connection(
            source_population=EntorhinalCortexPopulation.EC_II,
            target_population=EntorhinalCortexPopulation.EC_INHIBITORY,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.ec_ii_size,
                n_output=self.ec_inh_size,
                connectivity=0.5,
                mean=0.0,
                std=8.0 / self.ec_ii_size,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.25, tau_d=150.0, tau_f=20.0),
        )
        # EC_III → EC_INHIBITORY (feedforward excitation).
        self._add_internal_connection(
            source_population=EntorhinalCortexPopulation.EC_III,
            target_population=EntorhinalCortexPopulation.EC_INHIBITORY,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.ec_iii_size,
                n_output=self.ec_inh_size,
                connectivity=0.5,
                mean=0.0,
                std=8.0 / self.ec_iii_size,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.25, tau_d=150.0, tau_f=20.0),
        )
        # EC_INHIBITORY → EC_II (perisomatic GABA_A inhibition).
        self._add_internal_connection(
            source_population=EntorhinalCortexPopulation.EC_INHIBITORY,
            target_population=EntorhinalCortexPopulation.EC_II,
            weights=WeightInitializer.sparse_random(
                n_input=self.ec_inh_size,
                n_output=self.ec_ii_size,
                connectivity=0.6,
                weight_scale=0.040,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.5, tau_d=150.0, tau_f=20.0),
        )
        # EC_INHIBITORY → EC_III (perisomatic GABA_A inhibition).
        self._add_internal_connection(
            source_population=EntorhinalCortexPopulation.EC_INHIBITORY,
            target_population=EntorhinalCortexPopulation.EC_III,
            weights=WeightInitializer.sparse_random(
                n_input=self.ec_inh_size,
                n_output=self.ec_iii_size,
                connectivity=0.6,
                weight_scale=0.040,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.5, tau_d=150.0, tau_f=20.0),
        )
        # EC_INHIBITORY → EC_V (weaker: layer V is functionally distinct).
        self._add_internal_connection(
            source_population=EntorhinalCortexPopulation.EC_INHIBITORY,
            target_population=EntorhinalCortexPopulation.EC_V,
            weights=WeightInitializer.sparse_random(
                n_input=self.ec_inh_size,
                n_output=self.ec_v_size,
                connectivity=0.3,
                weight_scale=0.018,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.5, tau_d=150.0, tau_f=20.0),
        )

        # Spike buffers for 1-step causal E→I and I→E delays
        # (standard CircularDelayBuffer pattern — see cortical_column.py).
        self._ec_ii_buf  = CircularDelayBuffer(max_delay=1, size=self.ec_ii_size,  dtype=torch.bool, device=device)
        self._ec_iii_buf = CircularDelayBuffer(max_delay=1, size=self.ec_iii_size, dtype=torch.bool, device=device)
        self._ec_inh_buf = CircularDelayBuffer(max_delay=1, size=self.ec_inh_size, dtype=torch.bool, device=device)

        self.to(device)

    # =========================================================================
    # FORWARD
    # =========================================================================

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Route cortical → hippocampal and hippocampal → cortical signals.

        Processing order:
        1. Integrate all cortical synaptic inputs into EC_II and EC_III dendrites.
        2. Add layer-specific tonic drive.
        3. Fire EC_II and EC_III neurons (output: perforant / temporoammonic spikes).
        4. Integrate hippocampal CA1 back-projection into EC_V dendrites.
        5. Fire EC_V neurons (output: back-projection spikes to neocortex).

        Args:
            synaptic_inputs: Routed ``SynapseId``-keyed spike tensors.
            neuromodulator_inputs: Broadcast neuromodulatory signals (not used
                directly — EC does not subscribe to neuromodulators, but the
                base class may inspect them).

        Returns:
            ``RegionOutput`` mapping ``EntorhinalCortexPopulation`` keys to boolean spike tensors.
        """
        config = self.config

        ec_ii_overrides: EntorhinalCortexPopulationConfig = config.population_overrides[EntorhinalCortexPopulation.EC_II]
        ec_iii_overrides: EntorhinalCortexPopulationConfig = config.population_overrides[EntorhinalCortexPopulation.EC_III]
        ec_v_overrides: EntorhinalCortexPopulationConfig = config.population_overrides[EntorhinalCortexPopulation.EC_V]

        # ── 1. Integrate cortical → EC_II dendrites ──────────────────────────
        # All synaptic connections whose ``target_population == EntorhinalCortexPopulation.EC_II``
        # are summed here (AMPA conductances from sensory + association cortex).
        ec_ii_raw = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ec_ii_size,
            filter_by_target_population=EntorhinalCortexPopulation.EC_II,
        ).g_ampa

        # Add tonic baseline drive (sub-threshold depolarisation from layer I)
        ec_ii_drive = ec_ii_raw + ec_ii_overrides.tonic_drive

        ec_ii_drive = torch.nn.functional.relu(ec_ii_drive)  # Non-negative conductance

        # ── 2. Integrate cortical → EC_III dendrites ─────────────────────────
        ec_iii_raw = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ec_iii_size,
            filter_by_target_population=EntorhinalCortexPopulation.EC_III,
        ).g_ampa

        ec_iii_drive = ec_iii_raw + ec_iii_overrides.tonic_drive
        ec_iii_drive = torch.nn.functional.relu(ec_iii_drive)

        # ── 3. Fire EC_INHIBITORY (PV basket cells) ───────────────────────────
        # Interneurons integrate: (a) external GABA-A routed to EC_INHIBITORY,
        # plus (b) same-population E→I drive from previous-step EC_II/III spikes
        # (1 dt ≈ 1 ms delay; PV cells respond within ~1–2 ms in biology).
        inh_ext = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ec_inh_size,
            filter_by_target_population=EntorhinalCortexPopulation.EC_INHIBITORY,
        )
        inh_int_ii = self._integrate_synaptic_inputs_at_dendrites(
            {SynapseId(
                source_region=self.region_name,
                source_population=EntorhinalCortexPopulation.EC_II,
                target_region=self.region_name,
                target_population=EntorhinalCortexPopulation.EC_INHIBITORY,
                receptor_type=ReceptorType.AMPA,
            ): self._ec_ii_buf.read(1)},
            n_neurons=self.ec_inh_size,
        )
        inh_int_iii = self._integrate_synaptic_inputs_at_dendrites(
            {SynapseId(
                source_region=self.region_name,
                source_population=EntorhinalCortexPopulation.EC_III,
                target_region=self.region_name,
                target_population=EntorhinalCortexPopulation.EC_INHIBITORY,
                receptor_type=ReceptorType.AMPA,
            ): self._ec_iii_buf.read(1)},
            n_neurons=self.ec_inh_size,
        )

        g_exc_inh = inh_ext.g_ampa + inh_int_ii.g_ampa + inh_int_iii.g_ampa
        g_exc_inh_ampa, g_exc_inh_nmda = split_excitatory_conductance(g_exc_inh, nmda_ratio=0.05)
        ec_inh_spikes, _ = self.ec_inh_neurons.forward(
            g_ampa_input=ConductanceTensor(g_exc_inh_ampa),
            g_nmda_input=ConductanceTensor(g_exc_inh_nmda),
            g_gaba_a_input=ConductanceTensor(inh_ext.g_gaba_a),
            g_gaba_b_input=None,
        )

        # ── 4. Compute I→E inhibition using previous-step PV spikes ──────────
        # 1-step causal delay: PV spikes at t-1 gate principal cell firing at t.
        inh_to_ii = self._integrate_synaptic_inputs_at_dendrites(
            {SynapseId(
                source_region=self.region_name,
                source_population=EntorhinalCortexPopulation.EC_INHIBITORY,
                target_region=self.region_name,
                target_population=EntorhinalCortexPopulation.EC_II,
                receptor_type=ReceptorType.GABA_A,
            ): self._ec_inh_buf.read(1)},
            n_neurons=self.ec_ii_size,
        )
        inh_to_iii = self._integrate_synaptic_inputs_at_dendrites(
            {SynapseId(
                source_region=self.region_name,
                source_population=EntorhinalCortexPopulation.EC_INHIBITORY,
                target_region=self.region_name,
                target_population=EntorhinalCortexPopulation.EC_III,
                receptor_type=ReceptorType.GABA_A,
            ): self._ec_inh_buf.read(1)},
            n_neurons=self.ec_iii_size,
        )
        inh_to_v = self._integrate_synaptic_inputs_at_dendrites(
            {SynapseId(
                source_region=self.region_name,
                source_population=EntorhinalCortexPopulation.EC_INHIBITORY,
                target_region=self.region_name,
                target_population=EntorhinalCortexPopulation.EC_V,
                receptor_type=ReceptorType.GABA_A,
            ): self._ec_inh_buf.read(1)},
            n_neurons=self.ec_v_size,
        )
        g_gaba_a_ii  = inh_to_ii.g_gaba_a
        g_gaba_a_iii = inh_to_iii.g_gaba_a
        g_gaba_a_v   = inh_to_v.g_gaba_a

        # ── 5. Fire EC_II neurons ─────────────────────────────────────────────
        ec_ii_g_ampa, ec_ii_g_nmda = split_excitatory_conductance(ec_ii_drive, nmda_ratio=0.3)
        ec_ii_spikes, _ = self.ec_ii_neurons.forward(
            g_ampa_input=ConductanceTensor(ec_ii_g_ampa),
            g_nmda_input=ConductanceTensor(ec_ii_g_nmda),
            g_gaba_a_input=ConductanceTensor(g_gaba_a_ii),
            g_gaba_b_input=None,
        )

        # ── 6. Fire EC_III neurons ────────────────────────────────────────────
        ec_iii_g_ampa, ec_iii_g_nmda = split_excitatory_conductance(ec_iii_drive, nmda_ratio=0.3)
        ec_iii_spikes, _ = self.ec_iii_neurons.forward(
            g_ampa_input=ConductanceTensor(ec_iii_g_ampa),
            g_nmda_input=ConductanceTensor(ec_iii_g_nmda),
            g_gaba_a_input=ConductanceTensor(g_gaba_a_iii),
            g_gaba_b_input=None,
        )

        # ── 7. Integrate hippocampal CA1 → EC_V dendrites ────────────────────
        ec_v_raw = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ec_v_size,
            filter_by_target_population=EntorhinalCortexPopulation.EC_V,
        ).g_ampa

        ec_v_drive = ec_v_raw + ec_v_overrides.tonic_drive
        ec_v_drive = torch.nn.functional.relu(ec_v_drive)

        # ── 8. Fire EC_V neurons ──────────────────────────────────────────────
        ec_v_g_ampa, ec_v_g_nmda = split_excitatory_conductance(ec_v_drive, nmda_ratio=0.25)
        ec_v_spikes, _ = self.ec_v_neurons.forward(
            g_ampa_input=ConductanceTensor(ec_v_g_ampa),
            g_nmda_input=ConductanceTensor(ec_v_g_nmda),
            g_gaba_a_input=ConductanceTensor(g_gaba_a_v),
            g_gaba_b_input=None,
        )

        # ── 9. Advance spike buffers for next timestep ────────────────────────
        self._ec_ii_buf.write_and_advance(ec_ii_spikes)
        self._ec_iii_buf.write_and_advance(ec_iii_spikes)
        self._ec_inh_buf.write_and_advance(ec_inh_spikes)

        region_outputs: RegionOutput = {
            EntorhinalCortexPopulation.EC_II: ec_ii_spikes,
            EntorhinalCortexPopulation.EC_III: ec_iii_spikes,
            EntorhinalCortexPopulation.EC_V: ec_v_spikes,
            EntorhinalCortexPopulation.EC_INHIBITORY: ec_inh_spikes,
        }

        return region_outputs
