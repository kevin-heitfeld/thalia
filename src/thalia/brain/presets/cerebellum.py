"""Cerebellar circuit connections preset.

Wires the cerebellum forward-model circuit:

* **Corticopontocerebellar input**: Sensory cortex L5 and PFC L5 drive granule
  cells via the pontine relay (Schmahmann 1996).
* **Cerebellothalamic output**: DCN predictions route through thalamus relay
  (VL/VA) to cortex.
* **Error signal**: Sensory cortex L5 → inferior olive climbing fibers drive
  Purkinje LTD (Marr-Albus-Ito framework).
* **Nucleo-olivary inhibition**: DCN → IO GABA silences error when predictions
  are correct (Bengtsson & Hesslow 2006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thalia.brain.regions.population_names import (
    CerebellumPopulation,
    CortexPopulation,
    ThalamusPopulation,
)
from thalia.brain.synapses import ConductanceScaledSpec, STPConfig
from thalia.typing import ReceptorType, SynapseId

if TYPE_CHECKING:
    from thalia.brain.brain_builder import BrainBuilder


def connect_cerebellum(builder: BrainBuilder) -> None:
    """Wire the cerebellum forward-model circuit.

    Input: sensory L5 and PFC drive granule cells (via pons).
    Output: DCN predictions route through thalamus relay (VL/VA) to cortex.
    """
    # Sensory cortex L5 → Cerebellum GRANULE: corticopontocerebellar pathway
    # Via pontine nuclei (Schmahmann 1996); distance ~10-15cm → 20-30ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="cerebellum",
            target_population=CerebellumPopulation.GRANULE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=25.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=2.5,
            target_v_inf=0.90,
            fraction_of_drive=0.10,
        ),
        stp_config=STPConfig(U=0.10, tau_d=100.0, tau_f=500.0),
    )

    # PFC → Cerebellum GRANULE: goal / context input (similar pathway length)
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="cerebellum",
            target_population=CerebellumPopulation.GRANULE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=25.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=2.5,
            target_v_inf=0.90,
            fraction_of_drive=0.08,
        ),
        stp_config=STPConfig(U=0.10, tau_d=100.0, tau_f=500.0),
    )

    # Cerebellum DCN → Thalamus RELAY: cerebellothalamic pathway (VL/VA nucleus)
    # DCN glutamatergic projection neurons target ventrolateral thalamic relay
    # neurons (Asanuma et al. 1983; Sakai et al. 1996).  The thalamic relay then
    # gates and transmits cerebellar predictions to cortex via the existing
    # thalamocortical connections (→ L4 PYR + L4 PV).
    # This two-stage relay enables TRN-mediated attention gating of cerebellar
    # predictions — cortex only receives motor predictions when thalamic relay
    # is in the appropriate state.
    # Distance: ~4-6cm (deep cerebellar nuclei → ventral thalamus) → 8-12ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cerebellum",
            source_population=CerebellumPopulation.DCN,
            target_region="thalamus_sensory",
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=50.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
        # DCN→thalamus is a facilitating pathway (cerebellar predictions build
        # up with DCN bursts).  U=0.05: low initial release → facilitates during
        # DCN bursts; tau_f > tau_d ensures net facilitation.
        stp_config=STPConfig(U=0.05, tau_d=150.0, tau_f=300.0),
    )

    # =========================================================================
    # Error signal to Inferior Olive (completes the supervised learning loop)
    # =========================================================================
    # The Marr-Albus-Ito framework requires:
    #   1. An error signal driving IO → climbing fiber → Purkinje LTD
    #   2. Nucleo-olivary inhibition (DCN → IO) so correct predictions silence IO

    # Sensory cortex L5 → Cerebellum IO: sensorimotor error signal
    # Cortical pyramidal tract neurons detect mismatches between predicted and
    # actual sensory feedback.  This error signal reaches IO via the
    # mesodiencephalic junction and central tegmental tract (Llinas et al. 2004;
    # Apps & Garwicz 2005).  High cortical error → IO fires → climbing fiber
    # drives Purkinje LTD at co-active parallel fiber synapses.
    # Distance: ~8-12cm (cortex → brainstem IO) → 15-25ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="cerebellum",
            target_population=CerebellumPopulation.INFERIOR_OLIVE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=20.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.40,  # IO needs strong drive: only fires ~1-5 Hz due to refractory
        ),
        stp_config=STPConfig(U=0.35, tau_d=600.0, tau_f=50.0),
    )

    # Cerebellum DCN_GABA → Cerebellum IO: nucleo-olivary inhibition (GABA)
    # DCN contains a dedicated GABAergic subpopulation that inhibits IO neurons,
    # creating a negative feedback loop: accurate predictions → high DCN_GABA
    # firing → IO suppression → reduced climbing fiber error signal → no LTD
    # (Bengtsson & Hesslow 2006).  This is the mechanism by which cerebellar
    # learning converges — error is suppressed as the forward model improves.
    # Distance: internal (~0.5cm) → 1-2ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cerebellum",
            source_population=CerebellumPopulation.DCN_GABA,
            target_region="cerebellum",
            target_population=CerebellumPopulation.INFERIOR_OLIVE,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=1.5,
        connectivity=0.40,
        weight_scale=0.003,  # Strong: IO must be silenced by correct DCN predictions
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )
