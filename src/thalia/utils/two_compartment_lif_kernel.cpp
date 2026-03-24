/**
 * Fused TwoCompartmentLIF neuron step — CPU kernel for Thalia.
 *
 * Replaces the ~60 individual PyTorch tensor operations in
 * TwoCompartmentLIF.forward() with a single C++ loop parallelised via
 * at::parallel_for.  All state buffers are updated in-place.
 *
 * Two-compartment model: soma (basal inputs) + apical dendrite (apical inputs)
 * connected by coupling conductance g_c.  Features:
 *   - Separate NMDA Mg²⁺ block at somatic vs dendritic voltage
 *   - Back-propagating action potential (BAP) on soma spike
 *   - Dendritic Ca²⁺ spikes when V_dend >= theta_Ca
 *   - OU noise on soma
 *   - Gap junctions on soma
 *
 * Loaded at runtime via torch.utils.cpp_extension.load().
 */

#include <torch/extension.h>
#include <ATen/Parallel.h>

#ifdef _MSC_VER
#  define RESTRICT __restrict
#else
#  define RESTRICT __restrict__
#endif

#include <cmath>
#include <cstdint>
#include <algorithm>

// =========================================================================
// Philox 2x32-10 RNG (inlined from philox_cpu_kernel.cpp)
// =========================================================================
static constexpr uint32_t W0 = 0x9E3779B9u;
static constexpr uint32_t W1 = 0xBB67AE85u;
static constexpr float TWO_PI = 6.283185307179586f;

static inline float philox_to_uniform(int64_t counter) {
    int64_t x = counter;
    for (int r = 0; r < 10; ++r) {
        uint32_t lo = static_cast<uint32_t>(x & 0xffffffff);
        uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(x) >> 32) & 0xffffffff);
        lo = static_cast<uint32_t>(static_cast<uint64_t>(lo) * W0);
        hi = static_cast<uint32_t>(static_cast<uint64_t>(hi) * W1);
        x  = (static_cast<int64_t>(hi) << 32) | static_cast<int64_t>(lo);
        x ^= static_cast<int64_t>(static_cast<uint64_t>(W0) * static_cast<uint64_t>(r + 1));
    }
    uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(x) & 0xffffffff);
    return static_cast<float>(static_cast<uint64_t>(bits) + 1u) / 4294967298.0f;
}

static inline float philox_gaussian_scalar(int64_t counter) {
    float u1 = philox_to_uniform(counter);
    float u2 = philox_to_uniform(counter + 1);
    return std::sqrtf(-2.0f * std::logf(u1)) * std::cosf(TWO_PI * u2);
}

// =========================================================================
// Scalar-or-per-neuron parameter access macros
// =========================================================================
#define PARAM_SETUP(varname, tensor) \
    auto varname##_c = (tensor).contiguous(); \
    const bool varname##_is_scalar = (varname##_c.dim() == 0); \
    const float varname##_sval = varname##_is_scalar ? varname##_c.item<float>() : 0.0f; \
    const float* RESTRICT varname##_ptr = varname##_is_scalar ? nullptr : varname##_c.data_ptr<float>()

#define PARAM(varname, i) (varname##_is_scalar ? varname##_sval : varname##_ptr[(i)])

// =========================================================================
// Fused TwoCompartmentLIF step kernel
// =========================================================================

torch::Tensor two_compartment_lif_step_cpp(
    // ── Somatic state tensors (in/out, modified in-place) ──
    torch::Tensor V_soma,            // [N]
    torch::Tensor g_E_basal,         // [N]  AMPA (basal)
    torch::Tensor g_I_basal,         // [N]  GABA_A (basal)
    torch::Tensor g_GABA_B_basal,    // [N]  GABA_B (basal)
    torch::Tensor g_nmda_basal,      // [N]  NMDA (basal)
    torch::Tensor g_adapt,           // [N]  SFA conductance

    // ── Dendritic state tensors (in/out, modified in-place) ──
    torch::Tensor V_dend,            // [N]
    torch::Tensor g_E_apical,        // [N]  AMPA (apical)
    torch::Tensor g_I_apical,        // [N]  GABA_A (apical)
    torch::Tensor g_nmda_apical,     // [N]  NMDA (apical)
    torch::Tensor g_Ca,              // [N]  Ca spike conductance
    torch::Tensor g_plateau,         // [N]  NMDA plateau conductance

    // ── Noise state ──
    torch::Tensor ou_noise,          // [N]
    torch::Tensor refractory,        // [N] int32

    // ── Basal synaptic inputs ──
    torch::Tensor g_ampa_basal_in,   // [N]
    torch::Tensor g_nmda_basal_in,   // [N]
    torch::Tensor g_gaba_a_basal_in, // [N]
    torch::Tensor g_gaba_b_basal_in, // [N]

    // ── Apical synaptic inputs ──
    torch::Tensor g_ampa_apical_in,  // [N]
    torch::Tensor g_nmda_apical_in,  // [N]
    torch::Tensor g_gaba_a_apical_in,// [N]

    // ── Per-neuron decay factors ──
    torch::Tensor g_E_decay,         // [N] or scalar
    torch::Tensor g_I_decay,         // [N] or scalar
    torch::Tensor g_nmda_decay,      // [N] or scalar
    torch::Tensor g_GABA_B_decay,    // [N] or scalar
    torch::Tensor g_Ca_decay,        // [N] or scalar
    torch::Tensor g_plateau_decay,   // [N] or scalar
    torch::Tensor adapt_decay,       // [N] or scalar
    torch::Tensor V_soma_decay,      // [N] per-neuron base decay
    torch::Tensor V_dend_decay,      // [N] or scalar

    // ── Per-neuron parameters ──
    torch::Tensor g_L,               // [N]
    torch::Tensor g_L_scale,         // [N]
    torch::Tensor v_threshold,       // [N]
    torch::Tensor adapt_increment,   // [N]
    torch::Tensor tau_ref_per_neuron,// [N]

    // ── Scalar-or-per-neuron constants ──
    torch::Tensor v_reset,           // [N] or scalar
    torch::Tensor E_E,               // [N] or scalar
    torch::Tensor E_I,               // [N] or scalar
    torch::Tensor E_nmda,            // [N] or scalar
    torch::Tensor E_GABA_B,          // [N] or scalar
    torch::Tensor E_adapt,           // [N] or scalar
    torch::Tensor E_Ca,              // [N] or scalar
    torch::Tensor nmda_a,            // [N] or scalar  Pre-computed Mg²⁺ block constant
    torch::Tensor nmda_b,            // [N] or scalar  Pre-computed Mg²⁺ block constant
    torch::Tensor g_nmda_max,        // [N] or scalar  NMDA saturation ceiling (inf = disabled)
    float dt_ms,

    // ── Two-compartment coupling parameters ──
    float g_c,                       // Coupling conductance (scalar config)
    float g_L_dend,                  // Dendritic leak conductance (scalar config)
    float bap_amplitude,             // BAP fraction (scalar config)
    float theta_Ca,                  // Ca spike threshold (scalar config)
    float g_Ca_spike,                // Peak Ca conductance on Ca spike (scalar config)

    // ── NMDA plateau potential parameters ──
    bool enable_nmda_plateau,        // Whether plateau is enabled
    float nmda_plateau_threshold,    // Min effective NMDA conductance to trigger
    float v_dend_plateau_threshold,  // Min dendritic voltage for initiation
    float g_nmda_plateau,            // Plateau conductance increment

    // ── Noise parameters ──
    bool enable_noise,
    torch::Tensor neuron_seeds,      // [N] int64
    int64_t rng_timestep,
    torch::Tensor ou_decay,          // [N] or scalar
    torch::Tensor ou_std,            // [N] or scalar

    // ── Optional: gap junctions (soma) ──
    bool has_gap_junctions,
    torch::Tensor g_gap_input,       // [N] or empty
    torch::Tensor E_gap_reversal     // [N] or empty
) {
    TORCH_CHECK(V_soma.is_cpu(), "All tensors must be CPU");
    const int64_t N = V_soma.numel();

    // Ensure contiguity for raw pointer access — state tensors
    auto V_soma_c       = V_soma.contiguous();
    auto g_E_basal_c    = g_E_basal.contiguous();
    auto g_I_basal_c    = g_I_basal.contiguous();
    auto g_GABA_B_b_c   = g_GABA_B_basal.contiguous();
    auto g_nmda_basal_c = g_nmda_basal.contiguous();
    auto g_adapt_c      = g_adapt.contiguous();
    auto V_dend_c       = V_dend.contiguous();
    auto g_E_apical_c   = g_E_apical.contiguous();
    auto g_I_apical_c   = g_I_apical.contiguous();
    auto g_nmda_apic_c  = g_nmda_apical.contiguous();
    auto g_Ca_c         = g_Ca.contiguous();
    auto g_plateau_c    = g_plateau.contiguous();
    auto ou_noise_c     = ou_noise.contiguous();
    auto refrac_c       = refractory.contiguous();

    // Output: boolean spike tensor
    auto spikes = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    // Raw pointers — somatic state (in/out)
    float* RESTRICT pVs       = V_soma_c.data_ptr<float>();
    float* RESTRICT pgEb      = g_E_basal_c.data_ptr<float>();
    float* RESTRICT pgIb      = g_I_basal_c.data_ptr<float>();
    float* RESTRICT pgGBb     = g_GABA_B_b_c.data_ptr<float>();
    float* RESTRICT pgNmb     = g_nmda_basal_c.data_ptr<float>();
    float* RESTRICT pgAd      = g_adapt_c.data_ptr<float>();

    // Raw pointers — dendritic state (in/out)
    float* RESTRICT pVd       = V_dend_c.data_ptr<float>();
    float* RESTRICT pgEa      = g_E_apical_c.data_ptr<float>();
    float* RESTRICT pgIa      = g_I_apical_c.data_ptr<float>();
    float* RESTRICT pgNma     = g_nmda_apic_c.data_ptr<float>();
    float* RESTRICT pgCa      = g_Ca_c.data_ptr<float>();
    float* RESTRICT pgPlat    = g_plateau_c.data_ptr<float>();

    float* RESTRICT pOuNoise  = ou_noise_c.data_ptr<float>();
    int32_t* RESTRICT pRefrac = refrac_c.data_ptr<int32_t>();
    bool* RESTRICT pSpikes    = spikes.data_ptr<bool>();

    // Raw pointers — basal inputs
    const float* RESTRICT pAmpaB  = g_ampa_basal_in.contiguous().data_ptr<float>();
    const float* RESTRICT pNmdaB  = g_nmda_basal_in.contiguous().data_ptr<float>();
    const float* RESTRICT pGabaAB = g_gaba_a_basal_in.contiguous().data_ptr<float>();
    const float* RESTRICT pGabaBB = g_gaba_b_basal_in.contiguous().data_ptr<float>();

    // Raw pointers — apical inputs
    const float* RESTRICT pAmpaA  = g_ampa_apical_in.contiguous().data_ptr<float>();
    const float* RESTRICT pNmdaA  = g_nmda_apical_in.contiguous().data_ptr<float>();
    const float* RESTRICT pGabaAA = g_gaba_a_apical_in.contiguous().data_ptr<float>();

    // Scalar-or-per-neuron parameters
    PARAM_SETUP(gE_decay, g_E_decay);
    PARAM_SETUP(gI_decay, g_I_decay);
    PARAM_SETUP(gNmda_decay, g_nmda_decay);
    PARAM_SETUP(gGABAB_decay, g_GABA_B_decay);
    PARAM_SETUP(gCa_decay, g_Ca_decay);
    PARAM_SETUP(gPlateau_decay, g_plateau_decay);
    PARAM_SETUP(adapt_decay, adapt_decay);
    PARAM_SETUP(vDend_decay, V_dend_decay);

    PARAM_SETUP(v_reset, v_reset);
    PARAM_SETUP(E_E, E_E);
    PARAM_SETUP(E_I, E_I);
    PARAM_SETUP(E_nmda, E_nmda);
    PARAM_SETUP(E_GABA_B, E_GABA_B);
    PARAM_SETUP(E_adapt, E_adapt);
    PARAM_SETUP(E_Ca, E_Ca);
    PARAM_SETUP(nmda_a, nmda_a);
    PARAM_SETUP(nmda_b, nmda_b);
    PARAM_SETUP(g_nmda_max, g_nmda_max);
    PARAM_SETUP(ou_decay, ou_decay);
    PARAM_SETUP(ou_std, ou_std);

    const float* RESTRICT pVSDecay   = V_soma_decay.contiguous().data_ptr<float>();
    const float* RESTRICT pgL        = g_L.contiguous().data_ptr<float>();
    const float* RESTRICT pgLScale   = g_L_scale.contiguous().data_ptr<float>();
    const float* RESTRICT pVThresh   = v_threshold.contiguous().data_ptr<float>();
    const float* RESTRICT pAdaptIncr = adapt_increment.contiguous().data_ptr<float>();
    const float* RESTRICT pTauRef    = tau_ref_per_neuron.contiguous().data_ptr<float>();

    // Optional pointers
    const int64_t* RESTRICT pSeeds = enable_noise ? neuron_seeds.contiguous().data_ptr<int64_t>() : nullptr;
    const float* RESTRICT pGapG    = has_gap_junctions ? g_gap_input.contiguous().data_ptr<float>() : nullptr;
    const float* RESTRICT pGapE    = has_gap_junctions ? E_gap_reversal.contiguous().data_ptr<float>() : nullptr;

    // Grain size: each neuron touches ~35 floats ≈ 140 bytes.
    // 128 neurons ≈ 17.5 KB — fits in L1 cache.
    at::parallel_for(0, N, /*grain=*/128, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            // ── Load somatic state ──
            float vs   = pVs[i];
            float gEb  = pgEb[i];
            float gIb  = pgIb[i];
            float gGBb = pgGBb[i];
            float gNmb = pgNmb[i];
            float gAd  = pgAd[i];

            // ── Load dendritic state ──
            float vd   = pVd[i];
            float gEa  = pgEa[i];
            float gIa  = pgIa[i];
            float gNma = pgNma[i];
            float gCa  = pgCa[i];
            float gPlat = pgPlat[i];

            // ══════════════════════════════════════════════════════
            // Conductance decay + input addition + clamp
            // ══════════════════════════════════════════════════════

            // Basal conductances
            float dE  = PARAM(gE_decay, i);
            float dI  = PARAM(gI_decay, i);
            float dNm = PARAM(gNmda_decay, i);
            float dGB = PARAM(gGABAB_decay, i);
            float dAd = PARAM(adapt_decay, i);

            gEb  = std::max(gEb  * dE  + pAmpaB[i],  0.0f);
            gNmb = std::max(gNmb * dNm + pNmdaB[i],  0.0f);
            gIb  = std::max(gIb  * dI  + pGabaAB[i], 0.0f);
            gGBb = std::max(gGBb * dGB + pGabaBB[i], 0.0f);

            // Apical conductances (same E/I/NMDA decay constants)
            gEa  = std::max(gEa  * dE  + pAmpaA[i],  0.0f);
            gNma = std::max(gNma * dNm + pNmdaA[i],  0.0f);
            gIa  = std::max(gIa  * dI  + pGabaAA[i], 0.0f);

            // NMDA receptor saturation (Michaelis-Menten)
            {
                float gnm_max = PARAM(g_nmda_max, i);
                if (gnm_max < 1.0e30f) {
                    gNmb = gNmb / (1.0f + gNmb / gnm_max);
                    gNma = gNma / (1.0f + gNma / gnm_max);
                }
            }

            // Ca spike conductance (decay only)
            float dCa = PARAM(gCa_decay, i);
            gCa = std::max(gCa * dCa, 0.0f);

            // NMDA plateau conductance (decay only; activation after Mg²⁺ block)
            if (enable_nmda_plateau) {
                float dPlat = PARAM(gPlateau_decay, i);
                gPlat = std::max(gPlat * dPlat, 0.0f);
            }

            // Adaptation decay (no input, no clamp)
            gAd *= dAd;

            // ══════════════════════════════════════════════════════
            // NMDA Mg²⁺ block
            // ══════════════════════════════════════════════════════
            float na = PARAM(nmda_a, i);
            float nb = PARAM(nmda_b, i);

            // Basal NMDA: block at somatic voltage
            float mg_block_soma = 1.0f / (1.0f + std::expf(-(na + nb * vs)));
            float gNmb_eff = gNmb * mg_block_soma;

            // Apical NMDA: block at dendritic voltage (biologically correct)
            float mg_block_dend = 1.0f / (1.0f + std::expf(-(na + nb * vd)));
            float gNma_eff = gNma * mg_block_dend;

            // NMDA plateau activation
            if (enable_nmda_plateau) {
                if (gNma_eff >= nmda_plateau_threshold && vd >= v_dend_plateau_threshold) {
                    gPlat += g_nmda_plateau;
                }
            }

            // ══════════════════════════════════════════════════════
            // Homeostatic leak conductance
            // ══════════════════════════════════════════════════════
            float gL_soma_eff = pgL[i] * pgLScale[i];

            // ══════════════════════════════════════════════════════
            // SOMATIC compartment dynamics
            // ══════════════════════════════════════════════════════
            float ee      = PARAM(E_E, i);
            float ei      = PARAM(E_I, i);
            float enmda   = PARAM(E_nmda, i);
            float egabab  = PARAM(E_GABA_B, i);
            float eadapt  = PARAM(E_adapt, i);

            float g_soma_total = gL_soma_eff + gEb + gNmb_eff + gIb + gGBb + gAd + g_c;
            float V_soma_inf_num =
                  gEb      * ee
                + gNmb_eff * enmda
                + gIb      * ei
                + gGBb     * egabab
                + gAd      * eadapt
                + g_c      * vd;   // coupling: target = dendritic voltage

            // Gap junctions on soma
            if (has_gap_junctions) {
                float gg = pGapG[i];
                g_soma_total   += gg;
                V_soma_inf_num += gg * pGapE[i];
            }

            float V_soma_inf = V_soma_inf_num / g_soma_total;
            float V_soma_decay_eff = std::powf(pVSDecay[i], g_soma_total / gL_soma_eff);
            float new_vs = V_soma_inf + (vs - V_soma_inf) * V_soma_decay_eff;

            // ══════════════════════════════════════════════════════
            // DENDRITIC compartment dynamics
            // ══════════════════════════════════════════════════════
            float eca = PARAM(E_Ca, i);

            float g_dend_total = g_L_dend + gEa + gIa + gNma_eff + gCa + g_c;
            float V_dend_inf_num =
                  gEa      * ee
                + gIa      * ei
                + gNma_eff * enmda
                + gCa      * eca    // Ca spike drives toward E_Ca
                + g_c      * vs;    // coupling: target = somatic voltage

            // NMDA plateau: sustained depolarizing conductance driving toward E_nmda
            if (enable_nmda_plateau) {
                g_dend_total   += gPlat;
                V_dend_inf_num += gPlat * enmda;
            }

            float V_dend_inf = V_dend_inf_num / g_dend_total;
            float V_dend_decay_eff = std::powf(PARAM(vDend_decay, i), g_dend_total / g_L_dend);
            float new_vd = V_dend_inf + (vd - V_dend_inf) * V_dend_decay_eff;

            // ══════════════════════════════════════════════════════
            // OU noise on soma
            // ══════════════════════════════════════════════════════
            if (enable_noise) {
                float noise = philox_gaussian_scalar(pSeeds[i] + rng_timestep);
                float ou = pOuNoise[i];
                ou = ou * PARAM(ou_decay, i) + noise * PARAM(ou_std, i);
                new_vs += ou;
                pOuNoise[i] = ou;
            }

            // Update voltages
            pVs[i] = new_vs;
            pVd[i] = new_vd;

            // ══════════════════════════════════════════════════════
            // Ca spike check (BEFORE spike check)
            // ══════════════════════════════════════════════════════
            if (new_vd >= theta_Ca) {
                gCa += g_Ca_spike;
            }

            // ══════════════════════════════════════════════════════
            // Refractory countdown
            // ══════════════════════════════════════════════════════
            int32_t ref = pRefrac[i];
            ref = std::max(ref - 1, 0);

            // ══════════════════════════════════════════════════════
            // Spike detection (somatic)
            // ══════════════════════════════════════════════════════
            bool not_refractory = (ref == 0);
            bool spiked = (new_vs >= pVThresh[i]) && not_refractory;

            if (spiked) {
                pVs[i] = PARAM(v_reset, i);
                ref = static_cast<int32_t>(pTauRef[i] / dt_ms);
                gAd += pAdaptIncr[i];
                // BAP: retrograde depolarisation of dendrite
                // V_dend += bap_amplitude * (E_Ca - V_dend)
                float bap_dv = bap_amplitude * (eca - new_vd);
                pVd[i] = new_vd + bap_dv;
            }

            // ── Write back state ──
            pgEb[i]  = gEb;
            pgNmb[i] = gNmb;
            pgIb[i]  = gIb;
            pgGBb[i] = gGBb;
            pgAd[i]  = gAd;
            pgEa[i]  = gEa;
            pgNma[i] = gNma;
            pgIa[i]  = gIa;
            pgCa[i]  = gCa;
            pgPlat[i] = gPlat;
            pRefrac[i] = ref;
            pSpikes[i] = spiked;
        }
    });

    return spikes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("two_compartment_lif_step_cpp", &two_compartment_lif_step_cpp,
          "Fused TwoCompartmentLIF neuron step — CPU, parallelised",
          py::call_guard<py::gil_scoped_release>());
}
