/**
 * Fused ConductanceLIF neuron step — CPU kernel for Thalia.
 *
 * Replaces the ~40 individual PyTorch tensor operations in
 * ConductanceLIF.forward() with a single C++ loop parallelised via
 * at::parallel_for.  All state buffers are updated in-place.
 *
 * Eliminates:
 *   - ~40 per-call PyTorch dispatcher round-trips (mul_, add_, clamp_, …)
 *   - nn.Module.__getattr__ lookups (direct pointer access)
 *   - Temporary tensor allocations (e.g., mg_block, V_dend_est, g_total)
 *   - Separate philox_gaussian call (noise is fused inline)
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
// These macros handle parameters that may be either 0-dim scalar tensors
// (broadcast to all neurons) or 1D per-neuron tensors.
// PARAM_SETUP extracts the scalar value or data pointer at kernel entry.
// PARAM reads the appropriate value inside the per-neuron loop.
#define PARAM_SETUP(varname, tensor) \
    auto varname##_c = (tensor).contiguous(); \
    const bool varname##_is_scalar = (varname##_c.dim() == 0); \
    const float varname##_sval = varname##_is_scalar ? varname##_c.item<float>() : 0.0f; \
    const float* RESTRICT varname##_ptr = varname##_is_scalar ? nullptr : varname##_c.data_ptr<float>()

#define PARAM(varname, i) (varname##_is_scalar ? varname##_sval : varname##_ptr[(i)])

// =========================================================================
// Fused ConductanceLIF step kernel
// =========================================================================

/**
 * conductance_lif_step_cpp(...)
 *
 * All state tensors are updated IN-PLACE.  Returns boolean spike tensor.
 *
 * This fuses the entire ConductanceLIF.forward() body into one tight loop.
 * Optional I_h and T-channel dynamics are controlled by boolean flags.
 */
torch::Tensor conductance_lif_step_cpp(
    // ── State tensors (in/out, modified in-place) ──
    torch::Tensor V_soma,          // [N]
    torch::Tensor g_E,             // [N]
    torch::Tensor g_I,             // [N]
    torch::Tensor g_nmda,          // [N]
    torch::Tensor g_GABA_B,        // [N]
    torch::Tensor g_adapt,         // [N]
    torch::Tensor ou_noise,        // [N]
    torch::Tensor refractory,      // [N] int32

    // ── Synaptic input conductances ──
    torch::Tensor g_ampa_input,    // [N]
    torch::Tensor g_nmda_input,    // [N]
    torch::Tensor g_gaba_a_input,  // [N]
    torch::Tensor g_gaba_b_input,  // [N]

    // ── Per-neuron parameters (read-only) ──
    torch::Tensor g_E_decay,       // [N] or scalar
    torch::Tensor g_I_decay,       // [N] or scalar
    torch::Tensor g_nmda_decay,    // [N] or scalar
    torch::Tensor g_GABA_B_decay,  // [N] or scalar
    torch::Tensor adapt_decay,     // [N] or scalar
    torch::Tensor V_soma_decay,    // [N] per-neuron base decay
    torch::Tensor g_L,             // [N]
    torch::Tensor g_L_scale,       // [N]
    torch::Tensor v_threshold,     // [N]
    torch::Tensor adapt_increment, // [N]
    torch::Tensor tau_ref_per_neuron, // [N]

    // ── Scalar-or-per-neuron constants ──
    torch::Tensor v_reset,         // [N] or scalar
    torch::Tensor E_E,             // [N] or scalar
    torch::Tensor E_I,             // [N] or scalar
    torch::Tensor E_nmda,          // [N] or scalar
    torch::Tensor E_GABA_B,        // [N] or scalar
    torch::Tensor E_adapt,         // [N] or scalar
    torch::Tensor dendrite_coupling_scale, // [N] or scalar
    torch::Tensor nmda_a,          // [N] or scalar  Pre-computed Mg²⁺ block constant
    torch::Tensor nmda_b,          // [N] or scalar  Pre-computed Mg²⁺ block constant
    torch::Tensor g_nmda_max,      // [N] or scalar  NMDA saturation ceiling (inf = disabled)
    float dt_ms,

    // ── Noise parameters ──
    bool enable_noise,
    torch::Tensor neuron_seeds,    // [N] int64
    int64_t rng_timestep,          // current timestep
    torch::Tensor ou_decay,        // [N] or scalar
    torch::Tensor ou_std,          // [N] or scalar

    // ── Optional: gap junctions ──
    bool has_gap_junctions,
    torch::Tensor g_gap_input,     // [N] or empty
    torch::Tensor E_gap_reversal,  // [N] or empty

    // ── Optional: T-channels ──
    bool enable_t_channels,
    torch::Tensor h_T,             // [N] or empty — state, modified in-place
    torch::Tensor h_T_decay,       // [N] or scalar
    torch::Tensor g_T,             // [N] or scalar  T-channel conductance
    torch::Tensor E_Ca,            // [N] or scalar  Calcium reversal
    torch::Tensor V_half_h_T,      // [N] or scalar  T-channel h gate half-activation
    torch::Tensor k_h_T,           // [N] or scalar  T-channel h gate slope

    // ── Optional: I_h (HCN) ──
    bool enable_ih,
    torch::Tensor h_gate,          // [N] or empty — state, modified in-place
    torch::Tensor h_decay,         // [N] or scalar
    torch::Tensor g_h_max,         // [N] or scalar  Maximum HCN conductance
    torch::Tensor E_h,             // [N] or scalar  HCN reversal
    torch::Tensor V_half_h,        // [N] or scalar  HCN half-activation voltage
    torch::Tensor k_h              // [N] or scalar  HCN slope factor
) {
    // Validate all tensors are CPU and contiguous
    TORCH_CHECK(V_soma.is_cpu(), "All tensors must be CPU");
    const int64_t N = V_soma.numel();

    // Ensure contiguity for raw pointer access
    auto V_soma_c   = V_soma.contiguous();
    auto g_E_c      = g_E.contiguous();
    auto g_I_c      = g_I.contiguous();
    auto g_nmda_c   = g_nmda.contiguous();
    auto g_GABA_B_c = g_GABA_B.contiguous();
    auto g_adapt_c  = g_adapt.contiguous();
    auto ou_noise_c = ou_noise.contiguous();
    auto refrac_c   = refractory.contiguous();

    // Output: boolean spike tensor
    auto spikes = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    // Raw pointers — state (in/out)
    float* RESTRICT pV        = V_soma_c.data_ptr<float>();
    float* RESTRICT pgE       = g_E_c.data_ptr<float>();
    float* RESTRICT pgI       = g_I_c.data_ptr<float>();
    float* RESTRICT pgNmda    = g_nmda_c.data_ptr<float>();
    float* RESTRICT pgGABAB   = g_GABA_B_c.data_ptr<float>();
    float* RESTRICT pgAdapt   = g_adapt_c.data_ptr<float>();
    float* RESTRICT pOuNoise  = ou_noise_c.data_ptr<float>();
    int32_t* RESTRICT pRefrac = refrac_c.data_ptr<int32_t>();
    bool* RESTRICT pSpikes    = spikes.data_ptr<bool>();

    // Raw pointers — inputs
    const float* RESTRICT pAmpaIn  = g_ampa_input.contiguous().data_ptr<float>();
    const float* RESTRICT pNmdaIn  = g_nmda_input.contiguous().data_ptr<float>();
    const float* RESTRICT pGabaAIn = g_gaba_a_input.contiguous().data_ptr<float>();
    const float* RESTRICT pGabaBIn = g_gaba_b_input.contiguous().data_ptr<float>();

    // Scalar-or-per-neuron parameters (using PARAM_SETUP/PARAM macros)
    PARAM_SETUP(gE_decay, g_E_decay);
    PARAM_SETUP(gI_decay, g_I_decay);
    PARAM_SETUP(gNmda_decay, g_nmda_decay);
    PARAM_SETUP(gGABAB_decay, g_GABA_B_decay);
    PARAM_SETUP(adapt_decay, adapt_decay);

    PARAM_SETUP(v_reset, v_reset);
    PARAM_SETUP(E_E, E_E);
    PARAM_SETUP(E_I, E_I);
    PARAM_SETUP(E_nmda, E_nmda);
    PARAM_SETUP(E_GABA_B, E_GABA_B);
    PARAM_SETUP(E_adapt, E_adapt);
    PARAM_SETUP(dendrite_coupling_scale, dendrite_coupling_scale);
    PARAM_SETUP(nmda_a, nmda_a);
    PARAM_SETUP(nmda_b, nmda_b);
    PARAM_SETUP(g_nmda_max, g_nmda_max);
    PARAM_SETUP(ou_decay, ou_decay);
    PARAM_SETUP(ou_std, ou_std);
    PARAM_SETUP(h_T_decay, h_T_decay);
    PARAM_SETUP(g_T, g_T);
    PARAM_SETUP(E_Ca, E_Ca);
    PARAM_SETUP(V_half_h_T, V_half_h_T);
    PARAM_SETUP(k_h_T, k_h_T);
    PARAM_SETUP(h_decay, h_decay);
    PARAM_SETUP(g_h_max, g_h_max);
    PARAM_SETUP(E_h, E_h);
    PARAM_SETUP(V_half_h, V_half_h);
    PARAM_SETUP(k_h, k_h);

    const float* RESTRICT pVDecay     = V_soma_decay.contiguous().data_ptr<float>();
    const float* RESTRICT pgL         = g_L.contiguous().data_ptr<float>();
    const float* RESTRICT pgLScale    = g_L_scale.contiguous().data_ptr<float>();
    const float* RESTRICT pVThresh    = v_threshold.contiguous().data_ptr<float>();
    const float* RESTRICT pAdaptIncr  = adapt_increment.contiguous().data_ptr<float>();
    const float* RESTRICT pTauRef     = tau_ref_per_neuron.contiguous().data_ptr<float>();

    // Optional pointers
    const int64_t* RESTRICT pSeeds  = enable_noise ? neuron_seeds.contiguous().data_ptr<int64_t>() : nullptr;
    const float* RESTRICT pGapG     = has_gap_junctions ? g_gap_input.contiguous().data_ptr<float>() : nullptr;
    const float* RESTRICT pGapE     = has_gap_junctions ? E_gap_reversal.contiguous().data_ptr<float>() : nullptr;
    float* RESTRICT pHT             = enable_t_channels ? h_T.contiguous().data_ptr<float>() : nullptr;
    float* RESTRICT pHGate          = enable_ih ? h_gate.contiguous().data_ptr<float>() : nullptr;

    // T-channel activation constants
    const float V_half_m_T = -0.5f;
    const float k_m_T = 0.1f;

    // Grain size: each neuron touches ~25 floats = ~100 bytes.
    // 128 neurons ≈ 12.8 KB — fits comfortably in L1 cache.
    at::parallel_for(0, N, /*grain=*/128, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            // ── Load state ──
            float v   = pV[i];
            float gE  = pgE[i];
            float gNm = pgNmda[i];
            float gIv = pgI[i];
            float gGB = pgGABAB[i];
            float gAd = pgAdapt[i];

            // ── Conductance decay + input addition + clamp ──
            float dE  = PARAM(gE_decay, i);
            float dI  = PARAM(gI_decay, i);
            float dNm = PARAM(gNmda_decay, i);
            float dGB = PARAM(gGABAB_decay, i);
            float dAd = PARAM(adapt_decay, i);

            gE  = std::max(gE  * dE  + pAmpaIn[i],  0.0f);
            gNm = std::max(gNm * dNm + pNmdaIn[i],  0.0f);
            // NMDA receptor saturation (Michaelis-Menten)
            {
                float gnm_max = PARAM(g_nmda_max, i);
                if (gnm_max < 1.0e30f) {
                    gNm = gNm / (1.0f + gNm / gnm_max);
                }
            }
            gIv = std::max(gIv * dI  + pGabaAIn[i], 0.0f);
            gGB = std::max(gGB * dGB + pGabaBIn[i], 0.0f);

            // Adaptation decay (no input, no clamp)
            gAd *= dAd;

            // ── Homeostatic leak conductance ──
            float gL_eff = pgL[i] * pgLScale[i];

            // ── NMDA Mg²⁺ block ──
            // Dendritic voltage estimate for Mg²⁺ unblocking
            float dcs = PARAM(dendrite_coupling_scale, i);
            float ee = PARAM(E_E, i);
            float V_dend_est;
            if (dcs > 0.0f) {
                float local_ampa_drive = gE * (ee - v);
                if (local_ampa_drive < 0.0f) local_ampa_drive = 0.0f;
                V_dend_est = v + local_ampa_drive * dcs;
            } else {
                V_dend_est = v;
            }
            // mg_block = sigmoid(nmda_a + nmda_b * V_dend_est)
            float mg_block = 1.0f / (1.0f + std::expf(-(PARAM(nmda_a, i) + PARAM(nmda_b, i) * V_dend_est)));
            float g_nmda_eff = gNm * mg_block;

            // ── Total conductance ──
            float g_total = gL_eff + gE + g_nmda_eff + gIv + gGB + gAd;

            // ── Equilibrium potential numerator ──
            // E_L = 0, so g_L_eff * E_L = 0 (no term needed)
            float V_inf_num = gE * ee
                            + g_nmda_eff * PARAM(E_nmda, i)
                            + gIv * PARAM(E_I, i)
                            + gGB * PARAM(E_GABA_B, i)
                            + gAd * PARAM(E_adapt, i);

            // ── Gap junctions (optional) ──
            if (has_gap_junctions) {
                float gg = pGapG[i];
                g_total   += gg;
                V_inf_num += gg * pGapE[i];
            }

            // ── T-channels (optional, per-neuron guard via g_T > 0) ──
            // Global flag enable_t_channels is true if ANY neuron has T-channels.
            // Per-neuron guard: skip computation for neurons with g_T == 0 to
            // avoid division by zero in h_T_inf when k_h_T == 0.
            if (enable_t_channels && PARAM(g_T, i) > 0.0f) {
                float h_T_val = pHT[i];

                // Steady-state de-inactivation
                float h_T_inf = 1.0f / (1.0f + std::expf((v - PARAM(V_half_h_T, i)) / PARAM(k_h_T, i)));

                // Exponential relaxation: h_T = h_T_inf + (h_T - h_T_inf) * decay
                h_T_val = h_T_inf + (h_T_val - h_T_inf) * PARAM(h_T_decay, i);

                // T-channel activation (instantaneous)
                float m_T_inf = 1.0f / (1.0f + std::expf((V_half_m_T - v) / k_m_T));

                float g_T_eff = PARAM(g_T, i) * m_T_inf * h_T_val;
                g_total   += g_T_eff;
                V_inf_num += g_T_eff * PARAM(E_Ca, i);

                pHT[i] = h_T_val;
            }

            // ── I_h / HCN pacemaker (optional, per-neuron guard via g_h_max > 0) ──
            // Same pattern: global flag + per-neuron guard prevents division by
            // zero in h_inf when k_h == 0 for non-I_h neurons.
            if (enable_ih && PARAM(g_h_max, i) > 0.0f) {
                float hg = pHGate[i];

                // Steady-state activation (activated by hyperpolarization)
                float h_inf = 1.0f / (1.0f + std::expf((v - PARAM(V_half_h, i)) / PARAM(k_h, i)));

                // Exponential relaxation
                hg = h_inf + (hg - h_inf) * PARAM(h_decay, i);

                float g_ih = PARAM(g_h_max, i) * hg;
                g_total   += g_ih;
                V_inf_num += g_ih * PARAM(E_h, i);

                pHGate[i] = hg;
            }

            // ── Membrane voltage update ──
            float V_inf = V_inf_num / g_total;

            // Effective decay: pow(base_decay, g_total / g_L_eff)
            float V_decay_eff = std::powf(pVDecay[i], g_total / gL_eff);

            // Exponential relaxation toward V_inf
            v = V_inf + (v - V_inf) * V_decay_eff;

            // ── Ornstein-Uhlenbeck noise (optional) ──
            if (enable_noise) {
                float noise = philox_gaussian_scalar(pSeeds[i] + rng_timestep);
                float ou = pOuNoise[i];
                ou = ou * PARAM(ou_decay, i) + noise * PARAM(ou_std, i);
                v += ou;
                pOuNoise[i] = ou;
            }

            // ── Refractory countdown ──
            int32_t ref = pRefrac[i];
            ref = std::max(ref - 1, 0);

            // ── Spike detection ──
            bool not_refractory = (ref == 0);
            bool spiked = (v >= pVThresh[i]) && not_refractory;

            if (spiked) {
                v = PARAM(v_reset, i);
                ref = static_cast<int32_t>(pTauRef[i] / dt_ms);
                gAd += pAdaptIncr[i];
            }

            // ── Write back state ──
            pV[i]      = v;
            pgE[i]     = gE;
            pgNmda[i]  = gNm;
            pgI[i]     = gIv;
            pgGABAB[i] = gGB;
            pgAdapt[i] = gAd;
            pRefrac[i] = ref;
            pSpikes[i] = spiked;
        }
    });

    return spikes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conductance_lif_step_cpp", &conductance_lif_step_cpp,
          "Fused ConductanceLIF neuron step — CPU, parallelised",
          py::call_guard<py::gil_scoped_release>());
}
