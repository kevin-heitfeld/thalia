/*
 * Fused C++ kernel for cortical inhibitory network synaptic integration.
 *
 * Replaces 16 separate Python-level _integrate_single_synaptic_input() calls
 * (each doing STP-efficacy lookup + torch.mv + clamp) with a single C++ call
 * that exploits spike sparsity for significant speedup.
 *
 * Connection order (fixed, matches Python-side cache):
 *   E→I:  [0] Pyr→PV   [1] Pyr→SST  [2] Pyr→VIP  [3] Pyr→NGC
 *   I→I:  [4] PV→PV    [5] SST→PV   [6] VIP→PV   [7] PV→SST
 *         [8] VIP→SST_A [9] VIP→SST_B [10] SST→VIP [11] SST→SST
 *   I→E: [12] PV→Pyr  [13] SST→Pyr  [14] VIP→Pyr  [15] NGC→Pyr
 *
 * Connections 0-14 have STP (efficacy vectors provided).
 * Connection 15 (NGC→Pyr) has no STP (volume transmission).
 *
 * Build: torch.utils.cpp_extension.load() (same pattern as stp_kernel.cpp)
 */

#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace {

/*
 * Gather indices of active (spiking) neurons from a float spike vector.
 * Spikes are 0.0 or 1.0; we threshold at 0.5.
 */
struct ActiveSpikes {
    std::vector<int32_t> idx;
    int32_t n;
};

inline ActiveSpikes gather_active(const float* spikes, int32_t size) {
    ActiveSpikes as;
    as.idx.reserve(std::max(size / 8, 8));  // expect ~5-15% sparsity
    for (int32_t i = 0; i < size; ++i) {
        if (spikes[i] > 0.5f) {
            as.idx.push_back(i);
        }
    }
    as.n = static_cast<int32_t>(as.idx.size());
    return as;
}

/*
 * Sparse STP-modulated matrix-vector multiply with clamp to [0, +inf).
 *
 * Computes: out[j] += max(0, sum_k W[j, active[k]] * eff[active[k]])
 *
 * When eff is nullptr (no STP), computes: out[j] += max(0, sum_k W[j, active[k]])
 *
 * Exploits that spike vectors are very sparse (typically 1-10% active),
 * reducing work proportionally compared to dense torch.mv.
 */
inline void sparse_stp_mv_add(
    float* out,
    const float* W,
    const float* eff,           // nullptr for no-STP connections
    const int32_t* active,
    int32_t n_active,
    int32_t n_out,
    int32_t row_stride
) {
    if (n_active == 0) return;  // no spikes → zero contribution (output pre-zeroed)

    for (int32_t j = 0; j < n_out; ++j) {
        const float* row = W + static_cast<int64_t>(j) * row_stride;
        float sum = 0.0f;
        if (eff) {
            for (int32_t k = 0; k < n_active; ++k) {
                const int32_t i = active[k];
                sum += row[i] * eff[i];
            }
        } else {
            for (int32_t k = 0; k < n_active; ++k) {
                sum += row[active[k]];
            }
        }
        // Clamp per-connection result before accumulating (matches Python semantics)
        out[j] += std::max(sum, 0.0f);
    }
}

}  // namespace


/*
 * Main kernel: fuses all 15 STP-modulated matmuls for one cortical inhibitory
 * network layer into a single C++ call with sparse spike exploitation.
 *
 * Returns 13 tensors:
 *   [0]  pv_g_exc         [pv_n]   — E→I AMPA to PV
 *   [1]  sst_g_exc        [sst_n]  — E→I AMPA to SST
 *   [2]  vip_g_exc        [vip_n]  — E→I AMPA to VIP
 *   [3]  ngc_g_exc        [ngc_n]  — E→I AMPA to NGC
 *   [4]  pv_g_gaba_a      [pv_n]   — I→I GABA_A to PV  (PV→PV + SST→PV + VIP→PV)
 *   [5]  sst_g_gaba_a     [sst_n]  — I→I GABA_A to SST (PV→SST + VIP→SST_A + SST→SST)
 *   [6]  sst_g_gaba_b     [sst_n]  — I→I GABA_B to SST (VIP→SST_B)
 *   [7]  vip_g_gaba_a     [vip_n]  — I→I GABA_A to VIP (SST→VIP)
 *   [8]  perisomatic      [pyr_n]  — I→E PV→Pyr GABA_A
 *   [9]  dendritic        [pyr_n]  — I→E SST→Pyr GABA_A
 *  [10]  vip_to_pyr       [pyr_n]  — I→E VIP→Pyr GABA_A
 *  [11]  ngc_to_pyr       [pyr_n]  — I→E NGC→Pyr GABA_A
 *  [12]  total_inhibition [pyr_n]  — sum of [8]+[9]+[10]+[11]
 */
std::vector<torch::Tensor> cortical_inhibitory_step_cpp(
    // Spike vectors (float32, values 0.0 or 1.0)
    torch::Tensor pyr_f,    // [pyr_n]
    torch::Tensor pv_f,     // [pv_n]
    torch::Tensor sst_f,    // [sst_n]
    torch::Tensor vip_f,    // [vip_n]
    torch::Tensor ngc_f,    // [ngc_n]
    // 15 weight matrices  [tgt_size, src_size] in connection order
    std::vector<torch::Tensor> weights,
    // 14 STP efficacy vectors (connections 0-13; connection 14 has no STP)
    std::vector<torch::Tensor> efficacies
) {
    TORCH_CHECK(weights.size() == 16, "Expected 16 weight matrices, got ", weights.size());
    TORCH_CHECK(efficacies.size() == 15, "Expected 15 efficacy vectors, got ", efficacies.size());

    const int32_t pyr_n = static_cast<int32_t>(pyr_f.size(0));
    const int32_t pv_n  = static_cast<int32_t>(pv_f.size(0));
    const int32_t sst_n = static_cast<int32_t>(sst_f.size(0));
    const int32_t vip_n = static_cast<int32_t>(vip_f.size(0));
    const int32_t ngc_n = static_cast<int32_t>(ngc_f.size(0));

    // Ensure contiguous float32
    auto pyr_c = pyr_f.contiguous();
    auto pv_c  = pv_f.contiguous();
    auto sst_c = sst_f.contiguous();
    auto vip_c = vip_f.contiguous();
    auto ngc_c = ngc_f.contiguous();

    const float* pyr_ptr = pyr_c.data_ptr<float>();
    const float* pv_ptr  = pv_c.data_ptr<float>();
    const float* sst_ptr = sst_c.data_ptr<float>();
    const float* vip_ptr = vip_c.data_ptr<float>();
    const float* ngc_ptr = ngc_c.data_ptr<float>();

    // Gather active spike indices for each source population (once, reused)
    auto active_pyr = gather_active(pyr_ptr, pyr_n);
    auto active_pv  = gather_active(pv_ptr,  pv_n);
    auto active_sst = gather_active(sst_ptr, sst_n);
    auto active_vip = gather_active(vip_ptr, vip_n);
    auto active_ngc = gather_active(ngc_ptr, ngc_n);

    // Get weight matrix data pointers (contiguous row-major)
    const float* w[16];
    for (int i = 0; i < 16; ++i) {
        auto wc = weights[i].contiguous();
        weights[i] = wc;  // keep alive
        w[i] = wc.data_ptr<float>();
    }

    // Get efficacy data pointers (15 connections with STP)
    const float* eff[15];
    for (int i = 0; i < 15; ++i) {
        auto ec = efficacies[i].contiguous();
        efficacies[i] = ec;  // keep alive
        eff[i] = ec.data_ptr<float>();
    }

    // Source sizes (= row_stride for weight matrices)
    // Connection → source mapping:
    //   0-3:     pyr (stride=pyr_n)
    //   4:       pv  (stride=pv_n)
    //   5,10,11: sst (stride=sst_n)
    //   6,8,9:   vip (stride=vip_n)
    //   7,12:    pv  (stride=pv_n)
    //   13:      sst (stride=sst_n)
    //   14:      vip (stride=vip_n)
    //   15:      ngc (stride=ngc_n)

    // Allocate zero-initialised output tensors
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(pyr_f.device());
    auto pv_g_exc      = torch::zeros({pv_n},  opts);
    auto sst_g_exc     = torch::zeros({sst_n}, opts);
    auto vip_g_exc     = torch::zeros({vip_n}, opts);
    auto ngc_g_exc     = torch::zeros({ngc_n}, opts);
    auto pv_g_gaba_a   = torch::zeros({pv_n},  opts);
    auto sst_g_gaba_a  = torch::zeros({sst_n}, opts);
    auto sst_g_gaba_b  = torch::zeros({sst_n}, opts);
    auto vip_g_gaba_a  = torch::zeros({vip_n}, opts);
    auto perisomatic    = torch::zeros({pyr_n}, opts);
    auto dendritic      = torch::zeros({pyr_n}, opts);
    auto vip_to_pyr     = torch::zeros({pyr_n}, opts);
    auto ngc_to_pyr     = torch::zeros({pyr_n}, opts);
    auto total_inhib    = torch::zeros({pyr_n}, opts);

    float* out_pv_exc     = pv_g_exc.data_ptr<float>();
    float* out_sst_exc    = sst_g_exc.data_ptr<float>();
    float* out_vip_exc    = vip_g_exc.data_ptr<float>();
    float* out_ngc_exc    = ngc_g_exc.data_ptr<float>();
    float* out_pv_gaba    = pv_g_gaba_a.data_ptr<float>();
    float* out_sst_gaba_a = sst_g_gaba_a.data_ptr<float>();
    float* out_sst_gaba_b = sst_g_gaba_b.data_ptr<float>();
    float* out_vip_gaba   = vip_g_gaba_a.data_ptr<float>();
    float* out_peri       = perisomatic.data_ptr<float>();
    float* out_dend       = dendritic.data_ptr<float>();
    float* out_v2p        = vip_to_pyr.data_ptr<float>();
    float* out_n2p        = ngc_to_pyr.data_ptr<float>();
    float* out_total      = total_inhib.data_ptr<float>();

    // =====================================================================
    // E → I  (4 connections, all use pyr spikes, all have STP)
    // =====================================================================
    // [0] Pyr→PV:   pv_g_exc  = clamp(W[0] @ (eff[0] * pyr), 0)
    sparse_stp_mv_add(out_pv_exc,  w[0], eff[0],
        active_pyr.idx.data(), active_pyr.n, pv_n, pyr_n);
    // [1] Pyr→SST:  sst_g_exc = clamp(W[1] @ (eff[1] * pyr), 0)
    sparse_stp_mv_add(out_sst_exc, w[1], eff[1],
        active_pyr.idx.data(), active_pyr.n, sst_n, pyr_n);
    // [2] Pyr→VIP:  vip_g_exc = clamp(W[2] @ (eff[2] * pyr), 0)
    sparse_stp_mv_add(out_vip_exc, w[2], eff[2],
        active_pyr.idx.data(), active_pyr.n, vip_n, pyr_n);
    // [3] Pyr→NGC:  ngc_g_exc = clamp(W[3] @ (eff[3] * pyr), 0)
    sparse_stp_mv_add(out_ngc_exc, w[3], eff[3],
        active_pyr.idx.data(), active_pyr.n, ngc_n, pyr_n);

    // =====================================================================
    // I → I  (7 connections)
    // =====================================================================
    // PV receives: PV→PV + SST→PV + VIP→PV  (3 contributions to pv_g_gaba_a)
    // [4] PV→PV
    sparse_stp_mv_add(out_pv_gaba, w[4], eff[4],
        active_pv.idx.data(), active_pv.n, pv_n, pv_n);
    // [5] SST→PV
    sparse_stp_mv_add(out_pv_gaba, w[5], eff[5],
        active_sst.idx.data(), active_sst.n, pv_n, sst_n);
    // [6] VIP→PV
    sparse_stp_mv_add(out_pv_gaba, w[6], eff[6],
        active_vip.idx.data(), active_vip.n, pv_n, vip_n);

    // SST receives: PV→SST + VIP→SST_A  (2 contributions to sst_g_gaba_a)
    // [7] PV→SST
    sparse_stp_mv_add(out_sst_gaba_a, w[7], eff[7],
        active_pv.idx.data(), active_pv.n, sst_n, pv_n);
    // [8] VIP→SST (GABA_A)
    sparse_stp_mv_add(out_sst_gaba_a, w[8], eff[8],
        active_vip.idx.data(), active_vip.n, sst_n, vip_n);

    // [9] VIP→SST (GABA_B) — separate output
    sparse_stp_mv_add(out_sst_gaba_b, w[9], eff[9],
        active_vip.idx.data(), active_vip.n, sst_n, vip_n);

    // VIP receives: SST→VIP  (1 contribution)
    // [10] SST→VIP
    sparse_stp_mv_add(out_vip_gaba, w[10], eff[10],
        active_sst.idx.data(), active_sst.n, vip_n, sst_n);

    // SST receives: SST→SST  (mutual inhibition among Martinotti cells)
    // [11] SST→SST
    sparse_stp_mv_add(out_sst_gaba_a, w[11], eff[11],
        active_sst.idx.data(), active_sst.n, sst_n, sst_n);

    // =====================================================================
    // I → E  (4 connections)
    // =====================================================================
    // [12] PV→Pyr (perisomatic)
    sparse_stp_mv_add(out_peri, w[12], eff[12],
        active_pv.idx.data(), active_pv.n, pyr_n, pv_n);
    // [13] SST→Pyr (dendritic)
    sparse_stp_mv_add(out_dend, w[13], eff[13],
        active_sst.idx.data(), active_sst.n, pyr_n, sst_n);
    // [14] VIP→Pyr
    sparse_stp_mv_add(out_v2p, w[14], eff[14],
        active_vip.idx.data(), active_vip.n, pyr_n, vip_n);
    // [15] NGC→Pyr (no STP — nullptr for efficacy)
    sparse_stp_mv_add(out_n2p, w[15], nullptr,
        active_ngc.idx.data(), active_ngc.n, pyr_n, ngc_n);

    // =====================================================================
    // Total inhibition = perisomatic + dendritic + vip_to_pyr + ngc_to_pyr
    // =====================================================================
    for (int32_t j = 0; j < pyr_n; ++j) {
        out_total[j] = out_peri[j] + out_dend[j] + out_v2p[j] + out_n2p[j];
    }

    return {
        pv_g_exc, sst_g_exc, vip_g_exc, ngc_g_exc,
        pv_g_gaba_a, sst_g_gaba_a, sst_g_gaba_b, vip_g_gaba_a,
        perisomatic, dendritic, vip_to_pyr, ngc_to_pyr, total_inhib
    };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cortical_inhibitory_step_cpp", &cortical_inhibitory_step_cpp,
          "Fused cortical inhibitory network synaptic integration — sparse, CPU",
          py::call_guard<py::gil_scoped_release>());
}
