/*
 * Fused C++ kernel for batched Short-Term Plasticity (STP) step.
 *
 * Tsodyks-Markram model: all STP instances across the brain are
 * concatenated into contiguous arrays and processed in one parallelised loop.
 * Eliminates ~377 Python calls/step × 10 tensor ops each.
 *
 * Build: torch.utils.cpp_extension.load() (same pattern as conductance_lif_kernel.cpp)
 */

#include <torch/extension.h>
#include <ATen/Parallel.h>

torch::Tensor stp_step_cpp(
    torch::Tensor u,            // [N] in/out — release probability
    torch::Tensor x,            // [N] in/out — available resources
    torch::Tensor U,            // [N] baseline release probability
    torch::Tensor decay_d,      // [N] depression decay  = exp(-dt/tau_d)
    torch::Tensor decay_f,      // [N] facilitation decay = exp(-dt/tau_f)
    torch::Tensor recovery_d,   // [N] = 1 - decay_d
    torch::Tensor recovery_f,   // [N] = U * (1 - decay_f)
    torch::Tensor pre_spikes,   // [N] presynaptic spikes (float, 0 or 1)
    int64_t N
) {
    // Output: pre-spike efficacy u*x (before spike-triggered update)
    auto efficacy = torch::empty({N}, u.options());

    // Raw pointers for branchless inner loop
    float* u_ptr   = u.contiguous().data_ptr<float>();
    float* x_ptr   = x.contiguous().data_ptr<float>();
    const float* U_ptr   = U.contiguous().data_ptr<float>();
    const float* dd_ptr  = decay_d.contiguous().data_ptr<float>();
    const float* df_ptr  = decay_f.contiguous().data_ptr<float>();
    const float* rd_ptr  = recovery_d.contiguous().data_ptr<float>();
    const float* rf_ptr  = recovery_f.contiguous().data_ptr<float>();
    const float* sp_ptr  = pre_spikes.contiguous().data_ptr<float>();
    float* eff_ptr = efficacy.data_ptr<float>();

    at::parallel_for(0, N, /*grain=*/256, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            // --- Continuous dynamics ---
            // u decays toward U:  u = u * decay_f + U * (1 - decay_f)
            float ui = u_ptr[i] * df_ptr[i] + rf_ptr[i];
            // x recovers toward 1: x = x * decay_d + (1 - decay_d)
            float xi = x_ptr[i] * dd_ptr[i] + rd_ptr[i];

            // --- Pre-spike efficacy ---
            float eff = ui * xi;
            eff_ptr[i] = eff;

            // --- Spike-triggered dynamics ---
            float s = sp_ptr[i];
            if (s > 0.0f) {
                // Depression: x -= spike * efficacy  (uses OLD u via eff)
                xi -= s * eff;
                // Facilitation: u += spike * U * (1 - u)
                ui += s * U_ptr[i] * (1.0f - ui);
            }

            // --- Clamp to [0, 1] ---
            u_ptr[i] = ui < 0.0f ? 0.0f : (ui > 1.0f ? 1.0f : ui);
            x_ptr[i] = xi < 0.0f ? 0.0f : (xi > 1.0f ? 1.0f : xi);
        }
    });

    return efficacy;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stp_step_cpp", &stp_step_cpp, "Batched STP step (Tsodyks-Markram, C++)",
          py::call_guard<py::gil_scoped_release>());
}
