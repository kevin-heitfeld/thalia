/**
 * Philox 2x32-10 counter-based RNG — CPU kernel for Thalia.
 *
 * Implements the same algorithm as rng.py but in a tight C++ scalar loop
 * parallelised with at::parallel_for, eliminating:
 *   - per-round tensor allocations
 *   - Python/TorchScript dispatch overhead
 *   - intermediate .clone() / .to() tensor copies
 *
 * Drop-in replacement for the @torch.jit.script versions in rng.py.
 * Loaded at runtime via torch.utils.cpp_extension.load().
 */

#include <torch/extension.h>
#include <ATen/Parallel.h>

// MSVC uses __restrict; GCC/Clang use __restrict__
#ifdef _MSC_VER
#  define RESTRICT __restrict
#else
#  define RESTRICT __restrict__
#endif

#include <cmath>
#include <cstdint>

// Weyl constants (must match rng.py exactly)
static constexpr uint32_t W0 = 0x9E3779B9u;
static constexpr uint32_t W1 = 0xBB67AE85u;

// Constant 2*pi as float
static constexpr float TWO_PI = 6.283185307179586f;

/**
 * Run one full 10-round Philox 2x32 pass on a single 64-bit counter value
 * and return a float in (0, 1).
 *
 * Matches the Python reference exactly:
 *   lo = (lo * W0) & 0xffffffff      (lower 32-bit word × weyl0, truncated)
 *   hi = (hi * W1) & 0xffffffff      (upper 32-bit word × weyl1, truncated)
 *   x  = (hi << 32 | lo) ^ W0*(r+1) (recombine, XOR round constant)
 */
static inline float philox_to_uniform(int64_t counter) {
    int64_t x = counter;
    for (int r = 0; r < 10; ++r) {
        uint32_t lo = static_cast<uint32_t>(x & 0xffffffff);
        uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(x) >> 32) & 0xffffffff);
        lo = static_cast<uint32_t>(static_cast<uint64_t>(lo) * W0);      // truncates to 32-bit
        hi = static_cast<uint32_t>(static_cast<uint64_t>(hi) * W1);      // truncates to 32-bit
        x  = (static_cast<int64_t>(hi) << 32) | static_cast<int64_t>(lo);
        x ^= static_cast<int64_t>(static_cast<uint64_t>(W0) * static_cast<uint64_t>(r + 1));
    }
    uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(x) & 0xffffffff);
    // Map to (0, 1): (bits + 1) / (2^32 + 2) — avoids exact 0 and 1
    return static_cast<float>(static_cast<uint64_t>(bits) + 1u) / 4294967298.0f;
}

/**
 * philox_uniform(counters) -> float32 [n]
 *
 * Args:
 *   counters: int32 or int64 CPU tensor [n], per-neuron counter values
 *
 * Returns:
 *   float32 tensor [n] with values in (0, 1), one per neuron
 */
torch::Tensor philox_uniform_cpp(const torch::Tensor& counters) {
    TORCH_CHECK(counters.is_cpu(), "philox_uniform_cpp: counters must be a CPU tensor");
    TORCH_CHECK(counters.dim() == 1, "philox_uniform_cpp: counters must be 1D");

    auto c64 = counters.to(torch::kInt64).contiguous();
    const int64_t n = c64.numel();

    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    const int64_t* RESTRICT in_ptr  = c64.data_ptr<int64_t>();
    float*         RESTRICT out_ptr = out.data_ptr<float>();

    // Grain size: each task processes a contiguous block; 512 elements gives
    // reasonable parallelism without excessive scheduling overhead.
    at::parallel_for(0, n, 512, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            out_ptr[i] = philox_to_uniform(in_ptr[i]);
        }
    });

    return out;
}

/**
 * philox_gaussian(counters) -> float32 [n]
 *
 * Generates per-neuron independent Gaussian(0, 1) noise in a single pass
 * using Box-Muller on two Philox draws (counter and counter+1), exactly
 * matching rng.py::philox_gaussian().
 *
 * Fusing the two uniform draws + Box-Muller into one loop avoids the
 * overhead of two separate tensor allocations and two kernel dispatches.
 *
 * Args:
 *   counters: int32 or int64 CPU tensor [n]
 *
 * Returns:
 *   float32 tensor [n], Gaussian(0, 1)
 */
torch::Tensor philox_gaussian_cpp(const torch::Tensor& counters) {
    TORCH_CHECK(counters.is_cpu(), "philox_gaussian_cpp: counters must be a CPU tensor");
    TORCH_CHECK(counters.dim() == 1, "philox_gaussian_cpp: counters must be 1D");

    auto c64 = counters.to(torch::kInt64).contiguous();
    const int64_t n = c64.numel();

    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    const int64_t* RESTRICT in_ptr  = c64.data_ptr<int64_t>();
    float*         RESTRICT out_ptr = out.data_ptr<float>();

    at::parallel_for(0, n, 512, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            float u1 = philox_to_uniform(in_ptr[i]);
            float u2 = philox_to_uniform(in_ptr[i] + 1);
            // Box-Muller: N(0,1) = sqrt(-2 ln u1) * cos(2π u2)
            out_ptr[i] = std::sqrtf(-2.0f * std::logf(u1)) * std::cosf(TWO_PI * u2);
        }
    });

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("philox_uniform_cpp",  &philox_uniform_cpp,  "Philox 2x32-10 uniform  (0,1) — CPU, parallelised");
    m.def("philox_gaussian_cpp", &philox_gaussian_cpp, "Philox 2x32-10 Gaussian (0,1) via Box-Muller — CPU, parallelised");
}
