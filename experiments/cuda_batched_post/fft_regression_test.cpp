#include "cpu_istft_reference.h"

#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        m = std::max(m, std::fabs(static_cast<double>(a[i] - b[i])));
    }
    return m;
}

int main() {
    const int n_fft = 2048;
    const int n_freq = n_fft / 2 + 1;
    std::vector<float> planar(static_cast<size_t>(n_freq) * 2, 0.0f);
    planar[2] = 1.0f;
    planar[3] = 0.0f;

    std::vector<float> table_out(static_cast<size_t>(n_fft));
    cpu_tablefft_single_frame(planar.data(), n_freq, n_fft, table_out.data());

    std::vector<cufftComplex> h_in(static_cast<size_t>(n_fft));
    for (int k = 0; k < n_freq; ++k) {
        h_in[static_cast<size_t>(k)] = cufftComplex{planar[static_cast<size_t>(k) * 2],
                                                    planar[static_cast<size_t>(k) * 2 + 1]};
    }
    for (int i = n_freq; i < n_fft; ++i) {
        const cufftComplex src = h_in[static_cast<size_t>(n_fft - i)];
        h_in[static_cast<size_t>(i)] = cufftComplex{src.x, -src.y};
    }

    cufftComplex* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    cudaMalloc(&d_in, sizeof(cufftComplex) * n_fft);
    cudaMalloc(&d_out, sizeof(cufftComplex) * n_fft);
    cudaMemcpy(d_in, h_in.data(), sizeof(cufftComplex) * n_fft, cudaMemcpyHostToDevice);
    cufftHandle plan;
    cufftPlan1d(&plan, n_fft, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_in, d_out, CUFFT_INVERSE);
    std::vector<cufftComplex> h_out(n_fft);
    cudaMemcpy(h_out.data(), d_out, sizeof(cufftComplex) * n_fft, cudaMemcpyDeviceToHost);
    const float inv_n = 1.0f / static_cast<float>(n_fft);
    std::vector<float> cufft_out(static_cast<size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) cufft_out[static_cast<size_t>(i)] = h_out[static_cast<size_t>(i)].x * inv_n;

    const double diff = max_abs_diff(table_out, cufft_out);
    std::cout << "{\"single_bin_frame0_max_abs\":" << diff << "}\n";
    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
    return diff < 1e-3 ? 0 : 1;
}
