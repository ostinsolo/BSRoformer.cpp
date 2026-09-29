#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct CudaPostConfig {
    int n_fft = 2048;
    int hop_length = 512;
    int win_length = 2048;
    int n_frames = 0;
    int num_stems = 6;
    int channels = 2;
    int n_freq = 1025;
    int num_freq_indices = 2050;
    bool zero_dc = true;
    int output_length = 0;
};

struct CudaPostStageMs {
    double mask_h2d_ms = 0.0;
    double stft_h2d_ms = 0.0;
    double mask_kernel_ms = 0.0;
    double cufft_ms = 0.0;
    double ola_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

// Mode A: mask starts on host. Mode B: mask already on device (skip mask H2D).
struct CudaPostPipeline {
    explicit CudaPostPipeline(const CudaPostConfig& cfg);
    ~CudaPostPipeline();

    CudaPostPipeline(const CudaPostPipeline&) = delete;
    CudaPostPipeline& operator=(const CudaPostPipeline&) = delete;

    bool run_mode_a(const std::vector<float>& mask_host,
                    const std::vector<std::vector<float>>& stft_host,
                    const std::vector<int>& freq_indices,
                    const std::vector<int>& num_bands_per_freq,
                    std::vector<std::vector<float>>& stems_interleaved_host,
                    CudaPostStageMs& timings);

    bool upload_mask_for_mode_b(const std::vector<float>& mask_host);
    bool run_mode_b(const std::vector<std::vector<float>>& stft_host,
                    const std::vector<int>& freq_indices,
                    const std::vector<int>& num_bands_per_freq,
                    std::vector<std::vector<float>>& stems_interleaved_host,
                    CudaPostStageMs& timings);

private:
    struct Impl;
    Impl* impl_;
    bool run_core(const std::vector<float>* mask_host,
                  const std::vector<std::vector<float>>& stft_host,
                  const std::vector<int>& freq_indices,
                  const std::vector<int>& num_bands_per_freq,
                  std::vector<std::vector<float>>& stems_interleaved_host,
                  CudaPostStageMs& timings);
};

bool cuda_post_available();
