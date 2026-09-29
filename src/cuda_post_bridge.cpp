#include "cuda_post_bridge.h"

#include "../experiments/cuda_batched_post/cuda_post_pipeline.h"

#include <chrono>
#include <cuda_runtime.h>
#include <memory>

namespace cuda_post_bridge {

static bool g_enabled = true;
static std::unique_ptr<CudaPostPipeline> g_pipe;
static CudaPostConfig g_cfg;
static int g_cfg_key = -1;

bool available() { return true; }

void set_enabled(bool enabled) { g_enabled = enabled; }

bool enabled() { return g_enabled; }

static int config_key(int n_frames, int output_length, int num_stems) {
    return n_frames * 1000000 + output_length + num_stems * 7;
}

bool post_process(float* mask_device,
                  size_t mask_bytes,
                  const std::vector<std::vector<float>>& stft_outputs,
                  int n_frames,
                  int n_fft,
                  int hop_length,
                  int win_length,
                  int num_stems,
                  int output_length,
                  bool zero_dc,
                  const std::vector<int>& freq_indices,
                  const std::vector<int>& num_bands_per_freq,
                  std::vector<std::vector<float>>& stems_out,
                  double* post_ms_out) {
    if (!g_enabled || !mask_device) return false;
    const int key = config_key(n_frames, output_length, num_stems);
    if (!g_pipe || g_cfg_key != key) {
        g_cfg = CudaPostConfig{};
        g_cfg.n_fft = n_fft;
        g_cfg.hop_length = hop_length;
        g_cfg.win_length = win_length;
        g_cfg.n_frames = n_frames;
        g_cfg.num_stems = num_stems;
        g_cfg.n_freq = n_fft / 2 + 1;
        g_cfg.num_freq_indices = static_cast<int>(freq_indices.size());
        g_cfg.zero_dc = zero_dc;
        g_cfg.output_length = output_length;
        g_pipe = std::make_unique<CudaPostPipeline>(g_cfg);
        g_cfg_key = key;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> gpu_out;
    CudaPostStageMs timings{};
    if (!g_pipe->run_mode_b_device_mask(mask_device, mask_bytes, stft_outputs, freq_indices,
                                        num_bands_per_freq, gpu_out, timings)) {
        return false;
    }
    stems_out = std::move(gpu_out);
    const auto t1 = std::chrono::steady_clock::now();
    if (post_ms_out) {
        *post_ms_out =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
}

}  // namespace cuda_post_bridge
