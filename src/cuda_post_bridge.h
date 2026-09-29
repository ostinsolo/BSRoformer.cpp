#pragma once

#include <cstddef>
#include <vector>

struct CudaPostConfig;

namespace cuda_post_bridge {

bool available();

void set_enabled(bool enabled);
bool enabled();

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
                  double* post_ms_out);

}  // namespace cuda_post_bridge
