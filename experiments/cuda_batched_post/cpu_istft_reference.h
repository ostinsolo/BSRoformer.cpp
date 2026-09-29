#pragma once

// Host reference ISTFT using the same stft::TableFFT path as PostProcessAndISTFT.

void cpu_istft_from_planar(const float* istft_planar,
                           int n_freq,
                           int n_frames,
                           int n_fft,
                           int hop_length,
                           int win_length,
                           bool center,
                           int output_length,
                           float* output_audio);

void cpu_tablefft_single_frame(const float* planar_frame_nfreq,
                               int n_freq,
                               int n_fft,
                               float* frame_out);
