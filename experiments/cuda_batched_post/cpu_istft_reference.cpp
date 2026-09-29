#include "cpu_istft_reference.h"

#include "stft.h"

#include <cstring>
#include <vector>

void cpu_istft_from_planar(const float* istft_planar,
                           int n_freq,
                           int n_frames,
                           int n_fft,
                           int hop_length,
                           int win_length,
                           bool center,
                           int output_length,
                           float* output_audio) {
    std::vector<float> hann(static_cast<size_t>(win_length));
    stft::hann_window(hann.data(), win_length, true);
    stft::compute_istft(istft_planar, n_freq, n_frames, n_fft, hop_length, win_length, hann.data(),
                        center, output_length, output_audio);
}

void cpu_tablefft_single_frame(const float* planar_frame_nfreq,
                               int n_freq,
                               int n_fft,
                               float* frame_out) {
    stft::STFTBuffer buffer;
    buffer.Resize(n_fft);
    std::vector<stft::Complex> fft_in(static_cast<size_t>(n_freq));
    for (int k = 0; k < n_freq; ++k) {
        fft_in[static_cast<size_t>(k)] =
            stft::Complex(planar_frame_nfreq[k * 2 + 0], planar_frame_nfreq[k * 2 + 1]);
    }
    stft::irfft(fft_in.data(), frame_out, n_fft, buffer,
                *stft::TableFFT::GetInstance(n_fft));
}
