#include "cuda_post_pipeline.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
        }                                                                                \
    } while (0)

#define CUFFT_CHECK(call)                                                                \
    do {                                                                                 \
        cufftResult err__ = (call);                                                      \
        if (err__ != CUFFT_SUCCESS) {                                                    \
            throw std::runtime_error("cuFFT error");                                     \
        }                                                                                \
    } while (0)

__global__ void scatter_mask_kernel(const float* __restrict__ mask,
                                    const int* __restrict__ freq_indices,
                                    int num_freq_indices,
                                    int mask_features,
                                    int stride_time,
                                    int stem,
                                    int channel,
                                    int channels,
                                    int n_frames,
                                    int n_freq,
                                    float* __restrict__ accum_planar) {
    const int fi = blockIdx.x * blockDim.x + threadIdx.x;
    if (fi >= num_freq_indices) return;
    const int freq_stereo_idx = freq_indices[fi];
    if (freq_stereo_idx % channels != channel) return;
    const int raw_f = freq_stereo_idx / channels;
    const int dst_base = raw_f * n_frames * 2;
    for (int t = 0; t < n_frames; ++t) {
        const int mask_idx = t * stride_time + stem * mask_features + fi * 2;
        const int dst_idx = dst_base + t * 2;
        accum_planar[dst_idx + 0] += mask[mask_idx + 0];
        accum_planar[dst_idx + 1] += mask[mask_idx + 1];
    }
}

__global__ void apply_norm_and_multiply_kernel(const float* __restrict__ accum_planar,
                                               const float* __restrict__ stft_ch,
                                               const float* __restrict__ denom,
                                               int n_frames,
                                               int n_freq,
                                               bool zero_dc,
                                               float* __restrict__ istft_planar) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_freq) return;
    const float inv_d = denom[f];
    for (int t = 0; t < n_frames; ++t) {
        const int idx = (f * n_frames + t) * 2;
        float mr = accum_planar[idx + 0] * inv_d;
        float mi = accum_planar[idx + 1] * inv_d;
        const float sr = stft_ch[idx + 0];
        const float si = stft_ch[idx + 1];
        istft_planar[idx + 0] = sr * mr - si * mi;
        istft_planar[idx + 1] = sr * mi + si * mr;
    }
    if (zero_dc && f == 0) {
        for (int t = 0; t < n_frames; ++t) {
            const int idx = t * 2;
            istft_planar[idx + 0] = 0.0f;
            istft_planar[idx + 1] = 0.0f;
        }
    }
}

__global__ void pack_ifft_input_kernel(const float* __restrict__ planar,
                                       cufftComplex* __restrict__ batch_in,
                                       int n_frames,
                                       int n_freq,
                                       int n_fft) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_frames) return;
    cufftComplex* row = batch_in + f * n_fft;
    for (int k = 0; k < n_fft; ++k) row[k] = cufftComplex{0.0f, 0.0f};
    for (int k = 0; k < n_freq; ++k) {
        const int idx = (k * n_frames + f) * 2;
        row[k] = cufftComplex{planar[idx + 0], planar[idx + 1]};
    }
    for (int k = n_freq; k < n_fft; ++k) {
        const int mirror = n_fft - k;
        if (mirror >= 0 && mirror < n_freq) {
            const int idx = (mirror * n_frames + f) * 2;
            row[k] = cufftComplex{planar[idx + 0], -planar[idx + 1]};
        }
    }
}

__global__ void extract_real_frames_kernel(const cufftComplex* __restrict__ fft_out,
                                           float* __restrict__ frames,
                                           int n_frames,
                                           int n_fft,
                                           float inv_n) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_frames) return;
    for (int i = 0; i < n_fft; ++i) {
        frames[f * n_fft + i] = fft_out[f * n_fft + i].x * inv_n;
    }
}

__global__ void ola_normalize_kernel(const float* __restrict__ frames,
                                     const float* __restrict__ window,
                                     int n_frames,
                                     int n_fft,
                                     int hop,
                                     int pad_amount,
                                     int output_len,
                                     float* __restrict__ out_audio) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int expected_len = n_fft + hop * (n_frames - 1);
    if (i >= output_len) return;

    const int y_idx = pad_amount + i;
    float sum = 0.0f;
    float wsum = 0.0f;
    if (y_idx >= 0 && y_idx < expected_len) {
        for (int f = 0; f < n_frames; ++f) {
            const int start = f * hop;
            const int j = y_idx - start;
            if (j < 0 || j >= n_fft) continue;
            const float frame = frames[f * n_fft + j];
            const float w = window[j];
            sum += frame * w;
            wsum += w * w;
        }
    }
    out_audio[i] = (wsum > 1e-8f) ? (sum / wsum) : 0.0f;
}

double elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return static_cast<double>(ms);
}

}  // namespace

struct CudaPostPipeline::Impl {
    CudaPostConfig cfg;
    cufftHandle plan = 0;
    int plan_frames = -1;

    float* d_mask = nullptr;
    size_t mask_bytes = 0;
    float* d_stft[2] = {nullptr, nullptr};
    float* d_window = nullptr;
    int* d_freq_indices = nullptr;
    float* d_denom = nullptr;
    float* d_planar = nullptr;
    float* d_accum = nullptr;
    cufftComplex* d_fft_in = nullptr;
    cufftComplex* d_fft_out = nullptr;
    float* d_frames = nullptr;
    float* d_audio = nullptr;

    void ensure_plan(int n_frames) {
        if (plan && plan_frames == n_frames) return;
        if (plan) cufftDestroy(plan);
        CUFFT_CHECK(cufftPlan1d(&plan, cfg.n_fft, CUFFT_C2C, n_frames));
        plan_frames = n_frames;
    }

    void ensure_buffers(size_t mask_bytes_in, size_t stft_bytes, size_t planar_bytes) {
        if (mask_bytes_in > mask_bytes) {
            if (d_mask) CUDA_CHECK(cudaFree(d_mask));
            CUDA_CHECK(cudaMalloc(&d_mask, mask_bytes_in));
            mask_bytes = mask_bytes_in;
        }
        for (int ch = 0; ch < 2; ++ch) {
            if (!d_stft[ch]) CUDA_CHECK(cudaMalloc(&d_stft[ch], stft_bytes));
        }
        if (!d_planar) CUDA_CHECK(cudaMalloc(&d_planar, planar_bytes));
        if (!d_accum) CUDA_CHECK(cudaMalloc(&d_accum, planar_bytes));
        if (!d_fft_in) CUDA_CHECK(cudaMalloc(&d_fft_in, sizeof(cufftComplex) * cfg.n_fft * cfg.n_frames));
        if (!d_fft_out) CUDA_CHECK(cudaMalloc(&d_fft_out, sizeof(cufftComplex) * cfg.n_fft * cfg.n_frames));
        if (!d_frames) CUDA_CHECK(cudaMalloc(&d_frames, sizeof(float) * cfg.n_fft * cfg.n_frames));
        if (!d_audio) CUDA_CHECK(cudaMalloc(&d_audio, sizeof(float) * cfg.output_length));
        if (!d_window) {
            CUDA_CHECK(cudaMalloc(&d_window, sizeof(float) * cfg.n_fft));
            std::vector<float> win(cfg.n_fft, 0.0f);
            const int left = (cfg.n_fft - cfg.win_length) / 2;
            for (int i = 0; i < cfg.win_length; ++i) {
                const double pi = std::acos(-1.0);
                const double n = static_cast<double>(1 - cfg.win_length + 2 * i);
                const double denom = static_cast<double>(cfg.win_length - 1);
                win[static_cast<size_t>(left + i)] =
                    static_cast<float>(0.5 * (1.0 - std::cos(2.0 * pi * n / denom)));
            }
            CUDA_CHECK(cudaMemcpy(d_window, win.data(), sizeof(float) * cfg.n_fft, cudaMemcpyHostToDevice));
        }
    }
};

bool cuda_post_available() { return true; }

CudaPostPipeline::CudaPostPipeline(const CudaPostConfig& cfg) : impl_(new Impl()) {
    impl_->cfg = cfg;
}

CudaPostPipeline::~CudaPostPipeline() {
    if (!impl_) return;
    if (impl_->plan) cufftDestroy(impl_->plan);
    if (impl_->d_mask) cudaFree(impl_->d_mask);
    for (int ch = 0; ch < 2; ++ch) {
        if (impl_->d_stft[ch]) cudaFree(impl_->d_stft[ch]);
    }
    if (impl_->d_window) cudaFree(impl_->d_window);
    if (impl_->d_freq_indices) cudaFree(impl_->d_freq_indices);
    if (impl_->d_denom) cudaFree(impl_->d_denom);
    if (impl_->d_planar) cudaFree(impl_->d_planar);
    if (impl_->d_accum) cudaFree(impl_->d_accum);
    if (impl_->d_fft_in) cudaFree(impl_->d_fft_in);
    if (impl_->d_fft_out) cudaFree(impl_->d_fft_out);
    if (impl_->d_frames) cudaFree(impl_->d_frames);
    if (impl_->d_audio) cudaFree(impl_->d_audio);
    delete impl_;
    impl_ = nullptr;
}

bool CudaPostPipeline::upload_mask_for_mode_b(const std::vector<float>& mask_host) {
    const size_t bytes = mask_host.size() * sizeof(float);
    impl_->ensure_buffers(bytes, 0, 0);
    CUDA_CHECK(cudaMemcpy(impl_->d_mask, mask_host.data(), bytes, cudaMemcpyHostToDevice));
    return true;
}

bool CudaPostPipeline::run_core(const std::vector<float>* mask_host,
                                const std::vector<std::vector<float>>& stft_host,
                                const std::vector<int>& freq_indices,
                                const std::vector<int>& num_bands_per_freq,
                                std::vector<std::vector<float>>& stems_interleaved_host,
                                CudaPostStageMs& timings) {
    Impl* impl = impl_;
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));
    CUDA_CHECK(cudaEventCreate(&e4));
    CUDA_CHECK(cudaEventCreate(&e5));

    const auto& cfg = impl->cfg;
    const size_t stft_bytes = static_cast<size_t>(cfg.n_freq) * cfg.n_frames * 2 * sizeof(float);
    const size_t planar_bytes = stft_bytes;
    const size_t mask_bytes =
        mask_host ? mask_host->size() * sizeof(float) : impl->mask_bytes;
    impl->ensure_buffers(mask_bytes, stft_bytes, planar_bytes);
    impl->ensure_plan(cfg.n_frames);

    if (!impl->d_freq_indices) {
        CUDA_CHECK(cudaMalloc(&impl->d_freq_indices, freq_indices.size() * sizeof(int)));
    }
    if (!impl->d_denom) {
        CUDA_CHECK(cudaMalloc(&impl->d_denom, cfg.n_freq * sizeof(float)));
    }

    CUDA_CHECK(cudaEventRecord(e0));
    if (mask_host) {
        CUDA_CHECK(cudaMemcpy(impl->d_mask, mask_host->data(), mask_host->size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(impl->d_stft[0], stft_host[0].data(), stft_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->d_stft[1], stft_host[1].data(), stft_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->d_freq_indices, freq_indices.data(),
                          freq_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<float> denom(cfg.n_freq);
    for (int f = 0; f < cfg.n_freq; ++f) {
        denom[static_cast<size_t>(f)] = 1.0f / std::max(1e-8f, static_cast<float>(num_bands_per_freq[f]));
    }
    CUDA_CHECK(cudaMemcpy(impl->d_denom, denom.data(), cfg.n_freq * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    const int mask_features = cfg.num_freq_indices * 2;
    const int stride_time = mask_features * cfg.num_stems;
    stems_interleaved_host.assign(cfg.num_stems, std::vector<float>());

    double mask_ms = 0.0;
    double cufft_ms = 0.0;
    double ola_ms = 0.0;

    for (int stem = 0; stem < cfg.num_stems; ++stem) {
        std::vector<float> ch0(cfg.output_length);
        std::vector<float> ch1(cfg.output_length);
        for (int ch = 0; ch < cfg.channels; ++ch) {
            cudaEvent_t km0, km1, cf0, cf1, ol0, ol1;
            CUDA_CHECK(cudaEventCreate(&km0));
            CUDA_CHECK(cudaEventCreate(&km1));
            CUDA_CHECK(cudaEventCreate(&cf0));
            CUDA_CHECK(cudaEventCreate(&cf1));
            CUDA_CHECK(cudaEventCreate(&ol0));
            CUDA_CHECK(cudaEventCreate(&ol1));

            CUDA_CHECK(cudaEventRecord(km0));
            CUDA_CHECK(cudaMemset(impl->d_accum, 0, planar_bytes));
            scatter_mask_kernel<<<(cfg.num_freq_indices + 255) / 256, 256>>>(
                impl->d_mask, impl->d_freq_indices, cfg.num_freq_indices, mask_features,
                stride_time, stem, ch, cfg.channels, cfg.n_frames, cfg.n_freq, impl->d_accum);
            apply_norm_and_multiply_kernel<<<(cfg.n_freq + 255) / 256, 256>>>(
                impl->d_accum, impl->d_stft[ch], impl->d_denom, cfg.n_frames, cfg.n_freq,
                cfg.zero_dc, impl->d_planar);
            CUDA_CHECK(cudaEventRecord(km1));

            CUDA_CHECK(cudaEventRecord(cf0));
            pack_ifft_input_kernel<<<(cfg.n_frames + 255) / 256, 256>>>(
                impl->d_planar, impl->d_fft_in, cfg.n_frames, cfg.n_freq, cfg.n_fft);
            CUFFT_CHECK(cufftExecC2C(impl->plan, impl->d_fft_in, impl->d_fft_out, CUFFT_INVERSE));
            const float inv_n = 1.0f / static_cast<float>(cfg.n_fft);
            extract_real_frames_kernel<<<(cfg.n_frames + 255) / 256, 256>>>(
                impl->d_fft_out, impl->d_frames, cfg.n_frames, cfg.n_fft, inv_n);
            CUDA_CHECK(cudaEventRecord(cf1));

            CUDA_CHECK(cudaEventRecord(ol0));
            const int pad = cfg.n_fft / 2;
            ola_normalize_kernel<<<(cfg.output_length + 255) / 256, 256>>>(
                impl->d_frames, impl->d_window, cfg.n_frames, cfg.n_fft, cfg.hop_length, pad,
                cfg.output_length, impl->d_audio);
            CUDA_CHECK(cudaEventRecord(ol1));

            std::vector<float> h_audio(static_cast<size_t>(cfg.output_length));
            CUDA_CHECK(cudaMemcpy(h_audio.data(), impl->d_audio, h_audio.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            if (ch == 0) ch0 = std::move(h_audio);
            else ch1 = std::move(h_audio);

            mask_ms += elapsed_ms(km0, km1);
            cufft_ms += elapsed_ms(cf0, cf1);
            ola_ms += elapsed_ms(ol0, ol1);

            CUDA_CHECK(cudaEventDestroy(km0));
            CUDA_CHECK(cudaEventDestroy(km1));
            CUDA_CHECK(cudaEventDestroy(cf0));
            CUDA_CHECK(cudaEventDestroy(cf1));
            CUDA_CHECK(cudaEventDestroy(ol0));
            CUDA_CHECK(cudaEventDestroy(ol1));
        }
        std::vector<float> interleaved(static_cast<size_t>(cfg.output_length) * 2);
        for (int i = 0; i < cfg.output_length; ++i) {
            interleaved[static_cast<size_t>(i) * 2 + 0] = ch0[static_cast<size_t>(i)];
            interleaved[static_cast<size_t>(i) * 2 + 1] = ch1[static_cast<size_t>(i)];
        }
        stems_interleaved_host[static_cast<size_t>(stem)] = std::move(interleaved);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e5));
    timings.mask_h2d_ms = elapsed_ms(e0, e1);
    timings.mask_kernel_ms = mask_ms;
    timings.cufft_ms = cufft_ms;
    timings.ola_ms = ola_ms;
    timings.total_ms = elapsed_ms(e0, e5);

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaEventDestroy(e3));
    CUDA_CHECK(cudaEventDestroy(e4));
    CUDA_CHECK(cudaEventDestroy(e5));
    return true;
}

bool CudaPostPipeline::run_mode_a(const std::vector<float>& mask_host,
                                  const std::vector<std::vector<float>>& stft_host,
                                  const std::vector<int>& freq_indices,
                                  const std::vector<int>& num_bands_per_freq,
                                  std::vector<std::vector<float>>& stems_interleaved_host,
                                  CudaPostStageMs& timings) {
    return run_core(&mask_host, stft_host, freq_indices, num_bands_per_freq, stems_interleaved_host,
                    timings);
}

bool CudaPostPipeline::run_mode_b(const std::vector<std::vector<float>>& stft_host,
                                  const std::vector<int>& freq_indices,
                                  const std::vector<int>& num_bands_per_freq,
                                  std::vector<std::vector<float>>& stems_interleaved_host,
                                  CudaPostStageMs& timings) {
    return run_core(nullptr, stft_host, freq_indices, num_bands_per_freq, stems_interleaved_host,
                    timings);
}
