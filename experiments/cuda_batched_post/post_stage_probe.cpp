#include "bs_roformer/inference.h"
#include "bs_roformer/audio.h"
#include "cpu_istft_reference.h"
#include "cuda_post_pipeline.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Metrics {
    double pearson = 0.0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rel_l2 = 0.0;
};

Metrics compare(const std::vector<float>& a, const std::vector<float>& b) {
    Metrics m;
    const size_t n = std::min(a.size(), b.size());
    if (n == 0) return m;
    double dot = 0, na = 0, nb = 0, diff2 = 0, ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= static_cast<double>(n);
    mb /= static_cast<double>(n);
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i] - b[i]);
        const double ad = std::fabs(d);
        m.max_abs = std::max(m.max_abs, ad);
        m.mean_abs += ad;
        diff2 += d * d;
        const double da = a[i] - ma;
        const double db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    m.mean_abs /= static_cast<double>(n);
    m.rel_l2 = (na > 1e-30) ? std::sqrt(diff2 / na) : 0.0;
    m.pearson = (std::sqrt(va * vb) > 1e-30) ? (cov / std::sqrt(va * vb)) : 0.0;
    return m;
}

void cpu_post_stages(Inference& engine,
                     const Inference::ChunkForwardArtifacts& art,
                     int stem,
                     int ch,
                     std::vector<float>& accum,
                     std::vector<float>& istft_planar,
                     std::vector<float>& audio) {
    const int n_fft = engine.GetNFFT();
    const int hop = engine.GetHopLength();
    const int win = engine.GetWinLength();
    const int n_freq = n_fft / 2 + 1;
    const int n_frames = art.n_frames;
    const int channels = 2;
    const auto& freq_indices = engine.GetFreqIndices();
    const int num_freq_indices = static_cast<int>(freq_indices.size());
    const int mask_features = num_freq_indices * 2;
    const int stride_time = mask_features * engine.GetNumStems();
    const auto& denom_vec = engine.GetNumBandsPerFreq();

    accum.assign(static_cast<size_t>(n_freq) * n_frames * 2, 0.0f);
    istft_planar.assign(static_cast<size_t>(n_freq) * n_frames * 2, 0.0f);

    for (int fi = 0; fi < num_freq_indices; ++fi) {
        const int freq_stereo_idx = freq_indices[static_cast<size_t>(fi)];
        if (freq_stereo_idx % channels != ch) continue;
        const int raw_f = freq_stereo_idx / channels;
        const int dst_base = raw_f * n_frames * 2;
        for (int t = 0; t < n_frames; ++t) {
            const int mask_idx = t * stride_time + stem * mask_features + fi * 2;
            const int dst_idx = dst_base + t * 2;
            accum[static_cast<size_t>(dst_idx) + 0] += art.mask_output[static_cast<size_t>(mask_idx) + 0];
            accum[static_cast<size_t>(dst_idx) + 1] += art.mask_output[static_cast<size_t>(mask_idx) + 1];
        }
    }

    for (int f = 0; f < n_freq; ++f) {
        float inv_d = 1.0f / std::max(1e-8f, static_cast<float>(denom_vec[static_cast<size_t>(f)]));
        for (int t = 0; t < n_frames; ++t) {
            const int idx = (f * n_frames + t) * 2;
            float mr = accum[static_cast<size_t>(idx) + 0] * inv_d;
            float mi = accum[static_cast<size_t>(idx) + 1] * inv_d;
            const float sr = art.stft_outputs[static_cast<size_t>(ch)][static_cast<size_t>(idx) + 0];
            const float si = art.stft_outputs[static_cast<size_t>(ch)][static_cast<size_t>(idx) + 1];
            istft_planar[static_cast<size_t>(idx) + 0] = sr * mr - si * mi;
            istft_planar[static_cast<size_t>(idx) + 1] = sr * mi + si * mr;
        }
    }
    if (engine.GetZeroDC()) {
        for (int t = 0; t < n_frames; ++t) {
            istft_planar[static_cast<size_t>(t) * 2 + 0] = 0.0f;
            istft_planar[static_cast<size_t>(t) * 2 + 1] = 0.0f;
        }
    }

    audio.assign(static_cast<size_t>(81920), 0.0f);
    cpu_istft_from_planar(istft_planar.data(), n_freq, n_frames, n_fft, hop, win, true,
                          static_cast<int>(audio.size()), audio.data());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: post_stage_probe <model.gguf> <wav> [out.json]\n";
        return 1;
    }
    try {
        const int samples = 81920;
        Inference engine(argv[1]);
        AudioBuffer audio = AudioFile::Load(argv[2]);
        if (audio.channels == 1) {
            std::vector<float> stereo(audio.samples * 2);
            for (size_t i = 0; i < audio.samples; ++i) {
                stereo[i * 2] = audio.data[i];
                stereo[i * 2 + 1] = audio.data[i];
            }
            audio.data = std::move(stereo);
            audio.channels = 2;
            audio.samples *= 2;
        }
        std::vector<float> window(static_cast<size_t>(samples) * 2, 0.0f);
        const int frames = static_cast<int>(audio.samples / 2);
        for (int i = 0; i < samples && i < frames; ++i) {
            window[static_cast<size_t>(i) * 2] = audio.data[static_cast<size_t>(i) * 2];
            window[static_cast<size_t>(i) * 2 + 1] = audio.data[static_cast<size_t>(i) * 2 + 1];
        }
        const auto art = engine.CaptureChunkForward(window);
        const int stem = 0;
        const int ch = 0;
        std::vector<float> accum, planar, cpu_audio;
        cpu_post_stages(engine, art, stem, ch, accum, planar, cpu_audio);

        std::vector<float> cufft_frame(static_cast<size_t>(engine.GetNFFT()));
        std::vector<float> table_frame(static_cast<size_t>(engine.GetNFFT()));
        const int n_freq = engine.GetNFFT() / 2 + 1;
        std::vector<float> frame_spec(static_cast<size_t>(n_freq) * 2);
        for (int k = 0; k < n_freq; ++k) {
            frame_spec[static_cast<size_t>(k) * 2 + 0] = planar[static_cast<size_t>(k) * 2 + 0];
            frame_spec[static_cast<size_t>(k) * 2 + 1] = planar[static_cast<size_t>(k) * 2 + 1];
        }
        cpu_tablefft_single_frame(frame_spec.data(), n_freq, engine.GetNFFT(), table_frame.data());

        std::ostringstream json;
        json << "{\n  \"samples\": " << samples << ",\n  \"stem\": " << stem << ",\n  \"channel\": "
             << ch << ",\n  \"stages\": {\n";
        json << "    \"planar_elems\": " << planar.size() << "\n  },\n";
        json << "  \"note\": \"Run with cuda post after Hann fix; table single-frame frame0 in table_frame\"\n}\n";
        if (argc >= 4) {
            std::ofstream f(argv[3]);
            f << json.str();
        }
        std::cout << json.str();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 2;
    }
    return 0;
}
