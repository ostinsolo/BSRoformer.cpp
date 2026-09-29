#include "bs_roformer/inference.h"
#include "bs_roformer/audio.h"
#include "cuda_post_pipeline.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct StemMetrics {
    double pearson = 0.0;
    double si_sdr_db = 0.0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double relative_l2 = 0.0;
};

StemMetrics score_stem(const std::vector<float>& ref, const std::vector<float>& est) {
    StemMetrics m;
    const size_t n = std::min(ref.size(), est.size());
    if (n == 0) return m;
    double dot = 0.0, ref_norm = 0.0, est_norm = 0.0, diff_norm = 0.0, sum_abs = 0.0;
    double ref_mean = 0.0, est_mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ref_mean += ref[i];
        est_mean += est[i];
    }
    ref_mean /= static_cast<double>(n);
    est_mean /= static_cast<double>(n);
    double cov = 0.0, ref_var = 0.0, est_var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(est[i] - ref[i]);
        const double ad = std::fabs(d);
        if (ad > m.max_abs) m.max_abs = ad;
        sum_abs += ad;
        diff_norm += d * d;
        const double rr = static_cast<double>(ref[i]) - ref_mean;
        const double ee = static_cast<double>(est[i]) - est_mean;
        cov += rr * ee;
        ref_var += rr * rr;
        est_var += ee * ee;
        dot += static_cast<double>(ref[i]) * static_cast<double>(est[i]);
        ref_norm += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        est_norm += static_cast<double>(est[i]) * static_cast<double>(est[i]);
    }
    m.mean_abs = sum_abs / static_cast<double>(n);
    m.relative_l2 = (ref_norm > 1e-30) ? std::sqrt(diff_norm / ref_norm) : 0.0;
    const double denom = std::sqrt(ref_var * est_var);
    m.pearson = (denom > 1e-30) ? (cov / denom) : 0.0;
    const double ref_energy = ref_norm;
    const double noise = diff_norm;
    m.si_sdr_db = 10.0 * std::log10((ref_energy + 1e-30) / (noise + 1e-30));
    return m;
}

double cpu_post_ms(Inference& engine, const Inference::ChunkForwardArtifacts& art,
                   std::vector<std::vector<float>>& cpu_out) {
    const auto t0 = std::chrono::steady_clock::now();
    engine.PostProcessCpu(art.mask_output, art.stft_outputs, art.n_frames, cpu_out);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: cuda_batched_post_bench <model.gguf> <input.wav> <samples> [json_out]\n";
        return 1;
    }
    try {
        const int samples = std::stoi(argv[3]);
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
        const int frames = static_cast<int>(audio.samples / 2);
        std::vector<float> window(static_cast<size_t>(samples) * 2, 0.0f);
        for (int i = 0; i < samples && i < frames; ++i) {
            window[static_cast<size_t>(i) * 2] = audio.data[static_cast<size_t>(i) * 2];
            window[static_cast<size_t>(i) * 2 + 1] = audio.data[static_cast<size_t>(i) * 2 + 1];
        }

        const auto art = engine.CaptureChunkForward(window);
        std::vector<std::vector<float>> cpu_out;
        const double cpu_ms = cpu_post_ms(engine, art, cpu_out);

        CudaPostConfig cfg;
        cfg.n_fft = engine.GetNFFT();
        cfg.hop_length = engine.GetHopLength();
        cfg.win_length = engine.GetWinLength();
        cfg.n_frames = art.n_frames;
        cfg.num_stems = engine.GetNumStems();
        cfg.n_freq = cfg.n_fft / 2 + 1;
        cfg.num_freq_indices = static_cast<int>(engine.GetFreqIndices().size());
        cfg.zero_dc = engine.GetZeroDC();
        cfg.output_length = samples;

        CudaPostPipeline pipe(cfg);
        std::vector<std::vector<float>> gpu_a;
        CudaPostStageMs ta{};
        pipe.run_mode_a(art.mask_output, art.stft_outputs, engine.GetFreqIndices(),
                        engine.GetNumBandsPerFreq(), gpu_a, ta);

        pipe.upload_mask_for_mode_b(art.mask_output);
        std::vector<std::vector<float>> gpu_b;
        CudaPostStageMs tb{};
        pipe.run_mode_b(art.stft_outputs, engine.GetFreqIndices(), engine.GetNumBandsPerFreq(),
                        gpu_b, tb);

        std::ostringstream json;
        json << "{\n"
             << "  \"samples\": " << samples << ",\n"
             << "  \"n_frames\": " << art.n_frames << ",\n"
             << "  \"cpu_post_ms\": " << cpu_ms << ",\n"
             << "  \"mode_a\": {"
             << "\"mask_h2d_ms\":" << ta.mask_h2d_ms << ","
             << "\"mask_kernel_ms\":" << ta.mask_kernel_ms << ","
             << "\"cufft_ms\":" << ta.cufft_ms << ","
             << "\"ola_ms\":" << ta.ola_ms << ","
             << "\"total_ms\":" << ta.total_ms << "},\n"
             << "  \"mode_b\": {"
             << "\"mask_h2d_ms\":" << tb.mask_h2d_ms << ","
             << "\"mask_kernel_ms\":" << tb.mask_kernel_ms << ","
             << "\"cufft_ms\":" << tb.cufft_ms << ","
             << "\"ola_ms\":" << tb.ola_ms << ","
             << "\"total_ms\":" << tb.total_ms << "},\n"
             << "  \"stems\": [\n";
        for (int s = 0; s < engine.GetNumStems(); ++s) {
            if (s) json << ",\n";
            const auto m = score_stem(cpu_out[static_cast<size_t>(s)], gpu_a[static_cast<size_t>(s)]);
            json << "    {\"stem\":" << s << ",\"pearson\":" << m.pearson << ",\"si_sdr_db\":"
                 << m.si_sdr_db << ",\"max_abs\":" << m.max_abs << ",\"mean_abs\":" << m.mean_abs
                 << ",\"relative_l2\":" << m.relative_l2 << "}";
        }
        json << "\n  ]\n}\n";
        const std::string out = json.str();
        if (argc >= 5) {
            std::ofstream f(argv[4]);
            f << out;
        }
        std::cout << out;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 2;
    }
    return 0;
}
