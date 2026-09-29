#include "bs_roformer/inference.h"
#include "bs_roformer/audio.h"
#include "roformer_mlx/rolling_ola.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kBenchWindows[] = {81920, 102400, 153600, 204800, 307200, 409600};
constexpr int kLiveHopSamples = 12288;
constexpr int kTeacherChunk = 409600;
constexpr int kTeacherStep = 352800;
constexpr const char* kStemNames[] = {
    "bass", "drums", "other", "vocals", "guitar", "piano"};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <model.gguf> <input.wav> <output.wav> [options]\n";
    std::cerr << "       " << program_name << " <model.gguf> --bench <input.wav> [--repeats N]\n";
    std::cerr << "       " << program_name << " <model.gguf> --process-bench <input.wav>\n";
    std::cerr << "       " << program_name << " <model.gguf> --separate <input.wav> <out_dir>\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --chunk-size <N>   Chunk size in samples (default: from model, fallback 352800)\n";
    std::cerr << "  --overlap <N>      Number of overlaps for crossfade (default: from model, fallback 2)\n";
    std::cerr << "  --bench <wav>      Warm stage-timing ladder (81920 .. 409600) on the WAV\n";
    std::cerr << "  --process-bench <wav>\n";
    std::cerr << "                     Wall-clock Process() / ProcessOverlapAddPipelined (JSON stdout)\n";
    std::cerr << "  --process-repeats <N>  Timed full-file repeats after one warmup (default 1)\n";
    std::cerr << "  --live-stress <wav> 81920 jitter JSON (use --stress-iters)\n";
    std::cerr << "  --stress-iters <N> Live stress iterations (default 1000)\n";
    std::cerr << "  --repeats <N>      Timed repeats per window after one warmup (default 3)\n";
    std::cerr << "  --separate <wav> <dir>\n";
    std::cerr << "                     Offline Hamming OLA, one WAV per stem\n";
    std::cerr << "  --chunk-samples <N> Separate window length (default 409600)\n";
    std::cerr << "  --step <N>         Separate hop (default 352800)\n";
    std::cerr << "  --help, -h         Show this help message\n";
}

double medianOf(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::vector<float> numpyHamming(int length) {
    std::vector<float> values(static_cast<size_t>(length), 1.0f);
    if (length <= 1) return values;
    const double pi = std::acos(-1.0);
    const double denominator = static_cast<double>(length - 1);
    for (int index = 0; index < length; ++index) {
        const double n = static_cast<double>(1 - length + 2 * index);
        values[static_cast<size_t>(index)] = static_cast<float>(
            0.54 + 0.46 * std::cos(pi * n / denominator));
    }
    return values;
}

std::vector<int> chunkStarts(int total_samples, int chunk_size, int step) {
    const int max_start = std::max(total_samples - chunk_size, 0);
    std::vector<int> starts;
    for (int start = 0; start <= max_start; start += step) {
        starts.push_back(start);
    }
    if (starts.empty()) {
        starts.push_back(0);
    } else if (starts.back() != max_start) {
        starts.push_back(max_start);
    }
    return starts;
}

AudioBuffer loadStereo44100(Inference& engine, const std::string& input_path) {
    std::cerr << "Loading audio: " << input_path << std::endl;
    AudioBuffer input_audio = AudioFile::Load(input_path);
    std::cerr << "Audio loaded: " << input_audio.samples << " samples, "
              << input_audio.channels << " channels, "
              << input_audio.sampleRate << " Hz" << std::endl;

    const int required_sr = engine.GetSampleRate();
    if (static_cast<int>(input_audio.sampleRate) != required_sr) {
        throw std::runtime_error("Input audio sample rate must be " + std::to_string(required_sr) +
                                 " Hz. Current: " + std::to_string(input_audio.sampleRate));
    }
    if (input_audio.channels == 1) {
        std::vector<float> stereo_data(input_audio.samples * 2);
        for (size_t i = 0; i < input_audio.samples; ++i) {
            stereo_data[i * 2 + 0] = input_audio.data[i];
            stereo_data[i * 2 + 1] = input_audio.data[i];
        }
        input_audio.data = std::move(stereo_data);
        input_audio.channels = 2;
        input_audio.samples *= 2;
    } else if (input_audio.channels != 2) {
        throw std::runtime_error("Input audio must be Stereo (2 channels) or Mono (1 channel). Current: " +
                                 std::to_string(input_audio.channels));
    }
    return input_audio;
}

int frameCount(const AudioBuffer& audio) {
    return static_cast<int>(audio.samples / audio.channels);
}

std::vector<float> interleavedWindow(const AudioBuffer& audio, int start, int frames) {
    std::vector<float> window(static_cast<size_t>(frames) * 2, 0.0f);
    const int total = frameCount(audio);
    const int n = std::min(frames, total - start);
    for (int i = 0; i < n; ++i) {
        const size_t src = static_cast<size_t>(start + i) * 2;
        window[static_cast<size_t>(i) * 2] = audio.data[src];
        window[static_cast<size_t>(i) * 2 + 1] = audio.data[src + 1];
    }
    return window;
}

bool backendIsCuda(const std::string& name) {
    return name.find("CUDA") != std::string::npos;
}

bool backendOkForBenchmark(const std::string& name) {
    if (backendIsCuda(name)) return true;
    return name == "CPU" || name.rfind("CPU", 0) == 0;
}

void appendJsonString(std::ostringstream& out, const std::string& value) {
    out << '"';
    for (char c : value) {
        if (c == '\\' || c == '"') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    out << '"';
}

void writeFloatBinary(const std::filesystem::path& path, const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
}

int runProbe(Inference& engine, const AudioBuffer& audio, int samples, const std::string& out_dir) {
    const int frames = frameCount(audio);
    if (frames < samples) {
        throw std::runtime_error("WAV too short for probe");
    }
    const auto window = interleavedWindow(audio, 0, samples);
    const auto probe = engine.ProbeChunk(window);
    std::filesystem::create_directories(out_dir);
    writeFloatBinary(std::filesystem::path(out_dir) / "model_input.f32", probe.stft_flattened);
    writeFloatBinary(std::filesystem::path(out_dir) / "mask_output.f32", probe.mask_output);
    std::ofstream meta(std::filesystem::path(out_dir) / "meta.json");
    meta << "{\n"
         << "  \"samples\": " << samples << ",\n"
         << "  \"n_frames\": " << probe.n_frames << ",\n"
         << "  \"num_freq_indices\": " << probe.num_freq_indices << ",\n"
         << "  \"model_input_elems\": " << probe.stft_flattened.size() << ",\n"
         << "  \"mask_output_elems\": " << probe.mask_output.size() << "\n"
         << "}\n";
    std::cerr << "Probe written to " << out_dir << std::endl;
    return 0;
}

int runBench(Inference& engine, const AudioBuffer& audio, int repeats, int only_samples) {
    const std::string backend = engine.GetBackendName();
    std::cerr << "Backend: " << backend << std::endl;
    if (!backendOkForBenchmark(backend)) {
        std::cerr << "Error: unsupported backend for --bench, got " << backend << std::endl;
        return 2;
    }
    if (repeats < 1) repeats = 1;
    const int frames = frameCount(audio);
    const int sample_rate = engine.GetSampleRate();

    std::ostringstream json;
    json << "{\n  \"backend\": \"" << backend << "\",\n";
    json << "  \"sample_rate\": " << sample_rate << ",\n";
    json << "  \"repeats\": " << repeats << ",\n";
    json << "  \"rows\": [\n";

    bool first = true;
    const int windows[] = {
        only_samples > 0 ? only_samples : kBenchWindows[0],
        only_samples > 0 ? 0 : kBenchWindows[1],
        only_samples > 0 ? 0 : kBenchWindows[2],
        only_samples > 0 ? 0 : kBenchWindows[3],
        only_samples > 0 ? 0 : kBenchWindows[4],
        only_samples > 0 ? 0 : kBenchWindows[5],
    };
    for (int samples : windows) {
        if (samples <= 0) continue;
        if (frames < samples) {
            throw std::runtime_error("WAV has " + std::to_string(frames) +
                                     " frames, need " + std::to_string(samples));
        }
        std::cerr << "Bench " << samples << " samples" << std::endl;
        const auto window = interleavedWindow(audio, 0, samples);
        Inference::ChunkStageMs warm;
        const auto warm_stems = engine.ProcessChunk(window, warm);
        double warm_energy = 0.0;
        for (const auto& stem : warm_stems) {
            for (float s : stem) warm_energy += std::fabs(s);
        }
        const double warm_e2e = warm.stft_ms + warm.pack_ms + warm.forward_ms + warm.post_ms;

        std::vector<double> e2e, stft, pack, forward, h2d, graph, d2h, post;
        e2e.reserve(static_cast<size_t>(repeats));
        for (int r = 0; r < repeats; ++r) {
            Inference::ChunkStageMs stage;
            engine.ProcessChunk(window, stage);
            stft.push_back(stage.stft_ms);
            pack.push_back(stage.pack_ms);
            forward.push_back(stage.forward_ms);
            h2d.push_back(stage.h2d_ms);
            graph.push_back(stage.graph_ms);
            d2h.push_back(stage.d2h_ms);
            post.push_back(stage.post_ms);
            e2e.push_back(stage.stft_ms + stage.pack_ms + stage.forward_ms + stage.post_ms);
        }
        const double med_e2e = medianOf(e2e);
        const double med_fwd = medianOf(forward);
        const double audio_sec = static_cast<double>(samples) / static_cast<double>(sample_rate);
        const double rtf = (med_e2e / 1000.0) / audio_sec;
        std::cerr << "  warm_e2e_ms=" << warm_e2e
                  << " median_forward_ms=" << med_fwd
                  << " median_e2e_ms=" << med_e2e
                  << " rtf=" << rtf
                  << " energy=" << warm_energy << std::endl;

        if (!first) json << ",\n";
        first = false;
        json << "    {\"samples\": " << samples
             << ", \"warmup_e2e_ms\": " << warm_e2e
             << ", \"warmup_forward_ms\": " << warm.forward_ms
             << ", \"warmup_stft_ms\": " << warm.stft_ms
             << ", \"warmup_pack_ms\": " << warm.pack_ms
             << ", \"warmup_h2d_ms\": " << warm.h2d_ms
             << ", \"warmup_graph_ms\": " << warm.graph_ms
             << ", \"warmup_d2h_ms\": " << warm.d2h_ms
             << ", \"warmup_post_ms\": " << warm.post_ms
             << ", \"median_e2e_ms\": " << med_e2e
             << ", \"median_forward_ms\": " << med_fwd
             << ", \"median_stft_ms\": " << medianOf(stft)
             << ", \"median_pack_ms\": " << medianOf(pack)
             << ", \"median_h2d_ms\": " << medianOf(h2d)
             << ", \"median_graph_ms\": " << medianOf(graph)
             << ", \"median_d2h_ms\": " << medianOf(d2h)
             << ", \"median_post_ms\": " << medianOf(post)
             << ", \"rtf\": " << rtf
             << ", \"warmup_abs_energy\": " << warm_energy;
        if (samples == 81920) {
            const double deadline_ms =
                (static_cast<double>(kLiveHopSamples) / static_cast<double>(sample_rate)) * 1000.0;
            json << ", \"live_hop_samples\": " << kLiveHopSamples
                 << ", \"deadline_ms\": " << deadline_ms
                 << ", \"deadline_utilisation\": " << (med_e2e / deadline_ms)
                 << ", \"deadline_headroom\": " << (deadline_ms / med_e2e);
        }
        json << "}";
    }
    json << "\n  ]\n}\n";
    std::cout << json.str();
    return 0;
}

int runProcessBench(Inference& engine, const AudioBuffer& audio, int chunk_size, int num_overlap,
                    int process_repeats, const std::string& model_path) {
    const std::string backend = engine.GetBackendName();
    std::cerr << "Backend: " << backend << std::endl;
    if (!backendOkForBenchmark(backend)) {
        std::cerr << "Error: unsupported backend for --process-bench, got " << backend << std::endl;
        return 2;
    }
    if (process_repeats < 1) process_repeats = 1;
    const int frames = frameCount(audio);
    const int sample_rate = engine.GetSampleRate();
    const double audio_sec = static_cast<double>(frames) / static_cast<double>(sample_rate);

    std::cerr << "Process-bench chunk_size=" << chunk_size << " overlap=" << num_overlap
              << " frames=" << frames << std::endl;

    auto run_once = [&]() -> double {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto stems =
            engine.Process(audio.data, chunk_size, num_overlap, nullptr);
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (stems.empty() || stems[0].empty()) {
            throw std::runtime_error("Process() returned empty stems");
        }
        double energy = 0.0;
        for (const auto& stem : stems) {
            for (float s : stem) energy += std::fabs(s);
        }
        if (energy <= 0.0) {
            throw std::runtime_error("Process() output energy is zero");
        }
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    const double warm_ms = run_once();
    std::cerr << "  warmup_wall_ms=" << warm_ms << std::endl;

    std::vector<double> walls;
    walls.reserve(static_cast<size_t>(process_repeats));
    for (int r = 0; r < process_repeats; ++r) {
        const double ms = run_once();
        walls.push_back(ms);
        std::cerr << "  repeat " << (r + 1) << " wall_ms=" << ms << std::endl;
    }
    const double med_wall_ms = medianOf(walls);
    const double med_wall_sec = med_wall_ms / 1000.0;
    const double rtf = med_wall_sec / audio_sec;
    const double speed_x = 1.0 / rtf;

    std::ostringstream json;
    json << "{\n"
         << "  \"mode\": \"process_overlap_add_pipelined\",\n"
         << "  \"model_gguf\": ";
    appendJsonString(json, model_path);
    json << ",\n"
         << "  \"backend\": \"" << backend << "\",\n"
         << "  \"sample_rate\": " << sample_rate << ",\n"
         << "  \"input_frames_per_channel\": " << frames << ",\n"
         << "  \"input_duration_sec\": " << audio_sec << ",\n"
         << "  \"chunk_size\": " << chunk_size << ",\n"
         << "  \"num_overlap\": " << num_overlap << ",\n"
         << "  \"process_repeats\": " << process_repeats << ",\n"
         << "  \"warmup_wall_ms\": " << warm_ms << ",\n"
         << "  \"median_wall_ms\": " << med_wall_ms << ",\n"
         << "  \"median_wall_sec\": " << med_wall_sec << ",\n"
         << "  \"rtf\": " << rtf << ",\n"
         << "  \"speed_x_realtime\": " << speed_x << "\n"
         << "}\n";
    std::cout << json.str();
    return 0;
}

double percentileSorted(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (p <= 0.0) return sorted.front();
    if (p >= 1.0) return sorted.back();
    const double idx = p * static_cast<double>(sorted.size() - 1);
    const int lo = static_cast<int>(std::floor(idx));
    const int hi = static_cast<int>(std::ceil(idx));
    if (lo == hi) return sorted[static_cast<size_t>(lo)];
    const double t = idx - static_cast<double>(lo);
    return sorted[static_cast<size_t>(lo)] * (1.0 - t) + sorted[static_cast<size_t>(hi)] * t;
}

int runLiveStress(Inference& engine, const AudioBuffer& audio, int samples, int iters) {
    const std::string backend = engine.GetBackendName();
    std::cerr << "Backend: " << backend << std::endl;
    if (!backendOkForBenchmark(backend)) return 2;
    if (iters < 1) iters = 1;
    const int frames = frameCount(audio);
    if (frames < samples) {
        throw std::runtime_error("WAV too short for live stress");
    }
    const auto window = interleavedWindow(audio, 0, samples);
    const double deadline_ms =
        (static_cast<double>(kLiveHopSamples) / static_cast<double>(engine.GetSampleRate())) * 1000.0;

    Inference::ChunkStageMs warm_stage;
    engine.ProcessChunk(window, warm_stage);

    std::vector<double> e2e;
    e2e.reserve(static_cast<size_t>(iters));
    int over_deadline = 0;
    for (int i = 0; i < iters; ++i) {
        Inference::ChunkStageMs stage;
        engine.ProcessChunk(window, stage);
        const double ms = stage.stft_ms + stage.pack_ms + stage.forward_ms + stage.post_ms;
        e2e.push_back(ms);
        if (ms > deadline_ms) ++over_deadline;
    }

    std::vector<double> sorted = e2e;
    std::sort(sorted.begin(), sorted.end());
    const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    const double mean = sum / static_cast<double>(sorted.size());
    double var = 0.0;
    for (double v : sorted) {
        const double d = v - mean;
        var += d * d;
    }
    const double stddev = std::sqrt(var / static_cast<double>(sorted.size()));

    std::ostringstream json;
    json << "{\n"
         << "  \"backend\": \"" << backend << "\",\n"
         << "  \"samples\": " << samples << ",\n"
         << "  \"live_hop_samples\": " << kLiveHopSamples << ",\n"
         << "  \"deadline_ms\": " << deadline_ms << ",\n"
         << "  \"iterations\": " << iters << ",\n"
         << "  \"over_deadline_count\": " << over_deadline << ",\n"
         << "  \"min_e2e_ms\": " << sorted.front() << ",\n"
         << "  \"median_e2e_ms\": " << medianOf(sorted) << ",\n"
         << "  \"mean_e2e_ms\": " << mean << ",\n"
         << "  \"p95_e2e_ms\": " << percentileSorted(sorted, 0.95) << ",\n"
         << "  \"p99_e2e_ms\": " << percentileSorted(sorted, 0.99) << ",\n"
         << "  \"max_e2e_ms\": " << sorted.back() << ",\n"
         << "  \"stddev_e2e_ms\": " << stddev << "\n"
         << "}\n";
    std::cout << json.str();
    return 0;
}

int runSeparate(Inference& engine, const AudioBuffer& audio, const std::string& out_dir,
                 int chunk_samples, int step_samples) {
    const std::string backend = engine.GetBackendName();
    std::cerr << "Backend: " << backend << std::endl;
    if (!backendOkForBenchmark(backend)) {
        std::cerr << "Error: unsupported backend for --separate, got " << backend << std::endl;
        return 2;
    }
    if (chunk_samples <= 0) chunk_samples = kTeacherChunk;
    if (step_samples <= 0) step_samples = kTeacherStep;
    const int frames = frameCount(audio);
    const int step = std::min(step_samples, chunk_samples);
    const auto starts = chunkStarts(frames, chunk_samples, step);
    const auto window = numpyHamming(chunk_samples);
    roformer_mlx::RollingOlaAccumulator acc(engine.GetNumStems(), 2, std::max(frames, 1));
    std::cerr << "Separate chunk=" << chunk_samples << " step=" << step
              << " windows=" << starts.size() << std::endl;

    for (size_t index = 0; index < starts.size(); ++index) {
        const int start = starts[index];
        const int length = std::min(chunk_samples, frames - start);
        const auto chunk = interleavedWindow(audio, start, length);
        const auto stems = engine.ProcessChunk(chunk);
        std::vector<float> planar(static_cast<size_t>(stems.size()) * 2 * static_cast<size_t>(length), 0.0f);
        for (int s = 0; s < static_cast<int>(stems.size()); ++s) {
            const auto& src = stems[static_cast<size_t>(s)];
            float* dst = planar.data() + static_cast<size_t>(s) * 2 * static_cast<size_t>(length);
            for (int i = 0; i < length; ++i) {
                dst[i] = src[static_cast<size_t>(i) * 2];
                dst[length + i] = src[static_cast<size_t>(i) * 2 + 1];
            }
        }
        const float* weights = window.data();
        std::vector<float> short_window;
        if (length != chunk_samples) {
            short_window.assign(window.begin(), window.begin() + length);
            weights = short_window.data();
        }
        acc.add(planar.data(), start, length, weights);
        std::cerr << "  window " << (index + 1) << "/" << starts.size() << std::endl;
    }

    std::filesystem::create_directories(out_dir);
    const int nstems = engine.GetNumStems();
    const int sample_rate = engine.GetSampleRate();
    for (int s = 0; s < nstems; ++s) {
        AudioBuffer out;
        out.channels = 2;
        out.sampleRate = static_cast<unsigned int>(sample_rate);
        out.data.resize(static_cast<size_t>(frames) * 2);
        for (int i = 0; i < frames; ++i) {
            out.data[static_cast<size_t>(i) * 2] = acc.normalized(s, 0, i);
            out.data[static_cast<size_t>(i) * 2 + 1] = acc.normalized(s, 1, i);
        }
        out.samples = out.data.size();
        const char* name = (s < 6) ? kStemNames[s] : "stem";
        const std::filesystem::path path = std::filesystem::path(out_dir) / (std::string(name) + ".wav");
        std::cerr << "Saving " << path.string() << std::endl;
        AudioFile::Save(path.string(), out);
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    int chunk_size = -1;
    int num_overlap = -1;
    bool chunk_size_set = false;
    bool num_overlap_set = false;
    bool bench = false;
    bool process_bench = false;
    bool live_stress = false;
    bool gpu_post = false;
    bool cpu_post = false;
    bool separate = false;
    int repeats = 3;
    int process_repeats = 1;
    int stress_iters = 1000;
    int separate_chunk = 0;
    int separate_step = 0;
    int bench_only = 0;
    int probe_samples = 0;
    std::string probe_dir;
    std::string separate_dir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--chunk-size" && i + 1 < argc) {
            chunk_size = std::stoi(argv[++i]);
            chunk_size_set = true;
        } else if (arg == "--overlap" && i + 1 < argc) {
            num_overlap = std::stoi(argv[++i]);
            num_overlap_set = true;
        } else if (arg == "--bench") {
            bench = true;
        } else if (arg == "--process-bench") {
            process_bench = true;
        } else if (arg == "--live-stress") {
            live_stress = true;
        } else if (arg == "--gpu-post") {
            gpu_post = true;
        } else if (arg == "--cpu-post") {
            cpu_post = true;
        } else if (arg == "--stress-iters" && i + 1 < argc) {
            stress_iters = std::stoi(argv[++i]);
        } else if (arg == "--process-repeats" && i + 1 < argc) {
            process_repeats = std::stoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (arg == "--bench-samples" && i + 1 < argc) {
            bench_only = std::stoi(argv[++i]);
        } else if (arg == "--separate") {
            separate = true;
        } else if (arg == "--chunk-samples" && i + 1 < argc) {
            separate_chunk = std::stoi(argv[++i]);
        } else if (arg == "--step" && i + 1 < argc) {
            separate_step = std::stoi(argv[++i]);
        } else if (arg == "--probe-dir" && i + 1 < argc) {
            probe_dir = argv[++i];
        } else if (arg == "--probe-samples" && i + 1 < argc) {
            probe_samples = std::stoi(argv[++i]);
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::string model_path = positional[0];
        std::cerr << "Initializing BSRoformer..." << std::endl;
        Inference engine(model_path);
        if (gpu_post) engine.SetUseCudaPost(true);
        if (cpu_post) engine.SetUseCudaPost(false);
        if (!chunk_size_set) chunk_size = engine.GetDefaultChunkSize();
        if (!num_overlap_set) num_overlap = engine.GetDefaultNumOverlap();

        if (!probe_dir.empty()) {
            if (positional.size() < 2 || probe_samples <= 0) {
                print_usage(argv[0]);
                return 1;
            }
            return runProbe(engine, loadStereo44100(engine, positional[1]), probe_samples, probe_dir);
        }
        if (bench) {
            if (positional.size() < 2) {
                print_usage(argv[0]);
                return 1;
            }
            return runBench(engine, loadStereo44100(engine, positional[1]), repeats, bench_only);
        }
        if (process_bench) {
            if (positional.size() < 2) {
                print_usage(argv[0]);
                return 1;
            }
            return runProcessBench(engine, loadStereo44100(engine, positional[1]), chunk_size,
                                   num_overlap, process_repeats, model_path);
        }
        if (live_stress) {
            if (positional.size() < 2) {
                print_usage(argv[0]);
                return 1;
            }
            return runLiveStress(engine, loadStereo44100(engine, positional[1]), 81920, stress_iters);
        }
        if (separate) {
            if (positional.size() < 3) {
                print_usage(argv[0]);
                return 1;
            }
            return runSeparate(engine, loadStereo44100(engine, positional[1]), positional[2],
                               separate_chunk, separate_step);
        }
        if (positional.size() < 3) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string input_path = positional[1];
        const std::string output_path = positional[2];
        AudioBuffer input_audio = loadStereo44100(engine, input_path);
        std::cerr << "Backend: " << engine.GetBackendName() << std::endl;
        if (!backendOkForBenchmark(engine.GetBackendName())) {
            std::cerr << "Error: unsupported backend for file Process()" << std::endl;
            return 2;
        }

        std::cerr << "Processing with chunk_size=" << chunk_size
                  << ", overlap=" << num_overlap << std::endl;
        const auto process_start = std::chrono::high_resolution_clock::now();
        auto progress_callback = [](float progress) {
            int barWidth = 50;
            std::cerr << "[";
            int pos = static_cast<int>(barWidth * progress);
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cerr << "=";
                else if (i == pos) std::cerr << ">";
                else std::cerr << " ";
            }
            std::cerr << "] " << int(progress * 100.0) << " %\r";
            std::cerr.flush();
        };
        std::vector<std::vector<float>> output_stems =
            engine.Process(input_audio.data, chunk_size, num_overlap, progress_callback);
        std::cerr << std::string(70, ' ') << "\r";
        const auto process_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = process_end - process_start;
        std::cerr << "Processed in " << diff.count() << " seconds." << std::endl;

        const int num_stems = static_cast<int>(output_stems.size());
        const int required_sr = engine.GetSampleRate();
        for (int i = 0; i < num_stems; ++i) {
            std::string current_output_path = output_path;
            if (num_stems > 1) {
                const size_t dot_pos = output_path.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    current_output_path = output_path.substr(0, dot_pos) + "_stem_" +
                                          std::to_string(i) + output_path.substr(dot_pos);
                } else {
                    current_output_path = output_path + "_stem_" + std::to_string(i);
                }
            }
            AudioBuffer output_audio_buf;
            output_audio_buf.data = std::move(output_stems[static_cast<size_t>(i)]);
            output_audio_buf.channels = 2;
            output_audio_buf.sampleRate = static_cast<unsigned int>(required_sr);
            output_audio_buf.samples = output_audio_buf.data.size();
            std::cerr << "Saving output stem " << i << ": " << current_output_path << std::endl;
            AudioFile::Save(current_output_path, output_audio_buf);
        }
        std::cerr << "Done!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
