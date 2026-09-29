#include "bs_roformer/inference.h"
#include "bs_roformer/audio.h"
#include "roformer_mlx/rolling_ola.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kBenchWindows[] = {81920, 102400, 153600, 204800, 307200, 409600};
constexpr int kTeacherChunk = 409600;
constexpr int kTeacherStep = 352800;
constexpr const char* kStemNames[] = {
    "bass", "drums", "other", "vocals", "guitar", "piano"};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <model.gguf> <input.wav> <output.wav> [options]\n";
    std::cerr << "       " << program_name << " <model.gguf> --bench <input.wav> [--repeats N]\n";
    std::cerr << "       " << program_name << " <model.gguf> --separate <input.wav> <out_dir>\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --chunk-size <N>   Chunk size in samples (default: from model, fallback 352800)\n";
    std::cerr << "  --overlap <N>      Number of overlaps for crossfade (default: from model, fallback 2)\n";
    std::cerr << "  --bench <wav>      Warm stage-timing ladder (81920 .. 409600) on the WAV\n";
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
    if (!backendIsCuda(backend)) {
        std::cerr << "Error: expected a CUDA backend, got " << backend << std::endl;
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
             << ", \"warmup_abs_energy\": " << warm_energy << "}";
    }
    json << "\n  ]\n}\n";
    std::cout << json.str();
    return 0;
}

int runSeparate(Inference& engine, const AudioBuffer& audio, const std::string& out_dir,
                 int chunk_samples, int step_samples) {
    const std::string backend = engine.GetBackendName();
    std::cerr << "Backend: " << backend << std::endl;
    if (!backendIsCuda(backend)) {
        std::cerr << "Error: expected a CUDA backend, got " << backend << std::endl;
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
    bool separate = false;
    int repeats = 3;
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
        if (!backendIsCuda(engine.GetBackendName())) {
            std::cerr << "Error: expected a CUDA backend" << std::endl;
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
