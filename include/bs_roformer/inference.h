#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
// Forward declaration
class BSRoformer;

struct ggml_context;
struct ggml_cgraph;
struct ggml_gallocr;
struct ggml_tensor;

class Inference {
public:
    using CancelCallback = std::function<bool()>;

    Inference(const std::string& model_path);
    ~Inference();

    // Process a full audio track (interleaved stereo float32)
    // Uses overlap-add chunking to handle long files
    // Process a full audio track (interleaved stereo float32)
    // Returns a vector of stems, where each stem is an interleaved stereo float vector
    std::vector<std::vector<float>> Process(const std::vector<float>& input_audio, 
                               int chunk_size = 352800, 
                               int num_overlap = 2,
                               std::function<void(float)> progress_callback = nullptr,
                               CancelCallback cancel_callback = nullptr);

    // Stage split for one ProcessChunk call. forward_ms includes host/device copies
    // around ggml_backend_graph_compute. stft_ms is STFT plus model-input pack.
    // post_ms is mask apply plus iSTFT.
    struct ChunkStageMs {
        double stft_ms = 0.0;
        double pack_ms = 0.0;
        double h2d_ms = 0.0;
        double graph_ms = 0.0;
        double d2h_ms = 0.0;
        double forward_ms = 0.0;
        double post_ms = 0.0;
    };

    struct ChunkProbe {
        int n_frames = 0;
        int num_freq_indices = 0;
        std::vector<float> stft_flattened;
        std::vector<float> mask_output;
    };

    ChunkProbe ProbeChunk(const std::vector<float>& chunk_audio);

    struct ChunkForwardArtifacts {
        int n_frames = 0;
        std::vector<std::vector<float>> stft_outputs;
        std::vector<float> mask_output;
    };

    // STFT + GGML forward without CPU post (for CUDA post benchmarks).
    ChunkForwardArtifacts CaptureChunkForward(const std::vector<float>& chunk_audio);

    void PostProcessCpu(const std::vector<float>& mask_output,
                        const std::vector<std::vector<float>>& stft_outputs,
                        int n_frames,
                        std::vector<std::vector<float>>& output_audio);

    int GetNFFT() const;
    int GetHopLength() const;
    int GetWinLength() const;
    bool GetZeroDC() const;
    const std::vector<int>& GetFreqIndices() const;
    const std::vector<int>& GetNumBandsPerFreq() const;

    // Low-level chunk processing (public for testing)
    std::vector<std::vector<float>> ProcessChunk(const std::vector<float>& chunk_audio);
    std::vector<std::vector<float>> ProcessChunk(const std::vector<float>& chunk_audio,
                                                  ChunkStageMs& stages);

    std::string GetBackendName() const;

    // Get model's recommended inference defaults
    int GetDefaultChunkSize() const;
    int GetDefaultNumOverlap() const;
    int GetSampleRate() const;
    int GetNumStems() const;

    // Static helper for Overlap-Add logic (matches Python exactly)
    // model_func: input [samples], output [stems][samples] (interleaved stereo)
    using ModelCallback = std::function<std::vector<std::vector<float>>(const std::vector<float>&)>;
    static std::vector<std::vector<float>> ProcessOverlapAdd(const std::vector<float>& input_audio, 
                                                int chunk_size, 
                                                int num_overlap,
                                                ModelCallback model_func,
                                                std::function<void(float)> progress_callback = nullptr,
                                                CancelCallback cancel_callback = nullptr);

private:
    // Pipelined Overlap-Add
    std::vector<std::vector<float>> ProcessOverlapAddPipelined(const std::vector<float>& input_audio, 
                                                  int chunk_size, 
                                                  int num_overlap,
                                                  std::function<void(float)> progress_callback,
                                                  CancelCallback cancel_callback);

private:
    std::unique_ptr<BSRoformer> model_;
    struct CpuScratch;

    struct GraphState {
        int n_frames = -1;
        ggml_context* ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_gallocr* allocr = nullptr;
        ggml_tensor* input_tensor = nullptr;
        ggml_tensor* pos_time = nullptr;
        ggml_tensor* pos_freq = nullptr;
        ggml_tensor* mask_out_tensor = nullptr;
        std::vector<int32_t> pos_time_data;
        std::vector<int32_t> pos_freq_data;
        size_t compute_buffer_size = 0;
        int graph_nodes = 0;
    };

    std::unordered_map<int, std::unique_ptr<GraphState>> graph_cache_;

    // Pipelined State Data
    struct ChunkState {
        int id = -1;
        std::vector<float> input_audio;       // Original chunk audio
        std::vector<float> stft_flattened;    // [Prepared Input for GPU]
        std::vector<std::vector<float>> stft_outputs; // Kept for reconstruction
        int n_frames = 0;
        
        std::vector<float> mask_output;       // Output from GPU
        std::vector<std::vector<float>> final_audio;       // Result after ISTFT [stems][samples]
        double h2d_ms = 0.0;
        double graph_ms = 0.0;
        double d2h_ms = 0.0;
    };

    // Helper to ensure graph is built for specific n_frames
    GraphState* EnsureGraph(int n_frames);
    void ReleaseGraphState(GraphState& graph);
    void ClearGraphCache();

    void ComputeSTFT(const std::vector<float>& input_audio,
                     std::vector<std::vector<float>>& stft_outputs,
                     int& n_frames,
                     CpuScratch& scratch);
                     
    void PrepareModelInput(const std::vector<std::vector<float>>& stft_outputs,
                           int n_frames,
                           std::vector<float>& model_input_rearranged);

    void PostProcessAndISTFT(const std::vector<float>& mask_output,
                             const std::vector<std::vector<float>>& stft_outputs,
                             int n_frames,
                             std::vector<std::vector<float>>& output_audio,
                             CpuScratch& scratch);

    // Pipeline Steps
    std::shared_ptr<ChunkState> PreProcessChunk(const std::vector<float>& chunk_audio, int id, CpuScratch& scratch);
    void RunInference(std::shared_ptr<ChunkState> state);
    void PostProcessChunk(std::shared_ptr<ChunkState> state, CpuScratch& scratch);
};
