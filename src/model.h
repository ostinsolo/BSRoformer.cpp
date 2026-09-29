#pragma once
#include <string>
#include <vector>
#include <memory>
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

// Forward declarations
struct ggml_context;
struct ggml_cgraph;
struct gguf_context;

/**
 * BSRoformer Model
 * 
 * This class handles:
 * 1. Loading weights from GGUF file
 * 2. Providing access to weights and buffers
 * 3. Building GGML computation graphs for each component
 * 
 * Execution is handled by test/inference code using these graphs.
 */
class BSRoformer {
public:
    BSRoformer();
    ~BSRoformer();

    // Initialize model from GGUF file
    void Initialize(const std::string& model_path);

    // GGML CPU graph thread count (ignored on CUDA). Default GGML_DEFAULT_N_THREADS until set.
    void SetGraphThreads(int n_threads);
    int GetGraphThreads() const { return graph_threads_; }

    // ========== Accessors for weights and config ==========
    
    // Get weight tensor by name
    ggml_tensor* GetWeight(const std::string& name) const;
    
    // Get backend
    ggml_backend_t GetBackend() const { return backend_; }
    
    // Get weights context (for creating tensors from weights)
    ggml_context* GetWeightsContext() const { return ctx_weights_; }
    
    // ========== Model Config Accessors ==========
    int GetDim() const { return dim_; }
    int GetDepth() const { return depth_; }
    int GetNumBands() const { return num_bands_; }
    int GetNFFT() const { return n_fft_; }
    int GetHopLength() const { return hop_length_; }
    int GetWinLength() const { return win_length_; }
    int GetNumStems() const { return num_stems_; }
    bool GetSkipConnection() const { return skip_connection_; }
    bool GetSTFTNormalized() const { return stft_normalized_; }
    int GetZeroDC() const { return zero_dc_; }
    int GetSampleRate() const { return sample_rate_; }
    int GetMlpExpansionFactor() const { return mlp_expansion_factor_; }
    
    // BS Roformer Support
    const std::string& GetArchitecture() const { return architecture_; }
    bool HasFinalNorm() const { return has_final_norm_; }
    bool GetTransformerNormOutput() const { return transformer_norm_output_; }
    
    // Inference defaults (from GGUF, can be overridden at runtime)
    int GetDefaultChunkSize() const { return default_chunk_size_; }
    int GetDefaultNumOverlap() const { return default_num_overlap_; }
    
    // ========== Buffer Accessors ==========
    const std::vector<int>& GetFreqIndices() const { return freq_indices_; }
    const std::vector<int>& GetNumBandsPerFreq() const { return num_bands_per_freq_; }
    const std::vector<int>& GetNumFreqsPerBand() const { return num_freqs_per_band_; }
    
    // Calculate dim_inputs for each band (num_freqs * 4 for stereo complex)
    std::vector<int> GetDimInputs() const;
    int GetTotalDimInput() const;
    
    // ========== Graph Building Functions ==========
    // These functions build GGML computation graph nodes.
    // They don't execute - execution is done by caller with gallocr + backend_graph_compute.
    
    /**
     * Build BandSplit subgraph
     * @param ctx Computation context (must have no_alloc=true)
     * @param input Input tensor [total_dim_input, n_frames, batch]
     * @param gf Graph to add nodes to
     * @return Output tensor [dim, num_bands, n_frames, batch]
     */
    ggml_tensor* BuildBandSplitGraph(
        ggml_context* ctx,
        ggml_tensor* input,
        ggml_cgraph* gf,
        int n_frames,
        int batch = 1
    );
    
    /**
     * Build Transformer layers subgraph (Time + Freq transformers)
     * @param ctx Computation context
     * @param input Input tensor [dim, num_bands, n_frames, batch]
     * @param gf Graph to add nodes to
     * @param pos_time_exp Expanded position tensor for time RoPE [T * F * B], with repeating [0..T-1] * (F*B) times
     * @param pos_freq_exp Expanded position tensor for freq RoPE [F * T * B], with repeating [0..F-1] * (T*B) times
     * @return Output tensor [dim, num_bands, n_frames, batch]
     */
    ggml_tensor* BuildTransformersGraph(
        ggml_context* ctx,
        ggml_tensor* input,
        ggml_cgraph* gf,
        ggml_tensor* pos_time_exp,
        ggml_tensor* pos_freq_exp,
        int n_frames,
        int batch = 1
    );
    
    /**
     * Build MaskEstimator subgraph
     * @param ctx Computation context
     * @param input Input tensor [dim, num_bands, n_frames, batch]
     * @param gf Graph to add nodes to
     * @return Output tensor [total_mask_dim, n_frames, batch]
     */
    ggml_tensor* BuildMaskEstimatorGraph(
        ggml_context* ctx,
        ggml_tensor* input,
        ggml_cgraph* gf,
        int n_frames,
        int batch = 1
    );

private:
    void ApplyGraphThreadsToBackend();

    // GGML Contexts
    ggml_context* ctx_weights_ = nullptr;

    // Backend
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_weights_ = nullptr;
    int graph_threads_ = 0;

    // Model Config
    int dim_ = 384;
    int depth_ = 6;
    int num_bands_ = 60;
    int heads_ = 8;
    int dim_head_ = 64;
    int n_fft_ = 2048;
    int hop_length_ = 441;
    int win_length_ = 2048;
    
    // New Params
    int num_stems_ = 1;
    bool skip_connection_ = false;
    bool stft_normalized_ = false;
    bool zero_dc_ = false;
    int mask_estimator_depth_ = 1;
    int mlp_expansion_factor_ = 4;
    
    // BS Roformer Specific
    std::string architecture_ = "mel_band";  // "mel_band" or "bs"
    bool has_final_norm_ = false;            // BS has a global final norm
    bool transformer_norm_output_ = true;    // MelBand=true, BS=false
    int mlp_num_layers_ = 3;                 // Detected from weights (BS=2 for depth=2)
    int sample_rate_ = 44100;

    // Inference defaults
    int default_chunk_size_ = 352800;
    int default_num_overlap_ = 2;
    
    // Buffers loaded from GGUF
    std::vector<int> freq_indices_;
    std::vector<int> num_bands_per_freq_;
    std::vector<int> num_freqs_per_band_;
    
    // Helper to load GGUF
    void LoadWeights(const std::string& path);
};
