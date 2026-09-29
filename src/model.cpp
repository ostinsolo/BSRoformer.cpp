#include "model.h"
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <gguf.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
bool EnvIsSet(const char* name) {
#if defined(_MSC_VER)
    size_t required = 0;
    return ::getenv_s(&required, nullptr, 0, name) == 0 && required > 0;
#else
    return std::getenv(name) != nullptr;
#endif
}

ggml_tensor* ApplyRopeExtNormalInplace(ggml_context* ctx, ggml_tensor* tensor, ggml_tensor* pos, int dim_head) {
    return ggml_rope_ext_inplace(ctx, tensor, pos, nullptr, dim_head,
                                 GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

ggml_tensor* RestoreDirectRopeFlat(ggml_context* ctx, ggml_tensor* rope_flat,
                                   int dim_head, int heads, int seq, int batch) {
    return ggml_view_4d(ctx, rope_flat, dim_head, heads, seq, batch,
                        rope_flat->nb[1], rope_flat->nb[2],
                        static_cast<size_t>(seq) * rope_flat->nb[2], 0);
}

bool CanSplitQkvWeight(const ggml_tensor* weight, int dim_inner) {
    return weight && weight->ne[1] == 3 * dim_inner;
}

void LogSplitQkvFallbackOnce(const char* scope, const ggml_tensor* weight, int dim_inner) {
    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;
    std::cerr << "[Model] QKV weight shape for " << scope << " is ["
              << (weight ? weight->ne[0] : 0) << ", "
              << (weight ? weight->ne[1] : 0)
              << "]; expected second dimension " << 3 * dim_inner
              << ". Falling back to fused QKV projection." << std::endl;
}

void SetTensorName(ggml_tensor* tensor, const std::string& name) {
    if (tensor) {
        ggml_set_name(tensor, name.c_str());
    }
}

ggml_tensor* ConcatBalancedRange(ggml_context* ctx,
                                 const std::vector<ggml_tensor*>& tensors,
                                 size_t begin,
                                 size_t end,
                                 int dim) {
    if (begin >= end) {
        return nullptr;
    }
    if (end - begin == 1) {
        return tensors[begin];
    }

    size_t mid = begin + (end - begin) / 2;
    ggml_tensor* left = ConcatBalancedRange(ctx, tensors, begin, mid, dim);
    ggml_tensor* right = ConcatBalancedRange(ctx, tensors, mid, end, dim);
    return ggml_concat(ctx, left, right, dim);
}

ggml_tensor* ConcatBalanced(ggml_context* ctx,
                            const std::vector<ggml_tensor*>& tensors,
                            int dim) {
    return ConcatBalancedRange(ctx, tensors, 0, tensors.size(), dim);
}
}

BSRoformer::BSRoformer() {
}

BSRoformer::~BSRoformer() {
    if (buffer_weights_) ggml_backend_buffer_free(buffer_weights_);
    if (backend_) ggml_backend_free(backend_);
    if (ctx_weights_) ggml_free(ctx_weights_);
}

void BSRoformer::Initialize(const std::string& model_path) {
    if (EnvIsSet("BSR_FORCE_CPU")) {
        backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else {
        backend_ = ggml_backend_init_best();
    }
    if (!backend_) {
        throw std::runtime_error("Failed to initialize ggml backend");
    }
    std::cout << "Using backend: " << ggml_backend_name(backend_) << std::endl;
    ApplyGraphThreadsToBackend();

    LoadWeights(model_path);
}

void BSRoformer::SetGraphThreads(int n_threads) {
    if (n_threads < 1) {
        n_threads = GGML_DEFAULT_N_THREADS;
    }
    graph_threads_ = n_threads;
    ApplyGraphThreadsToBackend();
}

void BSRoformer::ApplyGraphThreadsToBackend() {
    if (!backend_ || graph_threads_ < 1) {
        return;
    }
    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, graph_threads_);
        std::cout << "GGML CPU graph threads: " << graph_threads_ << std::endl;
    }
}

void BSRoformer::LoadWeights(const std::string& path) {
    std::cout << "Loading model from " << path << std::endl;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_weights_,
    };

    struct gguf_context* ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) {
        throw std::runtime_error("Failed to load GGUF file: " + path);
    }

    // 1. Read Architecture first to determine key prefix
    int kv_idx = gguf_find_key(ctx_gguf, "general.architecture");
    if (kv_idx >= 0) {
        architecture_ = gguf_get_val_str(ctx_gguf, kv_idx);
    } else {
        throw std::runtime_error("Key 'general.architecture' not found in GGUF file. Please re-convert the model with the latest script.");
    }
    
    std::cout << "Architecture: " << architecture_ << std::endl;
    
    // Normalization for legacy models (if any) or simplified internal handling
    if (architecture_ == "bs") architecture_ = "bs_roformer";
    if (architecture_ == "mel_band") architecture_ = "mel_band_roformer";

    std::string kp = architecture_ + "."; // key prefix, e.g. "bs_roformer." or "mel_band_roformer."

    // Set internal flags based on architecture
    if (architecture_ == "bs_roformer") {
        has_final_norm_ = true;
        transformer_norm_output_ = false;
    } else {
        // mel_band_roformer
        has_final_norm_ = false;
        transformer_norm_output_ = true;
    }

    // 2. Read Hyperparameters using key prefix
    
    kv_idx = gguf_find_key(ctx_gguf, (kp + "stft_n_fft").c_str());
    if (kv_idx >= 0) n_fft_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
    
    kv_idx = gguf_find_key(ctx_gguf, (kp + "stft_hop_length").c_str());
    if (kv_idx >= 0) hop_length_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
    
    kv_idx = gguf_find_key(ctx_gguf, (kp + "stft_win_length").c_str());
    if (kv_idx >= 0) win_length_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, (kp + "dim").c_str());
    if (kv_idx >= 0) dim_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, (kp + "num_bands").c_str());
    if (kv_idx >= 0) num_bands_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
    
    kv_idx = gguf_find_key(ctx_gguf, (kp + "depth").c_str());
    if (kv_idx >= 0) depth_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);

    // New Parameters
    kv_idx = gguf_find_key(ctx_gguf, (kp + "num_stems").c_str());
    if (kv_idx >= 0) num_stems_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
    
    kv_idx = gguf_find_key(ctx_gguf, (kp + "skip_connection").c_str());
    if (kv_idx >= 0) skip_connection_ = gguf_get_val_bool(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, (kp + "stft_normalized").c_str());
    if (kv_idx >= 0) stft_normalized_ = gguf_get_val_bool(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, (kp + "zero_dc").c_str());
    if (kv_idx >= 0) zero_dc_ = gguf_get_val_bool(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, (kp + "mask_estimator_depth").c_str());
    if (kv_idx >= 0) mask_estimator_depth_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, (kp + "mlp_expansion_factor").c_str());
    if (kv_idx >= 0) mlp_expansion_factor_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, (kp + "sample_rate").c_str());
    if (kv_idx >= 0) sample_rate_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
    
    // Inference defaults (optional, fallback to hardcoded values)
    kv_idx = gguf_find_key(ctx_gguf, (kp + "default_chunk_size").c_str());
    if (kv_idx >= 0) default_chunk_size_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
    
    kv_idx = gguf_find_key(ctx_gguf, (kp + "default_num_overlap").c_str());
    if (kv_idx >= 0) default_num_overlap_ = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
    
    kv_idx = gguf_find_key(ctx_gguf, (kp + "linear_transformer_depth").c_str());
    if (kv_idx >= 0) {
        int lin_depth = (int)gguf_get_val_u32(ctx_gguf, kv_idx);
        if (lin_depth > 0) {
            std::cerr << "\n[WARNING] Model uses Linear Attention (depth=" << lin_depth 
                      << "). This is NOT supported yet. Results will be incorrect.\n" << std::endl;
        }
    }

    std::cout << "Model Config: n_fft=" << n_fft_ << ", hop_length=" << hop_length_ 
              << ", num_bands=" << num_bands_ << ", dim=" << dim_ << ", depth=" << depth_ 
              << ", num_stems=" << num_stems_ << ", skip_conn=" << skip_connection_ << std::endl;
    std::cout << "Inference Defaults: chunk_size=" << default_chunk_size_ 
              << ", num_overlap=" << default_num_overlap_ << std::endl;

    // 2. Allocate backend buffer for ALL tensors
    buffer_weights_ = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_weights_, 
        ggml_backend_get_default_buffer_type(backend_)
    );
    if (!buffer_weights_) {
        throw std::runtime_error("Failed to allocate weight buffer");
    }

    // 3. Read data from file and upload to backend
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("Cannot open file");
    
    size_t data_offset = gguf_get_data_offset(ctx_gguf);
    
    struct ggml_tensor* t = ggml_get_first_tensor(ctx_weights_);
    std::vector<uint8_t> read_buf;
    
    while (t) {
        int tid = gguf_find_tensor(ctx_gguf, t->name);
        if (tid >= 0) {
            size_t offset = data_offset + gguf_get_tensor_offset(ctx_gguf, tid);
            size_t size = ggml_nbytes(t);
            
            if (read_buf.size() < size) read_buf.resize(size);
            
            fseek(file, (long)offset, SEEK_SET);
            fread(read_buf.data(), 1, size, file);
            
            // Upload to backend
            ggml_backend_tensor_set(t, read_buf.data(), 0, size);
            
            // Cache important buffers
            if (std::string(t->name) == "buffer_freq_indices") {
                freq_indices_.resize(ggml_nelements(t));
                if (t->type == GGML_TYPE_I32) {
                    memcpy(freq_indices_.data(), read_buf.data(), size);
                }
                std::cout << "  Loaded freq_indices: " << freq_indices_.size() << " indices" << std::endl;
            }
            if (std::string(t->name) == "buffer_num_bands_per_freq") {
                num_bands_per_freq_.resize(ggml_nelements(t));
                if (t->type == GGML_TYPE_I32) {
                    memcpy(num_bands_per_freq_.data(), read_buf.data(), size);
                }
            }
            if (std::string(t->name) == "buffer_num_freqs_per_band") {
                num_freqs_per_band_.resize(ggml_nelements(t));
                if (t->type == GGML_TYPE_I32) {
                    memcpy(num_freqs_per_band_.data(), read_buf.data(), size);
                }
            }
        }
        
        t = ggml_get_next_tensor(ctx_weights_, t);
    }
    
    fclose(file);
    
    int n_tensors = gguf_get_n_tensors(ctx_gguf);
    std::cout << "Loaded " << n_tensors << " tensors" << std::endl;
    
    // Dynamic MLP detection
    // Try to find mask_est.0.freq.0.mlp.{N}.weight
    mlp_num_layers_ = 0;
    const int MAX_MLP_LAYERS = 20;
    for (int idx = 0; idx <= MAX_MLP_LAYERS; idx += 2) {  // Check indices 0, 2, 4... up to MAX
        std::string probe = "mask_est.0.freq.0.mlp." + std::to_string(idx) + ".weight";
        if (GetWeight(probe) != nullptr) {
            mlp_num_layers_++;
        } else {
            break;
        }
    }
    std::cout << "Detected MLP layers: " << mlp_num_layers_ << std::endl;

    gguf_free(ctx_gguf);
}

ggml_tensor* BSRoformer::GetWeight(const std::string& name) const {
    return ggml_get_tensor(ctx_weights_, name.c_str());
}

std::vector<int> BSRoformer::GetDimInputs() const {
    std::vector<int> dim_inputs(num_bands_);
    for (int i = 0; i < num_bands_; ++i) {
        int num_freqs = num_freqs_per_band_[i];
        dim_inputs[i] = num_freqs * 4;  // stereo=2, complex=2
    }
    return dim_inputs;
}

int BSRoformer::GetTotalDimInput() const {
    if (architecture_ == "bs") {
        // BS: All frequencies * stereo * complex
        int n_freq = n_fft_ / 2 + 1;
        return n_freq * 2 * 2;  // freq * stereo * complex
    }

    int total = 0;
    for (int i = 0; i < num_bands_; ++i) {
        total += num_freqs_per_band_[i] * 4;
    }
    return total;
}

// ========== Graph Building Functions ==========

ggml_tensor* BSRoformer::BuildBandSplitGraph(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_cgraph* gf,
    int n_frames,
    int batch
) {
    // Following test_10_full_model.cpp implementation
    // Input: [total_dim_input, n_frames, batch]
    // Output: [dim, num_bands, n_frames, batch]
    
    std::vector<int> dim_inputs = GetDimInputs();
    
    std::vector<ggml_tensor*> projected_bands;
    projected_bands.reserve(num_bands_);
    
    size_t offset_elements = 0;
    for (int i = 0; i < num_bands_; ++i) {
        int dim_in = dim_inputs[i];
        
        // View for this band's input
        ggml_tensor* band_input = ggml_view_3d(ctx, input,
                                               dim_in, n_frames, batch,
                                               input->nb[1], input->nb[2],
                                               offset_elements * sizeof(float));
        
        // Get RMSNorm gamma weight
        // band_split.{i}.norm.weight
        std::string gamma_name = "band_split." + std::to_string(i) + ".norm.weight";
        ggml_tensor* gamma = GetWeight(gamma_name);
        if (!gamma) {
            std::cerr << "Missing weight: " << gamma_name << std::endl;
            return nullptr;
        }
        
        // RMSNorm
        ggml_tensor* normed = ggml_rms_norm(ctx, band_input, 1e-12f);
        normed = ggml_mul(ctx, normed, gamma);
        
        // Get Linear weight and bias
        // band_split.{i}.linear.weight
        std::string w_name = "band_split." + std::to_string(i) + ".linear.weight";
        std::string b_name = "band_split." + std::to_string(i) + ".linear.bias";
        ggml_tensor* weight = GetWeight(w_name);
        ggml_tensor* bias = GetWeight(b_name);
        
        if (!weight || !bias) {
            std::cerr << "Missing weight: " << w_name << " or " << b_name << std::endl;
            return nullptr;
        }
        
        // Linear projection
        ggml_tensor* projected = ggml_mul_mat(ctx, weight, normed);
        projected = ggml_add(ctx, projected, bias);
        
        ggml_tensor* projected_band = ggml_reshape_4d(ctx, projected, dim_, 1, n_frames, batch);
        projected_bands.push_back(projected_band);
        
        offset_elements += dim_in;
    }
    
    return ConcatBalanced(ctx, projected_bands, 1);
}

ggml_tensor* BSRoformer::BuildTransformersGraph(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_cgraph* gf,
    ggml_tensor* pos_time_exp,
    ggml_tensor* pos_freq_exp,
    int n_frames,
    int batch
) {
    // Following test_10_full_model.cpp implementation
    // Input: [dim, num_bands, n_frames, batch]
    
    const int D = dim_;
    const int F = num_bands_;
    const int T = n_frames;
    const int B = batch;
    const int HEADS = heads_;
    const int DIM_HEAD = dim_head_;
    const int DIM_INNER = HEADS * DIM_HEAD;
    
    ggml_tensor* x = input;
    std::vector<ggml_tensor*> skip_outputs;
    
    for (int layer = 0; layer < depth_; ++layer) {
        if (skip_connection_) {
            for (ggml_tensor* s : skip_outputs) {
                x = ggml_add(ctx, x, s);
            }
        }
        // ========== TIME TRANSFORMER ==========
        // Permute: [D, F, T, B] -> [D, T, F, B]
        x = ggml_permute(ctx, x, 0, 2, 1, 3);
        // Required before reshape: removing this CONT makes ggml_reshape_3d assert
        // on non-contiguous input in CUDA CTest (test_component_layers/test_inference).
        x = ggml_cont(ctx, x);
        
        int fb = F * B;
        ggml_tensor* x_packed = ggml_reshape_3d(ctx, x, D, T, fb);
        
        std::string time_prefix = "blk." + std::to_string(layer) + ".time_attn";
        std::string time_ff_prefix = "blk." + std::to_string(layer) + ".time_ff";
        
        // Attention Block
        // blk.{l}.time_attn_norm.weight
        ggml_tensor* t_attn_norm_w = GetWeight(time_prefix + "_norm.weight");
        if (!t_attn_norm_w) { std::cerr << "Missing: " << time_prefix << "_norm.weight\n"; return nullptr; }
        
        ggml_tensor* x_norm = ggml_rms_norm(ctx, x_packed, 1e-12f);
        x_norm = ggml_mul(ctx, x_norm, t_attn_norm_w);
        
        // blk.{l}.time_attn_qkv.weight
        ggml_tensor* t_qkv_w = GetWeight(time_prefix + "_qkv.weight");
        if (!t_qkv_w) { std::cerr << "Missing: " << time_prefix << "_qkv.weight\n"; return nullptr; }

        ggml_tensor* V = nullptr;
        ggml_tensor* time_qk_source = nullptr;
        size_t time_q_offset = 0;
        size_t time_k_offset = static_cast<size_t>(DIM_INNER) * sizeof(float);
        if (CanSplitQkvWeight(t_qkv_w, DIM_INNER)) {
            ggml_tensor* t_qk_w = ggml_view_2d(ctx, t_qkv_w, t_qkv_w->ne[0], 2 * DIM_INNER,
                                              t_qkv_w->nb[1], 0);
            ggml_tensor* t_v_w = ggml_view_2d(ctx, t_qkv_w, t_qkv_w->ne[0], DIM_INNER,
                                             t_qkv_w->nb[1],
                                             static_cast<size_t>(2 * DIM_INNER) * t_qkv_w->nb[1]);

            ggml_tensor* qk_out = ggml_mul_mat(ctx, t_qk_w, x_norm);
            ggml_tensor* v_out = ggml_mul_mat(ctx, t_v_w, x_norm);
            time_qk_source = qk_out;
            SetTensorName(qk_out, "t" + std::to_string(layer) + ".qk.mm");
            SetTensorName(v_out, "t" + std::to_string(layer) + ".v.mm");

            ggml_tensor* V_view = ggml_view_4d(ctx, v_out, DIM_HEAD, T, HEADS, fb,
                                              v_out->nb[1], DIM_HEAD * sizeof(float), v_out->nb[2], 0);
            V = V_view;
            SetTensorName(V, "t" + std::to_string(layer) + ".v.fa");
        } else {
            LogSplitQkvFallbackOnce("time attention", t_qkv_w, DIM_INNER);
            ggml_tensor* qkv_out = ggml_mul_mat(ctx, t_qkv_w, x_norm);
            time_qk_source = qkv_out;
            SetTensorName(qkv_out, "t" + std::to_string(layer) + ".qkv.mm");

            ggml_tensor* V_view = ggml_view_4d(ctx, qkv_out, DIM_HEAD, T, HEADS, fb,
                                              qkv_out->nb[1], DIM_HEAD * sizeof(float), qkv_out->nb[2],
                                              2 * DIM_INNER * sizeof(float));
            V = ggml_cont(ctx, V_view);
            SetTensorName(V, "t" + std::to_string(layer) + ".v.fa");
        }

        int T_fb = T * fb;
        ggml_tensor* Q_flat = ggml_view_4d(ctx, time_qk_source, DIM_HEAD, HEADS, T_fb, 1,
                                           DIM_HEAD * sizeof(float), time_qk_source->nb[1],
                                           static_cast<size_t>(T_fb) * time_qk_source->nb[1],
                                           time_q_offset);
        ggml_tensor* K_flat = ggml_view_4d(ctx, time_qk_source, DIM_HEAD, HEADS, T_fb, 1,
                                           DIM_HEAD * sizeof(float), time_qk_source->nb[1],
                                           static_cast<size_t>(T_fb) * time_qk_source->nb[1],
                                           time_k_offset);
        SetTensorName(Q_flat, "t" + std::to_string(layer) + ".q.rope.in.direct");
        SetTensorName(K_flat, "t" + std::to_string(layer) + ".k.rope.in.direct");

        ggml_tensor* Q_rope_flat = ApplyRopeExtNormalInplace(ctx, Q_flat, pos_time_exp, DIM_HEAD);
        ggml_tensor* K_rope_flat = ApplyRopeExtNormalInplace(ctx, K_flat, pos_time_exp, DIM_HEAD);
        SetTensorName(Q_rope_flat, "t" + std::to_string(layer) + ".q.rope.out");
        SetTensorName(K_rope_flat, "t" + std::to_string(layer) + ".k.rope.out");

        ggml_tensor* Q_rope_perm = RestoreDirectRopeFlat(ctx, Q_rope_flat, DIM_HEAD, HEADS, T, fb);
        ggml_tensor* K_rope_perm = RestoreDirectRopeFlat(ctx, K_rope_flat, DIM_HEAD, HEADS, T, fb);
        ggml_tensor* Q_rope = ggml_permute(ctx, Q_rope_perm, 0, 2, 1, 3);
        ggml_tensor* K_rope = ggml_permute(ctx, K_rope_perm, 0, 2, 1, 3);
        SetTensorName(Q_rope, "t" + std::to_string(layer) + ".q.fa");
        SetTensorName(K_rope, "t" + std::to_string(layer) + ".k.fa");

        // Flash Attention
        // Inputs: [DIM_HEAD, T, HEADS, fb]
        // Output: [DIM_HEAD, HEADS, T, fb] (permuted)
        ggml_tensor* Q_fa = Q_rope;
        ggml_tensor* K_fa = K_rope;
        ggml_tensor* V_fa = V; // [DIM_HEAD, T, HEADS, fb]

        float scale = 1.0f / sqrtf(static_cast<float>(DIM_HEAD));
        ggml_tensor* attn_out_fa = ggml_flash_attn_ext(ctx, Q_fa, K_fa, V_fa, nullptr, scale, 0.0f, 0.0f);
        SetTensorName(attn_out_fa, "t" + std::to_string(layer) + ".fa.out");
        
        // Gates
        // blk.{l}.time_attn_gate.weight/bias
        ggml_tensor* t_gate_w = GetWeight(time_prefix + "_gate.weight");
        ggml_tensor* t_gate_b = GetWeight(time_prefix + "_gate.bias");
        if (!t_gate_w || !t_gate_b) { std::cerr << "Missing gates weights\n"; return nullptr; }
        
        ggml_tensor* gates = ggml_mul_mat(ctx, t_gate_w, x_norm);
        gates = ggml_add(ctx, gates, t_gate_b);
        gates = ggml_sigmoid(ctx, gates);
        SetTensorName(gates, "t" + std::to_string(layer) + ".gate.sig");

        ggml_tensor* gates_bcast = ggml_view_4d(ctx, gates, 1, HEADS, T, fb,
                                               gates->nb[0], gates->nb[1], gates->nb[2], 0);
        SetTensorName(gates_bcast, "t" + std::to_string(layer) + ".gate.fa");
        ggml_tensor* gated_out = ggml_mul(ctx, attn_out_fa, gates_bcast);
        SetTensorName(gated_out, "t" + std::to_string(layer) + ".gate.mul.fa");
        ggml_tensor* out_flat = ggml_reshape_3d(ctx, gated_out, DIM_INNER, T, fb);
        SetTensorName(out_flat, "t" + std::to_string(layer) + ".out.flat");
        
        // blk.{l}.time_attn_out.weight
        ggml_tensor* t_attn_out_w = GetWeight(time_prefix + "_out.weight");
        if (!t_attn_out_w) { std::cerr << "Missing to_out_weight\n"; return nullptr; }
        
        ggml_tensor* attn_block_out = ggml_mul_mat(ctx, t_attn_out_w, out_flat);
        ggml_tensor* x_resid1 = ggml_add(ctx, x_packed, attn_block_out);
        
        // FeedForward Block
        // blk.{l}.time_ff_norm.weight
        ggml_tensor* t_ff_norm_w = GetWeight(time_ff_prefix + "_norm.weight");
        if (!t_ff_norm_w) { std::cerr << "Missing ff norm\n"; return nullptr; }
        
        ggml_tensor* x_resid1_norm = ggml_rms_norm(ctx, x_resid1, 1e-12f);
        x_resid1_norm = ggml_mul(ctx, x_resid1_norm, t_ff_norm_w);
        
        // blk.{l}.time_ff_in.weight/bias
        ggml_tensor* t_ff_in_w = GetWeight(time_ff_prefix + "_in.weight");
        ggml_tensor* t_ff_in_b = GetWeight(time_ff_prefix + "_in.bias");
        if (!t_ff_in_w || !t_ff_in_b) { std::cerr << "Missing ff in weights\n"; return nullptr; }
        
        ggml_tensor* ff_proj_in = ggml_mul_mat(ctx, t_ff_in_w, x_resid1_norm);
        ff_proj_in = ggml_add(ctx, ff_proj_in, t_ff_in_b);
        
        ggml_tensor* gelu_out = ggml_gelu_erf(ctx, ff_proj_in);
        

        
        // blk.{l}.time_ff_out.weight/bias
        ggml_tensor* t_ff_out_w = GetWeight(time_ff_prefix + "_out.weight");
        ggml_tensor* t_ff_out_b = GetWeight(time_ff_prefix + "_out.bias");
        if (!t_ff_out_w || !t_ff_out_b) { std::cerr << "Missing ff out weights\n"; return nullptr; }
        
        ggml_tensor* ff_block_out = ggml_mul_mat(ctx, t_ff_out_w, gelu_out);
        ff_block_out = ggml_add(ctx, ff_block_out, t_ff_out_b);
        
        x_packed = ggml_add(ctx, x_resid1, ff_block_out);
        

        
        // Time Transformer Final Norm
        // Only if transformer_norm_output_ is true (MelBand)
        if (transformer_norm_output_) {
            std::string time_norm_name = "blk." + std::to_string(layer) + ".time_norm.weight";
            ggml_tensor* time_norm_w = GetWeight(time_norm_name);
            if (!time_norm_w) { std::cerr << "Missing: " << time_norm_name << "\n"; return nullptr; }
            
            x_packed = ggml_rms_norm(ctx, x_packed, 1e-12f);
            x_packed = ggml_mul(ctx, x_packed, time_norm_w);
        }
        
        x = ggml_reshape_4d(ctx, x_packed, D, T, F, B);
        x = ggml_permute(ctx, x, 0, 2, 1, 3);
        // Required before freq reshape: removing this CONT makes ggml_reshape_3d
        // assert on non-contiguous input in CUDA CTest.
        x = ggml_cont(ctx, x);
        
        // ========== FREQ TRANSFORMER ==========
        int tb = T * B;
        ggml_tensor* x_freq_packed = ggml_reshape_3d(ctx, x, D, F, tb);
        

        
        std::string freq_prefix = "blk." + std::to_string(layer) + ".freq_attn";
        std::string freq_ff_prefix = "blk." + std::to_string(layer) + ".freq_ff";
        
        ggml_tensor* f_attn_norm_w = GetWeight(freq_prefix + "_norm.weight");
        if (!f_attn_norm_w) { std::cerr << "Missing freq norm\n"; return nullptr; }
        
        ggml_tensor* x_fnorm = ggml_rms_norm(ctx, x_freq_packed, 1e-12f);
        x_fnorm = ggml_mul(ctx, x_fnorm, f_attn_norm_w);
        

        
        ggml_tensor* f_qkv_w = GetWeight(freq_prefix + "_qkv.weight");
        if (!f_qkv_w) { std::cerr << "Missing freq qkv\n"; return nullptr; }

        ggml_tensor* fV = nullptr;
        ggml_tensor* freq_qk_source = nullptr;
        size_t freq_q_offset = 0;
        size_t freq_k_offset = static_cast<size_t>(DIM_INNER) * sizeof(float);
        if (CanSplitQkvWeight(f_qkv_w, DIM_INNER)) {
            ggml_tensor* f_qk_w = ggml_view_2d(ctx, f_qkv_w, f_qkv_w->ne[0], 2 * DIM_INNER,
                                               f_qkv_w->nb[1], 0);
            ggml_tensor* f_v_w = ggml_view_2d(ctx, f_qkv_w, f_qkv_w->ne[0], DIM_INNER,
                                              f_qkv_w->nb[1],
                                              static_cast<size_t>(2 * DIM_INNER) * f_qkv_w->nb[1]);

            ggml_tensor* f_qk_out = ggml_mul_mat(ctx, f_qk_w, x_fnorm);
            ggml_tensor* f_v_out = ggml_mul_mat(ctx, f_v_w, x_fnorm);
            freq_qk_source = f_qk_out;
            SetTensorName(f_qk_out, "f" + std::to_string(layer) + ".qk.mm");
            SetTensorName(f_v_out, "f" + std::to_string(layer) + ".v.mm");

            ggml_tensor* fV_view = ggml_view_4d(ctx, f_v_out, DIM_HEAD, F, HEADS, tb,
                                               f_v_out->nb[1], DIM_HEAD * sizeof(float), f_v_out->nb[2], 0);
            fV = fV_view;
            SetTensorName(fV, "f" + std::to_string(layer) + ".v.fa");
        } else {
            LogSplitQkvFallbackOnce("freq attention", f_qkv_w, DIM_INNER);
            ggml_tensor* f_qkv_out = ggml_mul_mat(ctx, f_qkv_w, x_fnorm);
            freq_qk_source = f_qkv_out;
            SetTensorName(f_qkv_out, "f" + std::to_string(layer) + ".qkv.mm");

            ggml_tensor* fV_view = ggml_view_4d(ctx, f_qkv_out, DIM_HEAD, F, HEADS, tb,
                                               f_qkv_out->nb[1], DIM_HEAD * sizeof(float), f_qkv_out->nb[2],
                                               2 * DIM_INNER * sizeof(float));
            fV = ggml_cont(ctx, fV_view);
            SetTensorName(fV, "f" + std::to_string(layer) + ".v.fa");
        }

        int F_tb = F * tb;
        ggml_tensor* fQ_flat = ggml_view_4d(ctx, freq_qk_source, DIM_HEAD, HEADS, F_tb, 1,
                                            DIM_HEAD * sizeof(float), freq_qk_source->nb[1],
                                            static_cast<size_t>(F_tb) * freq_qk_source->nb[1],
                                            freq_q_offset);
        ggml_tensor* fK_flat = ggml_view_4d(ctx, freq_qk_source, DIM_HEAD, HEADS, F_tb, 1,
                                            DIM_HEAD * sizeof(float), freq_qk_source->nb[1],
                                            static_cast<size_t>(F_tb) * freq_qk_source->nb[1],
                                            freq_k_offset);
        SetTensorName(fQ_flat, "f" + std::to_string(layer) + ".q.rope.in.direct");
        SetTensorName(fK_flat, "f" + std::to_string(layer) + ".k.rope.in.direct");

        ggml_tensor* fQ_rope_flat = ApplyRopeExtNormalInplace(ctx, fQ_flat, pos_freq_exp, DIM_HEAD);
        ggml_tensor* fK_rope_flat = ApplyRopeExtNormalInplace(ctx, fK_flat, pos_freq_exp, DIM_HEAD);
        SetTensorName(fQ_rope_flat, "f" + std::to_string(layer) + ".q.rope.out");
        SetTensorName(fK_rope_flat, "f" + std::to_string(layer) + ".k.rope.out");

        ggml_tensor* fQ_rope_perm = RestoreDirectRopeFlat(ctx, fQ_rope_flat, DIM_HEAD, HEADS, F, tb);
        ggml_tensor* fK_rope_perm = RestoreDirectRopeFlat(ctx, fK_rope_flat, DIM_HEAD, HEADS, F, tb);
        ggml_tensor* fQ_rope = ggml_permute(ctx, fQ_rope_perm, 0, 2, 1, 3);
        ggml_tensor* fK_rope = ggml_permute(ctx, fK_rope_perm, 0, 2, 1, 3);
        SetTensorName(fQ_rope, "f" + std::to_string(layer) + ".q.fa");
        SetTensorName(fK_rope, "f" + std::to_string(layer) + ".k.fa");
        
        // Flash Attention (Freq)
        // Inputs: [DIM_HEAD, F, HEADS, tb]
        ggml_tensor* fQ_fa = fQ_rope;
        ggml_tensor* fK_fa = fK_rope;
        ggml_tensor* fV_fa = fV; // [DIM_HEAD, F, HEADS, tb]

        ggml_tensor* f_attn_out_fa = ggml_flash_attn_ext(ctx, fQ_fa, fK_fa, fV_fa, nullptr, scale, 0.0f, 0.0f);
        SetTensorName(f_attn_out_fa, "f" + std::to_string(layer) + ".fa.out");

        ggml_tensor* f_gate_w = GetWeight(freq_prefix + "_gate.weight");
        ggml_tensor* f_gate_b = GetWeight(freq_prefix + "_gate.bias");
        if (!f_gate_w || !f_gate_b) { std::cerr << "Missing freq gates\n"; return nullptr; }
        
        ggml_tensor* f_gates = ggml_mul_mat(ctx, f_gate_w, x_fnorm);
        f_gates = ggml_add(ctx, f_gates, f_gate_b);
        f_gates = ggml_sigmoid(ctx, f_gates);
        SetTensorName(f_gates, "f" + std::to_string(layer) + ".gate.sig");

        ggml_tensor* f_gates_bcast = ggml_view_4d(ctx, f_gates, 1, HEADS, F, tb,
                                                  f_gates->nb[0], f_gates->nb[1], f_gates->nb[2], 0);
        SetTensorName(f_gates_bcast, "f" + std::to_string(layer) + ".gate.fa");
        ggml_tensor* f_gated_out = ggml_mul(ctx, f_attn_out_fa, f_gates_bcast);
        SetTensorName(f_gated_out, "f" + std::to_string(layer) + ".gate.mul.fa");
        ggml_tensor* f_out_flat = ggml_reshape_3d(ctx, f_gated_out, DIM_INNER, F, tb);
        SetTensorName(f_out_flat, "f" + std::to_string(layer) + ".out.flat");
        

        
        ggml_tensor* f_attn_out_w = GetWeight(freq_prefix + "_out.weight");
        if (!f_attn_out_w) { std::cerr << "Missing freq to_out\n"; return nullptr; }
        
        ggml_tensor* f_attn_block_out = ggml_mul_mat(ctx, f_attn_out_w, f_out_flat);
        ggml_tensor* f_x_resid1 = ggml_add(ctx, x_freq_packed, f_attn_block_out);
        

        
        // Freq FeedForward
        ggml_tensor* f_ff_norm_w = GetWeight(freq_ff_prefix + "_norm.weight");
        if (!f_ff_norm_w) { std::cerr << "Missing freq ff norm\n"; return nullptr; }
        
        ggml_tensor* f_x_resid1_norm = ggml_rms_norm(ctx, f_x_resid1, 1e-12f);
        f_x_resid1_norm = ggml_mul(ctx, f_x_resid1_norm, f_ff_norm_w);
        
        ggml_tensor* f_ff_in_w = GetWeight(freq_ff_prefix + "_in.weight");
        ggml_tensor* f_ff_in_b = GetWeight(freq_ff_prefix + "_in.bias");
        if (!f_ff_in_w || !f_ff_in_b) { std::cerr << "Missing freq ff in\n"; return nullptr; }
        
        ggml_tensor* f_ff_proj_in = ggml_mul_mat(ctx, f_ff_in_w, f_x_resid1_norm);
        f_ff_proj_in = ggml_add(ctx, f_ff_proj_in, f_ff_in_b);
        
        ggml_tensor* f_gelu_out = ggml_gelu_erf(ctx, f_ff_proj_in);
        

        
        ggml_tensor* f_ff_out_w = GetWeight(freq_ff_prefix + "_out.weight");
        ggml_tensor* f_ff_out_b = GetWeight(freq_ff_prefix + "_out.bias");
        if (!f_ff_out_w || !f_ff_out_b) { std::cerr << "Missing freq ff out\n"; return nullptr; }
        
        ggml_tensor* f_ff_block_out = ggml_mul_mat(ctx, f_ff_out_w, f_gelu_out);
        f_ff_block_out = ggml_add(ctx, f_ff_block_out, f_ff_out_b);
        
        x_freq_packed = ggml_add(ctx, f_x_resid1, f_ff_block_out);
        
        // Freq Transformer Final Norm
        // Only if transformer_norm_output_ is true (MelBand)
        if (transformer_norm_output_) {
            std::string freq_norm_name = "blk." + std::to_string(layer) + ".freq_norm.weight";
            ggml_tensor* freq_norm_w = GetWeight(freq_norm_name);
            if (!freq_norm_w) { std::cerr << "Missing: " << freq_norm_name << "\n"; return nullptr; }
            
            x_freq_packed = ggml_rms_norm(ctx, x_freq_packed, 1e-12f);
            x_freq_packed = ggml_mul(ctx, x_freq_packed, freq_norm_w);
        }
        
        x = ggml_reshape_4d(ctx, x_freq_packed, D, F, T, B);
        
        if (skip_connection_) {
            skip_outputs.push_back(x);
        }
    }
    
    // Global Final Norm (BS Roformer only)
    if (has_final_norm_) {
        ggml_tensor* final_norm_w = GetWeight("final_norm.weight");
        if (!final_norm_w) { std::cerr << "Missing: final_norm.weight\n"; return nullptr; }
        x = ggml_rms_norm(ctx, x, 1e-12f);
        x = ggml_mul(ctx, x, final_norm_w);
    }
    
    return x;
}

ggml_tensor* BSRoformer::BuildMaskEstimatorGraph(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_cgraph* gf,
    int n_frames,
    int batch
) {
    // Following test_10_full_model.cpp lines 532-618 EXACTLY
    // Input shape: [dim, num_bands, n_frames, batch]
    // Output: [total_out_dim, num_stems, n_frames, batch]
    
    const int DIM = dim_;
    const int NUM_BANDS = num_bands_;
    const int NUM_STEMS = num_stems_;
    
    // Calculate band_out_dims from mask_est.0.freq.{b}.mlp.4.weight shape
    // Calculate band_out_dims from last MLP weight
    std::vector<int> band_out_dims(NUM_BANDS);
    int total_out_dim = 0;
    
    // Last MLP layer index is (mlp_num_layers_ - 1) * 2
    int last_mlp_idx = (mlp_num_layers_ - 1) * 2;

    for (int b = 0; b < NUM_BANDS; ++b) {
        // mask_est.0.freq.{b}.mlp.{last}.weight
        std::string w_last_name = "mask_est.0.freq." + std::to_string(b) + ".mlp." + std::to_string(last_mlp_idx) + ".weight";
        ggml_tensor* w_last = GetWeight(w_last_name);
        if (!w_last) {
            std::cerr << "Missing weight for dim check: " << w_last_name << std::endl;
            return nullptr;
        }
        band_out_dims[b] = static_cast<int>(w_last->ne[1]) / 2;  // GLU halves the dimension
        total_out_dim += band_out_dims[b];
    }
    
    ggml_tensor* x = input;  // [D, F, T, B]
    
    std::vector<ggml_tensor*> stem_outputs;
    stem_outputs.reserve(NUM_STEMS);
    
    for (int s = 0; s < NUM_STEMS; ++s) {
        std::vector<ggml_tensor*> band_outputs;
        band_outputs.reserve(NUM_BANDS);
        
        for (int b = 0; b < NUM_BANDS; ++b) {
            // Extract band input: [DIM, n_frames, batch] for this band
            // Since input is same for all stems, we could cache this view? 
            // GGML graph deduplication might handle it, but explicit view is fine.
            ggml_tensor* band_in = ggml_view_3d(ctx, x,
                                                DIM, n_frames, batch,
                                                x->nb[2], x->nb[3],
                                                b * x->nb[1]);
            
            // mask_est.{s}.freq.{b}.mlp...
            std::string prefix = "mask_est." + std::to_string(s) + ".freq." + std::to_string(b) + ".mlp.";
            
            // MLP Layer 0
            // Dynamic MLP Construction
            ggml_tensor* mlp_current = band_in;

            for (int layer_idx = 0; layer_idx < mlp_num_layers_; ++layer_idx) {
                int seq_idx = layer_idx * 2; // 0, 2, 4...
                
                std::string w_name = prefix + std::to_string(seq_idx) + ".weight";
                std::string b_name = prefix + std::to_string(seq_idx) + ".bias";
                
                ggml_tensor* w = GetWeight(w_name);
                ggml_tensor* b = GetWeight(b_name);
                
                if (!w || !b) {
                    std::cerr << "Missing mask weights s=" << s << " b=" << b << " l=" << seq_idx << "\n";
                    return nullptr;
                }
                
                mlp_current = ggml_mul_mat(ctx, w, mlp_current);
                mlp_current = ggml_add(ctx, mlp_current, b);
                
                // Activation (Tanh) for all but last layer
                if (layer_idx < mlp_num_layers_ - 1) {
                    mlp_current = ggml_tanh(ctx, mlp_current);
                }
            }
            
            // GLU
            int dim_out = band_out_dims[b];
            
            ggml_tensor* glu_a = ggml_view_3d(ctx, mlp_current,
                                              dim_out, n_frames, batch,
                                              mlp_current->nb[1], mlp_current->nb[2], 0);
            ggml_tensor* glu_b = ggml_view_3d(ctx, mlp_current,
                                              dim_out, n_frames, batch,
                                              mlp_current->nb[1], mlp_current->nb[2],
                                              dim_out * sizeof(float));
            SetTensorName(glu_a, "mask.glu.a");
            SetTensorName(glu_b, "mask.glu.b");

            // Rejected: splitting the final GLU projection into separate A/B matmuls removed
            // this CONT, but changed the q8 CUDA output hash and increased nodes/latency
            // (2104->2290 nodes, 3.713s->3.762s on test_segment.wav).
            glu_b = ggml_cont(ctx, glu_b);
            SetTensorName(glu_b, "mask.glu.b.sig.cont");
            
            ggml_tensor* glu_b_sig = ggml_sigmoid(ctx, glu_b);
            ggml_tensor* band_out = ggml_mul(ctx, glu_a, glu_b_sig);
            SetTensorName(glu_b_sig, "mask.glu.b.sig");
            SetTensorName(band_out, "mask.glu.mul");
            
            ggml_tensor* band_out_4d = ggml_reshape_4d(ctx, band_out, dim_out, 1, n_frames, batch);
            band_outputs.push_back(band_out_4d);
        }

        ggml_tensor* stem_output = ConcatBalanced(ctx, band_outputs, 0);
        stem_outputs.push_back(stem_output);
    }

    ggml_tensor* mask_output = ConcatBalanced(ctx, stem_outputs, 1);
    ggml_set_output(mask_output);
    ggml_build_forward_expand(gf, mask_output);
    return mask_output;
}
