#pragma once

#include <ggml.h>
#include <gguf.h>

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace qwen3_asr {

struct ModelConfig {
    // Audio encoder
    int32_t audio_encoder_layers = 18;
    int32_t audio_d_model = 896;
    int32_t audio_attention_heads = 14;
    int32_t audio_ffn_dim = 3584;
    int32_t audio_num_mel_bins = 128;
    int32_t audio_downsample_hidden_size = 480;

    // Text decoder
    int32_t text_decoder_layers = 28;
    int32_t text_hidden_size = 1024;
    int32_t text_attention_heads = 16;
    int32_t text_kv_heads = 8;
    int32_t text_intermediate_size = 3072;
    float   text_rope_theta = 1000000.0f;
    float   text_rms_norm_eps = 1e-6f;
    int32_t text_head_dim = 128;
    int32_t vocab_size = 151936;

    // Special tokens
    int32_t audio_start_token_id = 151669;
    int32_t audio_end_token_id = 151670;
    int32_t audio_pad_token_id = 151676;

    // Forced aligner
    bool    is_forced_aligner = false;
    int32_t classify_num = 5000;
    int32_t timestamp_token_id = 151705;
};

struct ConverterConfig {
    std::string input_dir;
    std::string output_path;
    std::string output_type = "f16"; // f32, f16, q8_0, q4_1
};

class ModelConverter {
public:
    explicit ModelConverter(const ConverterConfig& config);
    ~ModelConverter();

    bool convert();

private:
    ConverterConfig config_;
    ModelConfig model_;

    // Internal state
    std::string model_name_;
    void* ggml_mem_ = nullptr;
    struct ggml_context* ggml_ctx_ = nullptr;

    bool load_config();
    bool register_tensors(struct gguf_context* gctx);
    bool write_tensor_data(struct gguf_context* gctx, FILE* f);
    bool add_tokenizer(struct gguf_context* gctx);

    // Tensor metadata
    struct TensorMeta {
        int file_idx;
        std::string hf_dtype;
        std::vector<int64_t> shape;
        uint64_t offset;
        uint64_t size;
    };
    std::vector<std::string> safetensor_files_;
    std::map<std::string, TensorMeta> registry_;

    // Tensor name mapping
    static std::string map_tensor_name(const std::string& hf_name);
    static bool matches(const std::string& name, const std::string& pattern);

    // Dtype conversion
    static int tensor_dtype_size(const std::string& dtype);
    static int convert_bf16_to_f16(const uint16_t* src, void* dst, int64_t n);
    static int convert_bf16_to_f32(const uint16_t* src, float* dst, int64_t n);

    // Safetensors reading
    struct TensorInfo {
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t offset;
        uint64_t size;
    };

    bool read_safetensor_header(const std::string& path,
                                 std::map<std::string, TensorInfo>& tensors,
                                 uint64_t& data_offset);
};

} // namespace qwen3_asr
