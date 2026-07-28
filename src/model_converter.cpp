#include "model_converter.h"

#include <ggml.h>
#include <gguf.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <cstdio>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <regex>
#include <filesystem>
#include <cstring>
#include <cmath>

namespace fs = std::filesystem;

namespace qwen3_asr {

// ============================================================
// Safetensors reading
// ============================================================

static std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

static uint64_t read_u64_le(const uint8_t* p) {
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

bool ModelConverter::read_safetensor_header(
    const std::string& path,
    std::map<std::string, TensorInfo>& tensors,
    uint64_t& data_offset)
{
    auto buf = read_file_bytes(path);
    if (buf.empty()) {
        std::cerr << "error: cannot read " << path << std::endl;
        return false;
    }

    uint64_t header_len = read_u64_le(buf.data());
    if (header_len + 8 > buf.size()) {
        std::cerr << "error: invalid safetensors header in " << path << std::endl;
        return false;
    }

    std::string header_str(reinterpret_cast<char*>(buf.data() + 8), header_len);
    json j;
    try {
        j = json::parse(header_str);
    } catch (...) {
        std::cerr << "error: invalid JSON header in " << path << std::endl;
        return false;
    }

    for (auto& [name, info] : j.items()) {
        if (name == "__metadata__") continue;
        TensorInfo ti;
        ti.dtype = info["dtype"].get<std::string>();
        auto shape = info["shape"];
        for (auto& s : shape) {
            ti.shape.push_back(s.get<int64_t>());
        }
        auto offsets = info["data_offsets"];
        uint64_t start = offsets[0].get<uint64_t>();
        uint64_t end   = offsets[1].get<uint64_t>();
        ti.offset = start;
        ti.size   = end - start;
        tensors[name] = std::move(ti);
    }

    data_offset = 8 + header_len;
    return true;
}

// ============================================================
// BFloat16 conversion helpers
// ============================================================

static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// Use ggml's built-in f32->f16 for bit-exact results

int ModelConverter::convert_bf16_to_f16(const uint16_t* src, void* dst, int64_t n) {
    auto* out = static_cast<uint16_t*>(dst);
    for (int64_t i = 0; i < n; i++) {
        float f = bf16_to_f32(src[i]);
        out[i] = ggml_fp32_to_fp16(f);
    }
    return 0;
}

int ModelConverter::convert_bf16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = bf16_to_f32(src[i]);
    }
    return 0;
}

// ============================================================
// Tensor dtype size
// ============================================================

int ModelConverter::tensor_dtype_size(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16") return 2;
    if (dtype == "I8") return 1;
    if (dtype == "F64") return 8;
    return 4;
}

// ============================================================
// Tensor name mapping
// ============================================================

bool ModelConverter::matches(const std::string& name, const std::string& pattern) {
    try {
        std::regex re(pattern);
        return std::regex_match(name, re);
    } catch (...) {
        return false;
    }
}

std::string ModelConverter::map_tensor_name(const std::string& hf_name) {
    // Direct mappings
    static const std::unordered_map<std::string, std::string> direct = {
        {"thinker.audio_tower.conv2d1.weight", "audio.encoder.conv1.weight"},
        {"thinker.audio_tower.conv2d1.bias",   "audio.encoder.conv1.bias"},
        {"thinker.audio_tower.conv2d2.weight", "audio.encoder.conv2.weight"},
        {"thinker.audio_tower.conv2d2.bias",   "audio.encoder.conv2.bias"},
        {"thinker.audio_tower.conv2d3.weight", "audio.encoder.conv3.weight"},
        {"thinker.audio_tower.conv2d3.bias",   "audio.encoder.conv3.bias"},
        {"thinker.audio_tower.conv_out.weight", "audio.encoder.conv_out.weight"},
        {"thinker.audio_tower.conv_out.bias",   "audio.encoder.conv_out.bias"},
        {"thinker.audio_tower.layer_norm.weight", "audio.encoder.ln.weight"},
        {"thinker.audio_tower.layer_norm.bias",   "audio.encoder.ln.bias"},
        {"thinker.audio_tower.ln_post.weight", "audio.encoder.ln_post.weight"},
        {"thinker.audio_tower.ln_post.bias",   "audio.encoder.ln_post.bias"},
        {"thinker.audio_tower.embed_positions.weight", "audio.encoder.pos_embd.weight"},
        {"thinker.audio_tower.proj1.weight", "audio.encoder.proj1.weight"},
        {"thinker.audio_tower.proj1.bias",   "audio.encoder.proj1.bias"},
        {"thinker.audio_tower.proj2.weight", "audio.encoder.proj2.weight"},
        {"thinker.audio_tower.proj2.bias",   "audio.encoder.proj2.bias"},
        {"thinker.model.embed_tokens.weight", "token_embd.weight"},
        {"thinker.model.norm.weight",         "output_norm.weight"},
        {"thinker.lm_head.weight",            "output.weight"},
        {"thinker.classify_head.weight",      "classify_head.weight"},
        {"thinker.classify_head.bias",        "classify_head.bias"},
    };

    auto it = direct.find(hf_name);
    if (it != direct.end()) return it->second;

    // Audio encoder layer patterns
    static const std::vector<std::pair<std::regex, std::string>> audio_patterns = {
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.weight)"),      "audio.encoder.blk.{}.attn_q.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight)"),      "audio.encoder.blk.{}.attn_k.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.weight)"),      "audio.encoder.blk.{}.attn_v.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.weight)"),    "audio.encoder.blk.{}.attn_out.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.bias)"),        "audio.encoder.blk.{}.attn_q.bias"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.bias)"),        "audio.encoder.blk.{}.attn_k.bias"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.bias)"),        "audio.encoder.blk.{}.attn_v.bias"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.bias)"),      "audio.encoder.blk.{}.attn_out.bias"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.weight)"),   "audio.encoder.blk.{}.attn_norm.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.bias)"),     "audio.encoder.blk.{}.attn_norm.bias"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.final_layer_norm\.weight)"),       "audio.encoder.blk.{}.ffn_norm.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.final_layer_norm\.bias)"),         "audio.encoder.blk.{}.ffn_norm.bias"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.fc1\.weight)"),                    "audio.encoder.blk.{}.ffn_up.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.fc1\.bias)"),                      "audio.encoder.blk.{}.ffn_up.bias"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.fc2\.weight)"),                    "audio.encoder.blk.{}.ffn_down.weight"},
        {std::regex(R"(thinker\.audio_tower\.layers\.(\d+)\.fc2\.bias)"),                      "audio.encoder.blk.{}.ffn_down.bias"},
    };

    // Text decoder layer patterns
    static const std::vector<std::pair<std::regex, std::string>> text_patterns = {
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.input_layernorm\.weight)"),            "blk.{}.attn_norm.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight)"),          "blk.{}.attn_q.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight)"),          "blk.{}.attn_k.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight)"),          "blk.{}.attn_v.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight)"),          "blk.{}.attn_output.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight)"),          "blk.{}.attn_q_norm.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight)"),          "blk.{}.attn_k_norm.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight)"),   "blk.{}.ffn_norm.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight)"),             "blk.{}.ffn_gate.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight)"),               "blk.{}.ffn_up.weight"},
        {std::regex(R"(thinker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight)"),             "blk.{}.ffn_down.weight"},
    };

    std::smatch m;

    for (const auto& [re, tmpl] : audio_patterns) {
        if (std::regex_match(hf_name, m, re)) {
            auto s = tmpl;
            auto pos = s.find("{}");
            if (pos != std::string::npos) {
                s.replace(pos, 2, m[1].str());
            }
            return s;
        }
    }

    for (const auto& [re, tmpl] : text_patterns) {
        if (std::regex_match(hf_name, m, re)) {
            auto s = tmpl;
            auto pos = s.find("{}");
            if (pos != std::string::npos) {
                s.replace(pos, 2, m[1].str());
            }
            return s;
        }
    }

    return {};
}

// ============================================================
// Config loading
// ============================================================

bool ModelConverter::load_config() {
    fs::path config_path = fs::path(config_.input_dir) / "config.json";
    auto buf = read_file_bytes(config_path.string());
    if (buf.empty()) {
        std::cerr << "error: config.json not found in " << config_.input_dir << std::endl;
        return false;
    }

    json j;
    try {
        j = json::parse(buf);
    } catch (...) {
        std::cerr << "error: invalid config.json" << std::endl;
        return false;
    }

    auto tc = j["thinker_config"];
    if (tc.is_null()) {
        std::cerr << "error: config.json missing thinker_config" << std::endl;
        return false;
    }

    auto ac = tc["audio_config"];
    auto txtc = tc["text_config"];

    model_.audio_encoder_layers      = ac.value("encoder_layers", ac.value("num_hidden_layers", 18));
    model_.audio_d_model             = ac.value("d_model", 896);
    model_.audio_attention_heads     = ac.value("encoder_attention_heads", 14);
    model_.audio_ffn_dim             = ac.value("encoder_ffn_dim", 3584);
    model_.audio_num_mel_bins        = ac.value("num_mel_bins", 128);
    model_.audio_downsample_hidden_size = ac.value("downsample_hidden_size", 480);

    model_.text_decoder_layers       = txtc.value("num_hidden_layers", 28);
    model_.text_hidden_size          = txtc.value("hidden_size", 1024);
    model_.text_attention_heads      = txtc.value("num_attention_heads", 16);
    model_.text_kv_heads             = txtc.value("num_key_value_heads", 8);
    model_.text_intermediate_size    = txtc.value("intermediate_size", 3072);
    model_.text_rope_theta           = txtc.value("rope_theta", 1000000.0f);
    model_.text_rms_norm_eps         = txtc.value("rms_norm_eps", 1e-6f);
    model_.text_head_dim             = txtc.value("head_dim", 128);
    model_.vocab_size                = txtc.value("vocab_size", 151936);

    model_.audio_start_token_id      = tc.value("audio_start_token_id", 151669);
    model_.audio_end_token_id        = tc.value("audio_end_token_id", 151670);
    model_.audio_pad_token_id        = tc.value("audio_token_id", 151676);

    std::string model_type = tc.value("model_type", "");
    model_.is_forced_aligner = (model_type == "qwen3_forced_aligner");

    if (model_.is_forced_aligner) {
        model_.classify_num         = tc.value("classify_num", 5000);
        model_.timestamp_token_id   = tc.value("timestamp_token_id", 151705);
    }

    // Determine model name
    uint64_t total_size = 0;
    for (auto& entry : fs::recursive_directory_iterator(config_.input_dir)) {
        if (entry.is_regular_file()) total_size += entry.file_size();
    }

    if (model_.is_forced_aligner) {
        model_name_ = "Qwen3-ForcedAligner-0.6B";
    } else if (total_size > 3ULL * 1024 * 1024 * 1024) {
        model_name_ = "Qwen3-ASR-1.7B";
    } else {
        model_name_ = "Qwen3-ASR-0.6B";
    }

    return true;
}

// ============================================================
// Tokenizer loading
// ============================================================

bool ModelConverter::add_tokenizer(struct gguf_context* gctx) {
    fs::path dir(config_.input_dir);
    auto vocab_path = dir / "vocab.json";
    auto merges_path = dir / "merges.txt";
    auto tok_config_path = dir / "tokenizer_config.json";

    // Load vocab.json
    if (!fs::exists(vocab_path)) {
        std::cerr << "error: vocab.json not found" << std::endl;
        return false;
    }
    auto vocab_buf = read_file_bytes(vocab_path.string());
    json vocab_json;
    try { vocab_json = json::parse(vocab_buf); }
    catch (...) { std::cerr << "error: invalid vocab.json" << std::endl; return false; }

    // Load tokenizer_config.json
    json tok_config;
    if (fs::exists(tok_config_path)) {
        auto buf = read_file_bytes(tok_config_path.string());
        try { tok_config = json::parse(buf); } catch (...) {}
    }

    // Build token arrays
    int max_id = 0;
    for (auto& [tok, id] : vocab_json.items()) {
        if (id.get<int>() > max_id) max_id = id.get<int>();
    }

    int token_count = std::max(max_id + 1, model_.vocab_size);

    // Load added tokens
    json added_tokens;
    if (tok_config.contains("added_tokens_decoder")) {
        added_tokens = tok_config["added_tokens_decoder"];
        for (auto& [id_str, info] : added_tokens.items()) {
            int tid = std::stoi(id_str);
            if (tid >= token_count) token_count = tid + 1;
        }
    }

    std::vector<std::string> tokens(token_count);
    std::vector<int32_t> toktypes(token_count, 0); // UNUSED

    for (auto& [tok, id] : vocab_json.items()) {
        int idx = id.get<int>();
        if (idx >= 0 && idx < token_count) {
            tokens[idx] = tok;
            toktypes[idx] = 1; // NORMAL
        }
    }

    for (auto& [id_str, info] : added_tokens.items()) {
        int tid = std::stoi(id_str);
        std::string content = info.value("content", "");
        bool is_special = info.value("special", false);
        if (tid >= 0 && tid < token_count) {
            tokens[tid] = content;
            toktypes[tid] = is_special ? 3 : 1; // CONTROL or NORMAL
        }
    }

    // Fill empty slots
    for (int i = 0; i < token_count; i++) {
        if (tokens[i].empty()) {
            tokens[i] = "[PAD" + std::to_string(i) + "]";
        }
    }

    // Load merges
    std::vector<std::string> merges;
    if (fs::exists(merges_path)) {
        std::ifstream f(merges_path);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line[0] != '#') {
                merges.push_back(line);
            }
        }
    }

    // Write tokenizer to GGUF
    gguf_set_val_str(gctx, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(gctx, "tokenizer.ggml.pre", "qwen2");

    // Token list
    std::vector<const char*> token_ptrs(token_count);
    for (int i = 0; i < token_count; i++) {
        token_ptrs[i] = tokens[i].c_str();
    }
    gguf_set_arr_str(gctx, "tokenizer.ggml.tokens", token_ptrs.data(), token_count);

    // Token types
    gguf_set_arr_data(gctx, "tokenizer.ggml.token_type",
                      GGUF_TYPE_INT32, toktypes.data(), token_count);

    // Merges
    if (!merges.empty()) {
        std::vector<const char*> merge_ptrs(merges.size());
        for (size_t i = 0; i < merges.size(); i++) {
            merge_ptrs[i] = merges[i].c_str();
        }
        gguf_set_arr_str(gctx, "tokenizer.ggml.merges", merge_ptrs.data(), merges.size());
    }

    // Build a reverse vocab lookup including added tokens
    std::map<std::string, int> full_vocab;
    for (auto& [tok, id] : vocab_json.items()) full_vocab[tok] = id.get<int>();
    if (tok_config.contains("added_tokens_decoder")) {
        for (auto& [id_str, info] : tok_config["added_tokens_decoder"].items()) {
            std::string content = info.value("content", "");
            if (!content.empty()) full_vocab[content] = std::stoi(id_str);
        }
    }

    // Special tokens
    if (tok_config.contains("eos_token")) {
        auto& eos = tok_config["eos_token"];
        std::string content = eos.is_string() ? eos.get<std::string>() : eos.value("content", "");
        if (!content.empty() && full_vocab.count(content)) {
            gguf_set_val_u32(gctx, "tokenizer.ggml.eos_token_id", full_vocab[content]);
        }
    }

    if (tok_config.contains("pad_token")) {
        auto& pad = tok_config["pad_token"];
        std::string content = pad.is_string() ? pad.get<std::string>() : pad.value("content", "");
        if (!content.empty() && full_vocab.count(content)) {
            gguf_set_val_u32(gctx, "tokenizer.ggml.padding_token_id", full_vocab[content]);
        }
    }

    if (tok_config.contains("chat_template")) {
        gguf_set_val_str(gctx, "tokenizer.ggml.chat_template",
                         tok_config["chat_template"].get<std::string>().c_str());
    }

    return true;
}

// ============================================================
// Tensor processing
// ============================================================

static ggml_type output_ggml_type(const std::string& output_type_str) {
    if (output_type_str == "f32")  return GGML_TYPE_F32;
    if (output_type_str == "f16")  return GGML_TYPE_F16;
    if (output_type_str == "q8_0") return GGML_TYPE_Q8_0;
    if (output_type_str == "q4_1") return GGML_TYPE_Q4_1;
    return GGML_TYPE_F16;
}

// Decide per-tensor output type
static ggml_type tensor_output_type(const std::string& ggml_name,
                                     const std::string& output_type_str,
                                     int n_dims)
{
    // 1D tensors (norms, biases) -> always F32
    if (n_dims <= 1) return GGML_TYPE_F32;

    ggml_type base = output_ggml_type(output_type_str);

    if (base == GGML_TYPE_F32) return GGML_TYPE_F32;

    // Keep embeddings/layer norms in F16 (not quantized)
    if (base != GGML_TYPE_F16) {
        if (ggml_name.find("token_embd") != std::string::npos ||
            ggml_name.find("output.weight") != std::string::npos ||
            ggml_name.find("pos_embd") != std::string::npos ||
            ggml_name.find("_norm") != std::string::npos ||
            ggml_name.find(".ln") != std::string::npos ||
            ggml_name.find("ln_") != std::string::npos ||
            ggml_name.find(".bias") != std::string::npos)
        {
            return GGML_TYPE_F16;
        }
    }

    return base;
}

// Quantize f32 data to the target type (Q8_0, Q4_1, etc.)
// Returns the quantized data size in bytes, or 0 on failure.
// Caller must free the returned buffer.
static size_t quantize_f32(const float* src, int64_t nrows, int64_t n_per_row,
                            ggml_type dst_type, void** dst_out)
{
    // nrows * n_per_row = total elements
    // Calculate max output size (quantized is always <= f32 size)
    size_t max_size = nrows * n_per_row * sizeof(float);
    auto* dst = malloc(max_size);
    if (!dst) return 0;

    // We use the imatrix=nullptr variant
    size_t total_size = 0;
    for (int64_t row = 0; row < nrows; row++) {
        const float* row_src = src + row * n_per_row;
        void* row_dst = (uint8_t*)dst + total_size;

        // ggml_quantize_chunk processes in chunks. For simplicity,
        // we call it for each row.
        // start = 0 means process from the beginning
        size_t sz = ggml_quantize_chunk(dst_type, row_src, row_dst,
                                        0, 1, n_per_row, nullptr);
        if (sz == 0) {
            free(dst);
            return 0;
        }
        total_size += sz;
    }

    *dst_out = dst;
    return total_size;
}

ModelConverter::ModelConverter(const ConverterConfig& config)
    : config_(config)
{
}

ModelConverter::~ModelConverter() {
    if (ggml_ctx_) ggml_free(ggml_ctx_);
    if (ggml_mem_) free(ggml_mem_);
}

bool ModelConverter::register_tensors(struct gguf_context* gctx) {
    safetensor_files_.clear();
    for (auto& entry : fs::directory_iterator(config_.input_dir)) {
        if (entry.path().extension() == ".safetensors") {
            safetensor_files_.push_back(entry.path().string());
        }
    }
    std::sort(safetensor_files_.begin(), safetensor_files_.end());
    if (safetensor_files_.empty()) {
        std::cerr << "error: no .safetensors files found" << std::endl;
        return false;
    }

    registry_.clear();
    for (size_t fi = 0; fi < safetensor_files_.size(); fi++) {
        std::map<std::string, TensorInfo> tensors;
        uint64_t data_offset;
        if (!read_safetensor_header(safetensor_files_[fi], tensors, data_offset)) return false;
        for (auto& [name, ti] : tensors) {
            TensorMeta tm;
            tm.file_idx = (int)fi;
            tm.hf_dtype = ti.dtype;
            tm.shape = ti.shape;
            tm.offset = data_offset + ti.offset;
            tm.size   = ti.size;
            registry_[name] = std::move(tm);
        }
    }

    size_t ggml_mem_size = registry_.size() * 512 + 1024;
    ggml_mem_ = malloc(ggml_mem_size);
    if (!ggml_mem_) return false;

    struct ggml_init_params params = { ggml_mem_size, ggml_mem_, true };
    ggml_ctx_ = ggml_init(params);

    for (auto& [hf_name, meta] : registry_) {
        auto ggml_name = map_tensor_name(hf_name);
        if (ggml_name.empty()) {
            std::cerr << "warning: skipping " << hf_name << std::endl;
            continue;
        }
        int n_dims = (int)meta.shape.size();
        int64_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) ne[i] = meta.shape[n_dims - 1 - i];

        ggml_type out_type = tensor_output_type(ggml_name, config_.output_type, n_dims);
        struct ggml_tensor* t = nullptr;
        switch (n_dims) {
            case 1: t = ggml_new_tensor_1d(ggml_ctx_, out_type, ne[0]); break;
            case 2: t = ggml_new_tensor_2d(ggml_ctx_, out_type, ne[0], ne[1]); break;
            case 3: t = ggml_new_tensor_3d(ggml_ctx_, out_type, ne[0], ne[1], ne[2]); break;
            case 4: t = ggml_new_tensor_4d(ggml_ctx_, out_type, ne[0], ne[1], ne[2], ne[3]); break;
            default: continue;
        }
        ggml_set_name(t, ggml_name.c_str());
        gguf_add_tensor(gctx, t);
    }
    return true;
}

// ============================================================
// Main convert method
// ============================================================

bool ModelConverter::convert() {
    std::cout << "loading config from " << config_.input_dir << std::endl;
    if (!load_config()) return false;
    std::cout << "model: " << model_name_ << std::endl;

    fs::path out_path(config_.output_path);
    fs::create_directories(out_path.parent_path());

    // 3. Initialize GGUF context
    struct gguf_context* gctx = gguf_init_empty();
    if (!gctx) {
        std::cerr << "error: failed to create GGUF context" << std::endl;
        return false;
    }

    // 4. Add metadata
    // Architecture
    gguf_set_val_str(gctx, "general.architecture", "qwen3-asr");
    gguf_set_val_str(gctx, "general.name", model_name_.c_str());
    int ftype = config_.output_type == "f32" ? 0 :
                config_.output_type == "f16" ? 1 :
                config_.output_type == "q8_0" ? 7 : 3;
    gguf_set_val_u32(gctx, "general.file_type", ftype);
    gguf_set_val_u32(gctx, "general.quantization_version", 2);

    // Text decoder params
    gguf_set_val_u32(gctx, "qwen3-asr.block_count", model_.text_decoder_layers);
    gguf_set_val_u32(gctx, "qwen3-asr.embedding_length", model_.text_hidden_size);
    gguf_set_val_u32(gctx, "qwen3-asr.feed_forward_length", model_.text_intermediate_size);
    gguf_set_val_u32(gctx, "qwen3-asr.attention.head_count", model_.text_attention_heads);
    gguf_set_val_u32(gctx, "qwen3-asr.attention.head_count_kv", model_.text_kv_heads);
    gguf_set_val_u32(gctx, "qwen3-asr.attention.key_length", model_.text_head_dim);
    gguf_set_val_u32(gctx, "qwen3-asr.attention.value_length", model_.text_head_dim);
    gguf_set_val_f32(gctx, "qwen3-asr.rope.freq_base", model_.text_rope_theta);
    gguf_set_val_f32(gctx, "qwen3-asr.attention.layer_norm_rms_epsilon", model_.text_rms_norm_eps);
    gguf_set_val_u32(gctx, "qwen3-asr.vocab_size", model_.vocab_size);

    // Audio encoder params
    gguf_set_val_u32(gctx, "qwen3-asr.audio.encoder.layer_count", model_.audio_encoder_layers);
    gguf_set_val_u32(gctx, "qwen3-asr.audio.encoder.embedding_length", model_.audio_d_model);
    gguf_set_val_u32(gctx, "qwen3-asr.audio.encoder.attention.head_count", model_.audio_attention_heads);
    gguf_set_val_u32(gctx, "qwen3-asr.audio.encoder.feed_forward_length", model_.audio_ffn_dim);
    gguf_set_val_u32(gctx, "qwen3-asr.audio.num_mel_bins", model_.audio_num_mel_bins);
    gguf_set_val_u32(gctx, "qwen3-asr.audio.conv_channels", model_.audio_downsample_hidden_size);

    // Special tokens
    gguf_set_val_u32(gctx, "qwen3-asr.audio.start_token_id", model_.audio_start_token_id);
    gguf_set_val_u32(gctx, "qwen3-asr.audio.end_token_id", model_.audio_end_token_id);
    gguf_set_val_u32(gctx, "qwen3-asr.audio.pad_token_id", model_.audio_pad_token_id);

    // Forced Aligner specific
    if (model_.is_forced_aligner) {
        gguf_set_val_u32(gctx, "qwen3-asr.classify_num", model_.classify_num);
        gguf_set_val_u32(gctx, "qwen3-asr.timestamp_token_id", model_.timestamp_token_id);
        gguf_set_val_u32(gctx, "qwen3-asr.timestamp_segment_time", 80);
    }

    // 5. Add tokenizer
    std::cout << "loading tokenizer..." << std::endl;
    if (!add_tokenizer(gctx)) {
        gguf_free(gctx);
        return false;
    }

    // 6. Register tensors
    std::cout << "registering tensors..." << std::endl;
    if (!register_tensors(gctx)) {
        gguf_free(gctx);
        return false;
    }

    // 7. Write metadata (header, KV pairs, tensor info) - no tensor data
    std::cout << "writing GGUF metadata to " << config_.output_path << std::endl;
    bool ok = gguf_write_to_file(gctx, config_.output_path.c_str(), true);
    if (!ok) {
        std::cerr << "error: failed to write GGUF metadata" << std::endl;
        gguf_free(gctx);
        return false;
    }

    // 8. Append tensor data directly to the file (two-pass approach)
    std::cout << "writing tensor data..." << std::endl;
    FILE* f = fopen(config_.output_path.c_str(), "ab");
    if (!f) {
        std::cerr << "error: cannot open output file" << std::endl;
        gguf_free(gctx);
        return false;
    }

    int converted = 0;
    for (auto& [hf_name, meta] : registry_) {
        auto ggml_name = map_tensor_name(hf_name);
        if (ggml_name.empty()) continue;

        int tid = gguf_find_tensor(gctx, ggml_name.c_str());
        if (tid < 0) continue;

        size_t tensor_size = gguf_get_tensor_size(gctx, tid);

        auto& filepath = safetensor_files_[meta.file_idx];
        std::ifstream sf(filepath, std::ios::binary);
        if (!sf) {
            std::cerr << "error: cannot open " << filepath << std::endl;
            fclose(f);
            gguf_free(gctx);
            return false;
        }

        std::vector<uint8_t> raw_data(meta.size);
        sf.seekg(meta.offset);
        sf.read(reinterpret_cast<char*>(raw_data.data()), meta.size);
        sf.close();

        // Get dimensions and output type
        int n_dims = (int)meta.shape.size();
        int64_t total_elements = 1;
        for (auto d : meta.shape) total_elements *= d;

        ggml_type out_type = tensor_output_type(ggml_name, config_.output_type, n_dims);

        // Convert data to output type
        void* data_to_write = raw_data.data();
        std::vector<uint16_t> bf16_to_f16_buf;
        std::vector<float> bf16_to_f32_buf;
        std::vector<uint16_t> f32_to_f16_buf;
        std::vector<uint8_t> quant_buf;

        if (meta.hf_dtype == "BF16") {
            if (out_type == GGML_TYPE_F16) {
                bf16_to_f16_buf.resize(total_elements);
                convert_bf16_to_f16(reinterpret_cast<const uint16_t*>(raw_data.data()),
                                    bf16_to_f16_buf.data(), total_elements);
                data_to_write = bf16_to_f16_buf.data();
            } else {
                bf16_to_f32_buf.resize(total_elements);
                convert_bf16_to_f32(reinterpret_cast<const uint16_t*>(raw_data.data()),
                                    bf16_to_f32_buf.data(), total_elements);
                if (out_type == GGML_TYPE_F32) {
                    data_to_write = bf16_to_f32_buf.data();
                } else {
                    void* qdata = nullptr;
                    int64_t n_per_row = meta.shape[0];
                    int64_t nrows = total_elements / n_per_row;
                    size_t qsize = quantize_f32(bf16_to_f32_buf.data(), nrows, n_per_row,
                                                out_type, &qdata);
                    if (qsize == 0) {
                        std::cerr << "error: quantization failed for " << ggml_name << std::endl;
                        fclose(f); gguf_free(gctx); return false;
                    }
                    quant_buf.resize(qsize);
                    memcpy(quant_buf.data(), qdata, qsize);
                    free(qdata);
                    data_to_write = quant_buf.data();
                }
            }
        } else if (meta.hf_dtype == "F32" && out_type == GGML_TYPE_F16) {
            const float* src = reinterpret_cast<const float*>(raw_data.data());
            f32_to_f16_buf.resize(total_elements);
            for (int64_t i = 0; i < total_elements; i++) {
                f32_to_f16_buf[i] = ggml_fp32_to_fp16(src[i]);
            }
            data_to_write = f32_to_f16_buf.data();
        } else if (meta.hf_dtype == "F32" && out_type != GGML_TYPE_F32) {
            const float* src = reinterpret_cast<const float*>(raw_data.data());
            void* qdata = nullptr;
            int64_t n_per_row = meta.shape[0];
            int64_t nrows = total_elements / n_per_row;
            size_t qsize = quantize_f32(src, nrows, n_per_row, out_type, &qdata);
            if (qsize == 0) {
                std::cerr << "error: quantization failed for " << ggml_name << std::endl;
                fclose(f); gguf_free(gctx); return false;
            }
            quant_buf.resize(qsize);
            memcpy(quant_buf.data(), qdata, qsize);
            free(qdata);
            data_to_write = quant_buf.data();
        } else if (meta.hf_dtype == "F16" && out_type == GGML_TYPE_F32) {
            const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(raw_data.data());
            bf16_to_f32_buf.resize(total_elements);
            ggml_fp16_to_fp32_row(src, bf16_to_f32_buf.data(), total_elements);
            data_to_write = bf16_to_f32_buf.data();
        } else if (meta.hf_dtype == "F16" && out_type != GGML_TYPE_F16) {
            const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(raw_data.data());
            bf16_to_f32_buf.resize(total_elements);
            ggml_fp16_to_fp32_row(src, bf16_to_f32_buf.data(), total_elements);
            void* qdata = nullptr;
            int64_t n_per_row = meta.shape[0];
            int64_t nrows = total_elements / n_per_row;
            size_t qsize = quantize_f32(bf16_to_f32_buf.data(), nrows, n_per_row, out_type, &qdata);
            if (qsize == 0) {
                std::cerr << "error: quantization failed for " << ggml_name << std::endl;
                fclose(f); gguf_free(gctx); return false;
            }
            quant_buf.resize(qsize);
            memcpy(quant_buf.data(), qdata, qsize);
            free(qdata);
            data_to_write = quant_buf.data();
        } // else: same type, no conversion

        // Write to file at the correct offset
        size_t data_offset = gguf_get_data_offset(gctx);
        size_t tensor_offset = gguf_get_tensor_offset(gctx, tid);
        fseek(f, data_offset + tensor_offset, SEEK_SET);
        fwrite(data_to_write, 1, tensor_size, f);
        converted++;
    }

    fclose(f);
    gguf_free(gctx);
    std::cout << "converted " << converted << " tensors" << std::endl;
    std::cout << "conversion complete!" << std::endl;
    return true;
}

} // namespace qwen3_asr
