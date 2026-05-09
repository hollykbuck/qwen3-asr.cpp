#define QWEN3_ASR_SHARED_EXPORTS
#include "qwen3_asr_c.h"
#include "qwen3_asr.h"

struct qwen3_asr_result {
    qwen3_asr::transcribe_result cpp_result;
};

qwen3_asr_params_t qwen3_asr_default_params() {
    qwen3_asr::transcribe_params cpp_params;
    qwen3_asr_params_t params;
    params.max_tokens = cpp_params.max_tokens;
    params.language = nullptr; 
    params.n_threads = cpp_params.n_threads;
    params.print_progress = cpp_params.print_progress;
    return params;
}

qwen3_asr_context_t* qwen3_asr_create() {
    return reinterpret_cast<qwen3_asr_context_t*>(new qwen3_asr::Qwen3ASR());
}

void qwen3_asr_free(qwen3_asr_context_t* ctx) {
    if (ctx) {
        delete reinterpret_cast<qwen3_asr::Qwen3ASR*>(ctx);
    }
}

int qwen3_asr_get_device_count() {
    return ggml_backend_dev_count();
}

const char* qwen3_asr_get_device_description(int index) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(index);
    return dev ? ggml_backend_dev_description(dev) : "Invalid device index";
}

void qwen3_asr_set_gpu_device(qwen3_asr_context_t* ctx, int device_id) {
    if (ctx) {
        auto* asr = reinterpret_cast<qwen3_asr::Qwen3ASR*>(ctx);
        asr->set_gpu_device(device_id);
    }
}

bool qwen3_asr_load_model(qwen3_asr_context_t* ctx, const char* model_path) {
    if (!ctx || !model_path) return false;
    auto* asr = reinterpret_cast<qwen3_asr::Qwen3ASR*>(ctx);
    return asr->load_model(model_path);
}

static qwen3_asr::transcribe_params convert_params(qwen3_asr_params_t params) {
    qwen3_asr::transcribe_params cpp_params;
    cpp_params.max_tokens = params.max_tokens;
    cpp_params.language = params.language ? params.language : "";
    cpp_params.n_threads = params.n_threads;
    cpp_params.print_progress = params.print_progress;
    return cpp_params;
}

qwen3_asr_result_t* qwen3_asr_transcribe_file(
    qwen3_asr_context_t* ctx, 
    const char* audio_path, 
    qwen3_asr_params_t params) 
{
    if (!ctx || !audio_path) return nullptr;
    auto* asr = reinterpret_cast<qwen3_asr::Qwen3ASR*>(ctx);
    auto* res = new qwen3_asr_result();
    res->cpp_result = asr->transcribe(audio_path, convert_params(params));
    return reinterpret_cast<qwen3_asr_result_t*>(res);
}

qwen3_asr_result_t* qwen3_asr_transcribe_buffer(
    qwen3_asr_context_t* ctx, 
    const float* samples, 
    int32_t n_samples, 
    qwen3_asr_params_t params) 
{
    if (!ctx || !samples) return nullptr;
    auto* asr = reinterpret_cast<qwen3_asr::Qwen3ASR*>(ctx);
    auto* res = new qwen3_asr_result();
    res->cpp_result = asr->transcribe(samples, n_samples, convert_params(params));
    return reinterpret_cast<qwen3_asr_result_t*>(res);
}

bool qwen3_asr_result_get_success(qwen3_asr_result_t* res) {
    return res ? res->cpp_result.success : false;
}

const char* qwen3_asr_result_get_text(qwen3_asr_result_t* res) {
    return res ? res->cpp_result.text.c_str() : "";
}

const char* qwen3_asr_result_get_error(qwen3_asr_result_t* res) {
    return res ? res->cpp_result.error_msg.c_str() : "Null result";
}

int32_t qwen3_asr_result_get_token_count(qwen3_asr_result_t* res) {
    return res ? static_cast<int32_t>(res->cpp_result.tokens.size()) : 0;
}

void qwen3_asr_result_get_tokens(qwen3_asr_result_t* res, int32_t* out_tokens) {
    if (res && out_tokens) {
        for (size_t i = 0; i < res->cpp_result.tokens.size(); ++i) {
            out_tokens[i] = res->cpp_result.tokens[i];
        }
    }
}

int64_t qwen3_asr_result_get_t_load_ms(qwen3_asr_result_t* res) { return res ? res->cpp_result.t_load_ms : 0; }
int64_t qwen3_asr_result_get_t_mel_ms(qwen3_asr_result_t* res) { return res ? res->cpp_result.t_mel_ms : 0; }
int64_t qwen3_asr_result_get_t_encode_ms(qwen3_asr_result_t* res) { return res ? res->cpp_result.t_encode_ms : 0; }
int64_t qwen3_asr_result_get_t_decode_ms(qwen3_asr_result_t* res) { return res ? res->cpp_result.t_decode_ms : 0; }
int64_t qwen3_asr_result_get_t_total_ms(qwen3_asr_result_t* res) { return res ? res->cpp_result.t_total_ms : 0; }

void qwen3_asr_result_free(qwen3_asr_result_t* res) {
    if (res) {
        delete res;
    }
}
