#ifndef QWEN3_ASR_C_H
#define QWEN3_ASR_C_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef QWEN3_ASR_SHARED_EXPORTS
#    define QWEN3_ASR_API __declspec(dllexport)
#  else
#    define QWEN3_ASR_API __declspec(dllimport)
#  endif
#else
#  define QWEN3_ASR_API __attribute__((visibility("default")))
#endif

// Opaque handles
typedef struct qwen3_asr_context qwen3_asr_context_t;
typedef struct qwen3_asr_result qwen3_asr_result_t;

// Transcription parameters
typedef struct {
    int32_t max_tokens;
    const char* language;
    int32_t n_threads;
    bool print_progress;
} qwen3_asr_params_t;

// Get default parameters
QWEN3_ASR_API qwen3_asr_params_t qwen3_asr_default_params();

// Create and destroy ASR context
QWEN3_ASR_API qwen3_asr_context_t* qwen3_asr_create();
QWEN3_ASR_API void qwen3_asr_free(qwen3_asr_context_t* ctx);

// Device management
QWEN3_ASR_API int qwen3_asr_get_device_count();
QWEN3_ASR_API const char* qwen3_asr_get_device_description(int index);
QWEN3_ASR_API void qwen3_asr_set_gpu_device(qwen3_asr_context_t* ctx, int device_id);

// Load model from GGUF file
QWEN3_ASR_API bool qwen3_asr_load_model(qwen3_asr_context_t* ctx, const char* model_path);

// Transcribe audio
QWEN3_ASR_API qwen3_asr_result_t* qwen3_asr_transcribe_file(
    qwen3_asr_context_t* ctx, 
    const char* audio_path, 
    qwen3_asr_params_t params);

QWEN3_ASR_API qwen3_asr_result_t* qwen3_asr_transcribe_buffer(
    qwen3_asr_context_t* ctx, 
    const float* samples, 
    int32_t n_samples, 
    qwen3_asr_params_t params);

// Result accessors
QWEN3_ASR_API bool qwen3_asr_result_get_success(qwen3_asr_result_t* res);
QWEN3_ASR_API const char* qwen3_asr_result_get_text(qwen3_asr_result_t* res);
QWEN3_ASR_API const char* qwen3_asr_result_get_error(qwen3_asr_result_t* res);
QWEN3_ASR_API int32_t qwen3_asr_result_get_token_count(qwen3_asr_result_t* res);
QWEN3_ASR_API void qwen3_asr_result_get_tokens(qwen3_asr_result_t* res, int32_t* out_tokens);

// Timing accessors (milliseconds)
QWEN3_ASR_API int64_t qwen3_asr_result_get_t_load_ms(qwen3_asr_result_t* res);
QWEN3_ASR_API int64_t qwen3_asr_result_get_t_mel_ms(qwen3_asr_result_t* res);
QWEN3_ASR_API int64_t qwen3_asr_result_get_t_encode_ms(qwen3_asr_result_t* res);
QWEN3_ASR_API int64_t qwen3_asr_result_get_t_decode_ms(qwen3_asr_result_t* res);
QWEN3_ASR_API int64_t qwen3_asr_result_get_t_total_ms(qwen3_asr_result_t* res);

// Free result
QWEN3_ASR_API void qwen3_asr_result_free(qwen3_asr_result_t* res);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_ASR_C_H
