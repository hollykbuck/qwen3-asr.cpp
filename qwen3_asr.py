import ctypes
import os
import platform
import numpy as np

class Qwen3ASRParams(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_int32),
        ("language", ctypes.c_char_p),
        ("n_threads", ctypes.c_int32),
        ("print_progress", ctypes.c_bool),
    ]

class Qwen3ASR:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Default library name based on platform
            system = platform.system()
            if system == "Windows":
                lib_name = "qwen3_asr_shared.dll"
            elif system == "Darwin":
                lib_name = "libqwen3_asr_shared.dylib"
            else:
                lib_name = "libqwen3_asr_shared.so"
            
            # Try current directory first
            lib_path = os.path.join(os.getcwd(), lib_name)
            if not os.path.exists(lib_path):
                # Search common build locations (presets, etc.)
                search_paths = [
                    os.path.join(os.getcwd(), "build", lib_name),
                    os.path.join(os.getcwd(), "build", "Release", lib_name),
                    os.path.join(os.getcwd(), "build", "RelWithDebInfo", lib_name),
                    os.path.join(os.getcwd(), "build", "Debug", lib_name),
                    os.path.join(os.getcwd(), "out", "build", "x64-Release", lib_name),
                    os.path.join(os.getcwd(), "build", "conan-relwithdebinfo", lib_name),
                ]
                for p in search_paths:
                    if os.path.exists(p):
                        lib_path = p
                        break

        self.lib = ctypes.CDLL(lib_path)

        # Setup function signatures
        self.lib.qwen3_asr_default_params.restype = Qwen3ASRParams
        
        self.lib.qwen3_asr_create.restype = ctypes.c_void_p
        
        self.lib.qwen3_asr_free.argtypes = [ctypes.c_void_p]
        
        self.lib.qwen3_asr_get_device_count.restype = ctypes.c_int
        self.lib.qwen3_asr_get_device_description.argtypes = [ctypes.c_int]
        self.lib.qwen3_asr_get_device_description.restype = ctypes.c_char_p
        
        self.lib.qwen3_asr_set_gpu_device.argtypes = [ctypes.c_void_p, ctypes.c_int]
        
        self.lib.qwen3_asr_load_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.qwen3_asr_load_model.restype = ctypes.c_bool
        
        self.lib.qwen3_asr_transcribe_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, Qwen3ASRParams]
        self.lib.qwen3_asr_transcribe_file.restype = ctypes.c_void_p
        
        self.lib.qwen3_asr_transcribe_buffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, Qwen3ASRParams]
        self.lib.qwen3_asr_transcribe_buffer.restype = ctypes.c_void_p
        
        self.lib.qwen3_asr_result_get_success.argtypes = [ctypes.c_void_p]
        self.lib.qwen3_asr_result_get_success.restype = ctypes.c_bool
        
        self.lib.qwen3_asr_result_get_text.argtypes = [ctypes.c_void_p]
        self.lib.qwen3_asr_result_get_text.restype = ctypes.c_char_p
        
        self.lib.qwen3_asr_result_get_error.argtypes = [ctypes.c_void_p]
        self.lib.qwen3_asr_result_get_error.restype = ctypes.c_char_p
        
        self.lib.qwen3_asr_result_get_token_count.argtypes = [ctypes.c_void_p]
        self.lib.qwen3_asr_result_get_token_count.restype = ctypes.c_int32
        
        self.lib.qwen3_asr_result_get_tokens.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        
        self.lib.qwen3_asr_result_get_t_total_ms.argtypes = [ctypes.c_void_p]
        self.lib.qwen3_asr_result_get_t_total_ms.restype = ctypes.c_int64
        
        self.lib.qwen3_asr_result_free.argtypes = [ctypes.c_void_p]

        self.ctx = self.lib.qwen3_asr_create()

    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.qwen3_asr_free(self.ctx)

    def get_devices(self):
        count = self.lib.qwen3_asr_get_device_count()
        devices = []
        for i in range(count):
            desc = self.lib.qwen3_asr_get_device_description(i).decode('utf-8')
            devices.append({"index": i, "description": desc})
        return devices

    def set_gpu_device(self, device_id):
        self.lib.qwen3_asr_set_gpu_device(self.ctx, device_id)

    def load_model(self, model_path):
        return self.lib.qwen3_asr_load_model(self.ctx, model_path.encode('utf-8'))

    def get_default_params(self):
        return self.lib.qwen3_asr_default_params()

    def transcribe(self, audio, params=None):
        if params is None:
            params = self.get_default_params()
        
        if isinstance(audio, str):
            res_ptr = self.lib.qwen3_asr_transcribe_file(self.ctx, audio.encode('utf-8'), params)
        elif isinstance(audio, (np.ndarray, list)):
            samples = np.array(audio, dtype=np.float32)
            ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            res_ptr = self.lib.qwen3_asr_transcribe_buffer(self.ctx, ptr, len(samples), params)
        else:
            raise ValueError("audio must be a file path (str) or a numpy array/list of floats")

        if not res_ptr:
            return {"success": False, "error": "Failed to initiate transcription"}

        try:
            success = self.lib.qwen3_asr_result_get_success(res_ptr)
            text = self.lib.qwen3_asr_result_get_text(res_ptr).decode('utf-8')
            error = self.lib.qwen3_asr_result_get_error(res_ptr).decode('utf-8')
            token_count = self.lib.qwen3_asr_result_get_token_count(res_ptr)
            total_time = self.lib.qwen3_asr_result_get_t_total_ms(res_ptr)
            
            tokens = []
            if token_count > 0:
                tokens_arr = (ctypes.c_int32 * token_count)()
                self.lib.qwen3_asr_result_get_tokens(res_ptr, tokens_arr)
                tokens = list(tokens_arr)

            return {
                "success": success,
                "text": text,
                "error": error,
                "tokens": tokens,
                "time_ms": total_time
            }
        finally:
            self.lib.qwen3_asr_result_free(res_ptr)

if __name__ == "__main__":
    # Example usage (requires the shared library to be built)
    import sys
    if len(sys.argv) < 3:
        print("Usage: python qwen3_asr.py <model_path> <audio_path>")
        sys.exit(1)
        
    asr = Qwen3ASR()
    if asr.load_model(sys.argv[1]):
        print("Model loaded.")
        result = asr.transcribe(sys.argv[2])
        if result["success"]:
            print(f"Result: {result['text']}")
            print(f"Time: {result['time_ms']}ms")
        else:
            print(f"Error: {result['error']}")
    else:
        print("Failed to load model.")
