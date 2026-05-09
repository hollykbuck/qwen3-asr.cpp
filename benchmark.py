import time
import sys
from qwen3_asr import Qwen3ASR

def benchmark(model_path, audio_path, device_id=-1):
    print(f"Initializing ASR with model: {model_path}")
    asr = Qwen3ASR()
    
    if device_id >= 0:
        devices = asr.get_devices()
        if device_id < len(devices):
            print(f"Selecting Device [{device_id}]: {devices[device_id]['description']}")
            asr.set_gpu_device(device_id)
        else:
            print(f"Warning: Device index {device_id} out of range")
    
    start_load = time.time()
    if not asr.load_model(model_path):
        print("Failed to load model")
        return
    end_load = time.time()
    print(f"Model loaded in {(end_load - start_load)*1000:.2f}ms")

    # Run once to warm up Vulkan/GPU
    print("Warming up...")
    asr.transcribe(audio_path)
    
    # Run second time for actual benchmark
    print(f"Benchmarking 30s audio: {audio_path}")
    
    # We'll measure both wall-clock time in Python and the internal C++ timing
    start_wall = time.time()
    result = asr.transcribe(audio_path)
    end_wall = time.time()
    
    if result["success"]:
        print("\n--- Benchmark Results ---")
        print(f"Internal C++ Total Time: {result['time_ms']}ms")
        print(f"Python Wall-clock Time:  {(end_wall - start_wall)*1000:.2f}ms")
        print(f"Transcription: {result['text'][:100]}...")
        
        # Calculate Real-time Factor (RTF)
        # Assuming output_30s.wav is 30 seconds
        duration_s = 30.0 
        rtf = (result['time_ms'] / 1000.0) / duration_s
        print(f"Real-time Factor (RTF): {rtf:.4f} (lower is better)")
        print(f"Speedup: {1/rtf:.2f}x faster than real-time")
    else:
        print(f"Transcription failed: {result['error']}")

if __name__ == "__main__":
    model = "models/qwen3-asr-0.6b-f16.gguf"
    audio = "output_30s.wav"
    device = -1
    if len(sys.argv) > 1: model = sys.argv[1]
    if len(sys.argv) > 2: audio = sys.argv[2]
    if len(sys.argv) > 3: device = int(sys.argv[3])
    benchmark(model, audio, device)
