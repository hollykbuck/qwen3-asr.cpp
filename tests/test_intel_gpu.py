from qwen3_asr import Qwen3ASR
import time

def test_intel_gpu():
    model_path = "models/qwen3-asr-0.6b-f16.gguf"
    audio_path = "sample.wav"
    
    print("Initializing ASR...")
    asr = Qwen3ASR()
    
    print("Selecting Device [1] Intel(R) UHD Graphics 770...")
    asr.set_gpu_device(1)
    
    start_time = time.time()
    if asr.load_model(model_path):
        print(f"Model loaded in {time.time() - start_time:.2f}s")
        
        print("Running transcription on Intel GPU...")
        result = asr.transcribe(audio_path)
        
        if result["success"]:
            print("\n--- Success ---")
            print(f"Result: {result['text']}")
            print(f"Internal Time: {result['time_ms']}ms")
        else:
            print(f"Transcription failed: {result['error']}")
    else:
        print("Failed to load model.")

if __name__ == "__main__":
    test_intel_gpu()
