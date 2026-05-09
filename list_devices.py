from qwen3_asr import Qwen3ASR

def list_devices():
    asr = Qwen3ASR()
    devices = asr.get_devices()
    print("Available Devices:")
    for dev in devices:
        print(f"[{dev['index']}] {dev['description']}")

if __name__ == "__main__":
    list_devices()
