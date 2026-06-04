import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available! Using {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("CUDA is not available. Falling back to CPU.")

check_cuda()