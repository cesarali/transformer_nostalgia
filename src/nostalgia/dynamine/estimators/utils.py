import torch
import time

def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return "mps"
    else:
        print("No GPU available. Using CPU.")
        return "cpu"

def print_time():
    current_time = time.time()  # Current time in seconds since the epoch
    local_time = time.localtime(current_time)  # Convert to local time

    formatted_time = time.strftime("%H:%M:%S", local_time)
    print("Current time:", formatted_time)
