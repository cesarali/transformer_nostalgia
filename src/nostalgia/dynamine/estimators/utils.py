import torch

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
