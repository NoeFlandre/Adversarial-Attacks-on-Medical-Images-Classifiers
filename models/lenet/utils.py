import torch

def get_device():
    """
    Get the appropriate device for PyTorch operations.
    Priority: CUDA > MPS > CPU
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    return device 