import torch


# %%
def get_compute_device(force_cpu=False, prefer_last=True):
    CUDA_available = torch.cuda.is_available()
    if CUDA_available:
        CUDA_count = torch.cuda.device_count()

    compute_device = torch.device("cpu")

    if force_cpu or not CUDA_available:
        return compute_device

    elif prefer_last and CUDA_count > 1:
        compute_device = torch.device(f"cuda:{CUDA_count - 1}")
        return compute_device

    else:
        compute_device = torch.device("cuda")
        return torch.device("cuda:0")
