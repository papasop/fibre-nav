def a100_device(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('A100 CUDA unavailable; switch Colab runtime to A100')
    name=torch.cuda.get_device_name(0)
    if 'A100' not in name:
        raise RuntimeError('This protocol requires NVIDIA A100; observed '+name)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda:0'),name
