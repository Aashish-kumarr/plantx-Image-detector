import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Compiled architectures:", torch.cuda.get_arch_list())

x = torch.randn(1000, 1000, device="cuda")
y = x @ x
print("CUDA tensor operation:", y.shape)
