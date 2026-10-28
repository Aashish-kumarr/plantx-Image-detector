import os
import torch
import torch.nn as nn
import traceback

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model1_resnet9 import create_model1_resnet9
from src.datasets.model1_dataset import create_model1_dataloaders

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    size_mb = total * 4 / (1024 ** 2)
    return total, trainable, non_trainable, size_mb

def run_gpu_validation():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model1_resnet9_architecture.txt"
    report = []
    report.append("Model 1 ResNet-9 Architecture Validation Report")
    report.append("===============================================")
    
    # --- 1. Architecture Info ---
    try:
        model = create_model1_resnet9()
        total, trainable, non_trainable, size_mb = count_parameters(model)
        
        report.append("\nModel Statistics:")
        report.append(f"  Total parameters: {total:,}")
        report.append(f"  Trainable parameters: {trainable:,}")
        report.append(f"  Non-trainable parameters: {non_trainable:,}")
        report.append(f"  Approximate FP32 Size: {size_mb:.2f} MB")
        report.append("  Initialization: Kaiming/He for Conv layers, ones/zeros for BatchNorm, normal/zeros for Linear")
        
        report.append("\nArchitecture Summary:")
        report.append(str(model))
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        return
        
    # --- 2. CPU Dummy Test ---
    report.append("\nValidation Test 1: CPU Dummy Forward")
    try:
        dummy_input = torch.randn(4, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            dummy_out = model(dummy_input)
            
        shape = list(dummy_out.shape)
        if shape == [4, 2]:
            report.append(f"  Input shape: {list(dummy_input.shape)}")
            report.append(f"  Output shape: {shape} (Expected: [4, 2])")
            report.append("  Result: PASS")
        else:
            report.append("  Result: FAIL")
    except Exception as e:
        report.append(f"  Result: FAIL with exception: {e}")
        
    # --- 3. CUDA Availability Check ---
    report.append("\nValidation Test 2: CUDA Real-Batch Forward")
    cuda_available = torch.cuda.is_available()
    report.append(f"  torch.cuda.is_available(): {cuda_available}")
    
    if not cuda_available:
        report.append("  ERROR: CUDA is unavailable. Stopping validation as requested.")
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print("CUDA Unavailable")
        return
        
    gpu_name = torch.cuda.get_device_name(0)
    cuda_device = torch.cuda.current_device()
    cuda_version = torch.version.cuda
    pytorch_version = torch.__version__
    report.append(f"  PyTorch Version: {pytorch_version}")
    report.append(f"  PyTorch CUDA Version: {cuda_version}")
    report.append(f"  GPU Model: {gpu_name}")
    report.append(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # --- 4. CUDA Real Batch Test ---
    try:
        # Move model to CUDA
        model = model.to('cuda')
        model.train() # Set to train for gradients later
        torch.cuda.reset_peak_memory_stats()
        
        # Get one batch
        print("Fetching a real batch...")
        train_loader, _, _ = create_model1_dataloaders(batch_size=64, num_workers=0, persistent_workers=False, prefetch_factor=None)
        
        iterator = iter(train_loader)
        real_images, real_labels = next(iterator)
        
        # Send to GPU
        real_images = real_images.to('cuda')
        real_labels = real_labels.to('cuda')
        
        # Forward pass
        logits = model(real_images)
        
        shape = list(logits.shape)
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        mem_res = torch.cuda.memory_reserved() / (1024**2)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        
        report.append(f"  Input batch shape: {list(real_images.shape)}")
        report.append(f"  Output logits shape: {shape} (Expected: [64, 2])")
        report.append(f"  NaNs present: {has_nan}")
        report.append(f"  Infs present: {has_inf}")
        report.append(f"  Current Allocated GPU Memory: {mem_alloc:.2f} MB")
        report.append(f"  Peak Allocated GPU Memory: {peak_alloc:.2f} MB")
        report.append(f"  Reserved GPU Memory: {mem_res:.2f} MB")
        
        forward_pass_ok = (shape == [64, 2] and not has_nan and not has_inf)
        if forward_pass_ok:
            report.append("  Forward Pass Result: PASS")
        else:
            report.append("  Forward Pass Result: FAIL")
            
        # --- 5. Gradient Sanity Check ---
        report.append("\nBasic Gradient Sanity Check:")
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, real_labels)
        
        is_finite_loss = torch.isfinite(loss).item()
        report.append(f"  CrossEntropyLoss Value: {loss.item():.4f}")
        report.append(f"  Loss is finite: {is_finite_loss}")
        
        loss.backward()
        
        # Select target layers
        layers_to_check = {
            "First Conv": model.block1.block[0],
            "ResBlock 1 Conv1": model.res1.conv1,
            "ResBlock 2 Conv1": model.res2.conv1,
            "Final Linear": model.classifier[2]
        }
        
        all_grads_valid = True
        
        for name, layer in layers_to_check.items():
            grad = layer.weight.grad
            if grad is None:
                report.append(f"  {name} gradient: NULL (FAIL)")
                all_grads_valid = False
                continue
                
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            is_zeros = torch.all(grad == 0).item()
            norm = torch.norm(grad).item()
            
            valid = not (has_nan or has_inf or is_zeros)
            if not valid:
                all_grads_valid = False
                
            report.append(f"  {name} grad norm: {norm:.4f} (Valid: {valid})")
            
        if all_grads_valid and is_finite_loss and forward_pass_ok:
            report.append("  Gradient Sanity Check Result: PASS (Finite non-zero gradients received)")
            report.append("\nFinal Status: STEP 15 PASSED")
        else:
            report.append("  Gradient Sanity Check Result: FAIL")
            report.append("\nFinal Status: STEP 15 FAILED")
            
        # Clean up
        model.zero_grad()
            
    except Exception as e:
        report.append(f"\nCUDA/Gradient Check FAILED with exception: {traceback.format_exc()}")
        report.append("\nFinal Status: STEP 15 FAILED")
        
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Validation complete. Report saved to {report_out}")

if __name__ == "__main__":
    run_gpu_validation()
