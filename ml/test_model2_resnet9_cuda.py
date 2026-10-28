import os
import torch
import torch.nn as nn
import traceback

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model2_resnet9 import create_model2_resnet9
from src.datasets.model2_dataset import create_model2_dataloaders

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    return total

def run_validation():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model2_resnet9_architecture.txt"
    report = []
    
    report.append("Model 2 ResNet-9 Architecture Validation Report")
    report.append("===============================================\n")
    
    try:
        # Initialize model
        model = create_model2_resnet9()
        total_params = count_parameters(model)
        
        report.append("Architecture Details:")
        report.append("  Base Architecture: ResNet-9 (Reused Model 1 Architecture)")
        report.append("  Output Classes: 3")
        report.append("  Label Mapping: Potato=0, Tomato=1, Other=2")
        report.append(f"  Total Parameters: {total_params:,}")
        
        # Setup CUDA
        cuda_available = torch.cuda.is_available()
        report.append(f"\nCUDA Status: {'Available' if cuda_available else 'Unavailable'}")
        if not cuda_available:
            raise RuntimeError("CUDA is required for this validation.")
            
        gpu_name = torch.cuda.get_device_name(0)
        report.append(f"  GPU Name: {gpu_name}")
        
        model = model.to('cuda')
        model.train()
        torch.cuda.reset_peak_memory_stats()
        
        # Load one batch
        print("Loading one training batch...")
        train_loader, _, _ = create_model2_dataloaders(num_workers=0)
        iterator = iter(train_loader)
        images, labels = next(iterator)
        
        images = images.to('cuda')
        labels = labels.to('cuda')
        
        # Forward pass
        print("Running forward pass...")
        logits = model(images)
        
        shape = list(logits.shape)
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        
        report.append(f"\nForward Pass:")
        report.append(f"  Input Shape: {list(images.shape)}")
        report.append(f"  Output Shape: {shape} (Expected: [64, 3])")
        report.append(f"  Contains NaNs: {has_nan}")
        report.append(f"  Contains Infs: {has_inf}")
        
        forward_ok = (shape == [images.size(0), 3] and not has_nan and not has_inf)
        
        # Weighted CrossEntropy
        print("Running backward pass...")
        class_weights = torch.tensor([8.4145, 1.0027, 0.5308], dtype=torch.float32, device='cuda')
        report.append(f"\nWeighted CrossEntropy Weights: {class_weights.tolist()}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loss = criterion(logits, labels)
        
        is_finite_loss = torch.isfinite(loss).item()
        report.append(f"  Sanity Loss Value: {loss.item():.4f} (Finite: {is_finite_loss})")
        
        # Backward pass
        loss.backward()
        
        layers_to_check = {
            "First Conv": model.block1.block[0],
            "Residual Block 1": model.res1.conv1,
            "Residual Block 2": model.res2.conv1,
            "Final Linear": model.classifier[2]
        }
        
        report.append("\nGradient Check Result:")
        all_grads_valid = True
        
        for name, layer in layers_to_check.items():
            grad = layer.weight.grad
            if grad is None:
                report.append(f"  {name}: NULL GRADIENT (FAIL)")
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
            
        model.zero_grad()
        
        # Memory
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        report.append(f"\nPeak GPU Allocated Memory: {peak_alloc:.2f} MB")
        
        if forward_ok and is_finite_loss and all_grads_valid:
            report.append("\nMODEL 2 ARCHITECTURE VALIDATION PASSED")
        else:
            report.append("\nMODEL 2 ARCHITECTURE VALIDATION FAILED")
            
    except Exception as e:
        report.append(f"\nValidation FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nMODEL 2 ARCHITECTURE VALIDATION FAILED")
        print(f"Failed with error: {e}")
        
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Validation complete. Report saved to {report_out}")

if __name__ == "__main__":
    run_validation()
