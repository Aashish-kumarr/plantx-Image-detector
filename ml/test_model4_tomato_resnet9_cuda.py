import os
import torch
import torch.nn as nn
import traceback

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model4_tomato_resnet9 import create_model4_tomato_resnet9
from src.datasets.model4_tomato_dataset import create_model4_tomato_dataloaders

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def run_validation():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model4_tomato_architecture.txt"
    report = []
    
    report.append("Model 4 (Tomato) ResNet-9 Architecture Validation Report")
    report.append("========================================================\n")
    
    try:
        # Initialize model
        model = create_model4_tomato_resnet9()
        total_params, trainable_params = count_parameters(model)
        
        # Approximate size in MB (4 bytes per param for FP32)
        model_size_mb = (total_params * 4) / (1024 ** 2)
        
        report.append("Architecture Details:")
        report.append("  Base Architecture: ResNet-9")
        report.append("  Output Classes: 10")
        report.append("  Label Mapping:")
        report.append("    0=healthy, 1=early_blight, 2=late_blight, 3=bacterial_spot, 4=leaf_mold")
        report.append("    5=septoria_leaf_spot, 6=spider_mites, 7=target_spot, 8=tomato_mosaic_virus, 9=tomato_yellow_leaf_curl_virus")
        report.append("  Normalization: Mean=[0.1070, 0.1361, 0.0718], Std=[0.1560, 0.1926, 0.1082]")
        report.append(f"  Total Parameters: {total_params:,}")
        report.append(f"  Trainable Parameters: {trainable_params:,}")
        report.append(f"  Approximate FP32 Model Size: {model_size_mb:.2f} MB")
        
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
        train_loader, _, _ = create_model4_tomato_dataloaders(num_workers=0)
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
        report.append(f"  Output Shape: {shape} (Expected: [64, 10])")
        report.append(f"  Contains NaNs: {has_nan}")
        report.append(f"  Contains Infs: {has_inf}")
        
        forward_ok = (shape == [images.size(0), 10] and not has_nan and not has_inf)
        
        # Weighted CrossEntropy
        print("Running backward pass...")
        class_weights = torch.tensor([
            0.9151, 1.1519, 0.8384, 0.7980, 1.1880, 
            0.8757, 0.8896, 0.9689, 1.8759, 0.4986
        ], dtype=torch.float32, device='cuda')
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
            report.append("\nMODEL 4 ARCHITECTURE VALIDATION PASSED")
        else:
            report.append("\nMODEL 4 ARCHITECTURE VALIDATION FAILED")
            
    except Exception as e:
        report.append(f"\nValidation FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nMODEL 4 ARCHITECTURE VALIDATION FAILED")
        print(f"Failed with error: {e}")
        
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Validation complete. Report saved to {report_out}")

if __name__ == "__main__":
    run_validation()
