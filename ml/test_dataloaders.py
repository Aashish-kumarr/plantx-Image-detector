import time
import torch
import os
from collections import Counter

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_dataset import create_model1_dataloaders

def validate_dataloaders():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model1_dataloader_report.txt"
    report = []
    
    report.append("Model 1 DataLoader Validation Report")
    report.append("=====================================")
    
    print("Initializing DataLoaders...")
    batch_size = 64
    num_workers = 4
    train_loader, val_loader, test_loader = create_model1_dataloaders(
        batch_size=batch_size, 
        num_workers=num_workers
    )
    
    # --- 1. Validate Dataset Sizes and Class Counts ---
    print("Validating lengths and counts...")
    expected_lengths = {'train': 88869, 'val': 11108, 'test': 11110}
    
    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    report.append("Dataset Configurations:")
    report.append(f"  Batch Size: {batch_size}")
    report.append(f"  Num Workers: {num_workers}")
    report.append("")
    report.append("Dataset Split Summary:")
    
    for split_name, loader in loaders.items():
        dataset = loader.dataset
        length = len(dataset)
        labels = [sample['label'] for sample in dataset.samples]
        counts = Counter(labels)
        
        report.append(f"  Split: {split_name}")
        report.append(f"    Total Images: {length} (Expected: {expected_lengths[split_name]})")
        report.append(f"    Label 0 (Non-Leaf): {counts.get(0, 0)}")
        report.append(f"    Label 1 (Leaf): {counts.get(1, 0)}")
        
        if length != expected_lengths[split_name]:
            report.append(f"    WARNING: Length mismatch for {split_name}!")
            
    # --- 2. Batch Verification ---
    print("Verifying batches...")
    report.append("\nBatch Verification:")
    failures = 0
    
    for split_name, loader in loaders.items():
        try:
            # Fetch just one batch for shape validation
            images, labels = next(iter(loader))
            
            shape = list(images.shape)
            dtype = images.dtype
            has_nan = torch.isnan(images).any().item()
            has_inf = torch.isinf(images).any().item()
            
            # Label check (only 0s and 1s)
            valid_labels = torch.all((labels == 0) | (labels == 1)).item()
            
            passed = (shape == [batch_size, 3, 224, 224]) and (dtype == torch.float32) and not has_nan and not has_inf and valid_labels
            
            report.append(f"  {split_name} DataLoader:")
            report.append(f"    Image Batch Shape: {shape}")
            report.append(f"    Label Batch Shape: {list(labels.shape)}")
            report.append(f"    Image dtype: {dtype}")
            report.append(f"    NaNs Present: {has_nan}")
            report.append(f"    Infs Present: {has_inf}")
            report.append(f"    Labels are valid (0/1): {valid_labels}")
            report.append(f"    Status: {'PASS' if passed else 'FAIL'}")
            
            if not passed:
                failures += 1
                
        except Exception as e:
            report.append(f"  {split_name} DataLoader: FAILED with error: {e}")
            failures += 1

    # --- 3. Performance Sanity Check ---
    print("Running performance sanity check (10 batches)...")
    report.append("\nPerformance Sanity Check (Training Loader):")
    
    num_test_batches = 10
    start_time = time.time()
    
    # Iterate exactly 10 batches
    try:
        iterator = iter(train_loader)
        for i in range(num_test_batches):
            images, labels = next(iterator)
            
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_test_batches
        images_per_second = batch_size / avg_time_per_batch
        
        report.append(f"  Total time for {num_test_batches} batches: {total_time:.2f}s")
        report.append(f"  Average time per batch: {avg_time_per_batch:.2f}s")
        report.append(f"  Approximate images per second: {images_per_second:.1f} imgs/s")
        report.append("  Note: This includes disk I/O, OpenCV morphology, OpenCV masking, PIL scaling, and CPU augmentations.")
        
    except Exception as e:
        report.append(f"  Performance check FAILED with error: {e}")
        failures += 1

    report.append(f"\nTotal Errors Encountered: {failures}")
    
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Validation complete. Report saved to {report_out}")

if __name__ == '__main__':
    validate_dataloaders()
