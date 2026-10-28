import os
import torch
import json
import traceback

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.datasets.model2_dataset import create_model2_dataloaders

def run_test():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model2_preprocessing_report.txt"
    report = []
    
    report.append("Model 2 Preprocessing & DataLoader Report")
    report.append("=========================================\n")
    
    try:
        # Load normalization stats
        json_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model2_normalization.json"
        with open(json_path, 'r') as f:
            norm_stats = json.load(f)
            
        report.append("Normalization Information:")
        report.append(f"  Mean: {norm_stats['mean']}")
        report.append(f"  Std: {norm_stats['std']}")
        report.append(f"  Images used for computation: {norm_stats['number_of_images']}")
        report.append(f"  DSP Preprocessing Applied: {norm_stats['dsp_preprocessing']}")
        
        report.append("\nDSP & Augmentation Strategy:")
        report.append("  DSP Pipeline: Reused Model 1 (HSV masking + morphology -> crop/resize to 224x224)")
        report.append("  Training Augmentation: Reused Model 1 (Flip, Rotation, Affine, ColorJitter)")
        
        # Load dataloaders
        print("Initializing dataloaders...")
        train_loader, val_loader, test_loader = create_model2_dataloaders(num_workers=0)
        
        report.append("\nDataLoader Configuration:")
        report.append("  batch_size: 64")
        report.append("  num_workers: 0")
        report.append("  pin_memory: False")
        
        train_len = len(train_loader.dataset)
        val_len = len(val_loader.dataset)
        test_len = len(test_loader.dataset)
        
        report.append("\nDataset Counts:")
        report.append(f"  Train: {train_len}")
        report.append(f"  Validation: {val_len}")
        report.append(f"  Test: {test_len}")
        
        # Verify batches
        print("Testing batches...")
        iterator = iter(train_loader)
        images, labels = next(iterator)
        
        shape = list(images.shape)
        unique_labels = sorted(torch.unique(labels).tolist())
        has_nans = torch.isnan(images).any().item()
        has_infs = torch.isinf(images).any().item()
        
        report.append("\nBatch Verification (Train):")
        report.append(f"  Tensor shape: {shape} (Expected: [64, 3, 224, 224] or smaller for last batch)")
        report.append(f"  Labels in batch: {unique_labels} (Must be subset of [0, 1, 2])")
        report.append(f"  Contains NaNs: {has_nans}")
        report.append(f"  Contains Infs: {has_infs}")
        
        valid = (len(shape) == 4 and shape[1:] == [3, 224, 224] 
                 and all(lbl in [0,1,2] for lbl in unique_labels) 
                 and not has_nans and not has_infs)
                 
        if valid:
            report.append("  Batch test: PASSED")
            report.append("\nSTEP 19 PASSED")
        else:
            report.append("  Batch test: FAILED")
            report.append("\nSTEP 19 FAILED")
            
        # Clean up memory
        del iterator
        del train_loader
        del val_loader
        del test_loader
        
    except Exception as e:
        report.append(f"\nValidation FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nSTEP 19 FAILED")
        print(f"Failed with error: {e}")
        
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {report_out}")

if __name__ == "__main__":
    run_test()
