import os
import csv
import json
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_dsp_preprocessing import get_model1_dsp_pipeline

split_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_split.csv"
norm_json = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_normalization.json"
report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model1_dsp_normalization_report.txt"

train_paths = []
samples = {"plantvillage": None, "caltech256": None, "places365": None}

with open(split_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["split"] == "train":
            train_paths.append(row["image_path"])
        
        # Collect one sample from each source for verification
        src = row["source"]
        if samples.get(src) is None:
            samples[src] = row["image_path"]

class SimpleImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path)
            tensor = self.transform(img)
            return tensor, path, True
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return torch.zeros(3, 224, 224), path, False

if __name__ == '__main__':
    start_time = time.time()
    
    # Unnormalized transform for stats
    base_transform = get_model1_dsp_pipeline(mean=None, std=None)
    
    dataset = SimpleImageDataset(train_paths, base_transform)
    loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False)
    
    fst_moment = torch.zeros(3)
    snd_moment = torch.zeros(3)
    cnt = 0
    failed_images = 0
    
    print(f"Computing DSP-specific statistics over {len(train_paths)} training images...")
    for batch, paths, valid in loader:
        B = batch.size(0)
        
        # Filter out failed images
        valid_mask = valid.bool()
        if not valid_mask.any():
            failed_images += B
            continue
            
        valid_batch = batch[valid_mask]
        failed_images += (B - valid_batch.size(0))
        
        # valid_batch shape: [V, 3, 224, 224]
        fst_moment += torch.mean(valid_batch, dim=[2, 3]).sum(dim=0)
        snd_moment += torch.mean(valid_batch ** 2, dim=[2, 3]).sum(dim=0)
        cnt += valid_batch.size(0)
        
    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment / cnt - mean ** 2)
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    res = {
        "mean": mean_list,
        "std": std_list,
        "image_size": [3, 224, 224],
        "computed_from": "train_split_only",
        "number_of_images": cnt,
        "dsp_preprocessing": True,
        "split": "train",
        "dsp_settings": {
            "channel": "HSV Saturation",
            "filter": "GaussianBlur 5x5",
            "thresholding": "Otsu",
            "morphology": "Open (2), Close (2), Ellipse 5x5",
            "noise_removal": "Largest Connected Component"
        }
    }
    
    with open(norm_json, 'w') as f:
        json.dump(res, f, indent=4)
        
    print(f"Saved normalization statistics to {norm_json}")
    
    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------
    print("Running verification on samples...")
    normalized_transform = get_model1_dsp_pipeline(mean=mean_list, std=std_list)
    
    report = []
    report.append("Model 1 DSP Normalization Report")
    report.append("================================")
    report.append(f"Number of training images processed: {res['number_of_images']}")
    report.append(f"Computed RGB mean: {res['mean']}")
    report.append(f"Computed RGB standard deviation: {res['std']}")
    report.append(f"Processing time: {processing_time:.2f} seconds")
    report.append(f"Failed images: {failed_images}")
    report.append("")
    report.append("Final Deterministic Preprocessing Sequence:")
    report.append(str(normalized_transform))
    report.append("")
    report.append("Sample Verification Results:")
    
    failures = 0
    for src, path in samples.items():
        try:
            img = Image.open(path)
            tensor = normalized_transform(img)
            shape = list(tensor.shape)
            is_float = tensor.dtype == torch.float32
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            
            report.append(f"\nSource: {src}")
            report.append(f"  Path: {path}")
            report.append(f"  Shape (3x224x224 expected): {shape}")
            report.append(f"  Dtype (float32 expected): {tensor.dtype}")
            report.append(f"  NaNs present: {has_nan}")
            report.append(f"  Infs present: {has_inf}")
            
            passed = (shape == [3, 224, 224]) and is_float and (not has_nan) and (not has_inf)
            report.append("  Status: " + ("PASS" if passed else "FAIL"))
            
            if not passed:
                failures += 1
        except Exception as e:
            report.append(f"\nSource: {src} - FAILED with error: {e}")
            failures += 1
            
    report.append(f"\nTotal failures encountered: {failures}")
    
    with open(report_out, 'w') as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {report_out}")
    print("Done!")
