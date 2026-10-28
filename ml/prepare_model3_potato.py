import os
import json
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.datasets.model1_dsp_preprocessing import apply_dsp_preprocessing

class TempNormDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        pil_img = Image.open(img_path).convert('RGB')
        dsp_pil = apply_dsp_preprocessing(pil_img)
        img_rgb = np.array(dsp_pil)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor

def run():
    input_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model2_14class_split.csv"
    output_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model3_potato_split.csv"
    report_file = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model3_potato_dataset_report.txt"
    norm_json = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model3_potato_normalization.json"
    
    print("Loading existing metadata...")
    df = pd.read_csv(input_csv)
    
    potato_classes = [
        'Potato___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight'
    ]
    
    label_map = {
        'Potato___healthy': ('healthy', 0),
        'Potato___Early_blight': ('early_blight', 1),
        'Potato___Late_blight': ('late_blight', 2)
    }
    
    # 1. Filter Potato images
    df_potato = df[df['original_class'].isin(potato_classes)].copy()
    
    # Map labels
    mapped = df_potato['original_class'].map(label_map)
    df_potato['label'] = [x[0] for x in mapped]
    df_potato['label_id'] = [x[1] for x in mapped]
    
    cols = ['image_path', 'label', 'label_id', 'original_class', 'split']
    other_cols = [c for c in df_potato.columns if c not in cols]
    df_potato = df_potato[cols + other_cols]
    
    # Validation checks
    total_imgs = len(df_potato)
    train_count = len(df_potato[df_potato['split'] == 'train'])
    val_count = len(df_potato[df_potato['split'] == 'val'])
    test_count = len(df_potato[df_potato['split'] == 'test'])
    
    missing = sum(not os.path.exists(p) for p in df_potato['image_path'])
    duplicates = df_potato['image_path'].duplicated().sum()
    has_non_potato = not all(df_potato['original_class'].isin(potato_classes))
    has_grayscale = df_potato['image_path'].str.contains('grayscale', case=False).any()
    has_segmented = df_potato['image_path'].str.contains('segmented', case=False).any()
    
    df_potato.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
    
    # 2. Compute Normalization
    print("Computing Normalization for Model 3 (Train split only)...")
    train_df = df_potato[df_potato['split'] == 'train'].reset_index(drop=True)
    norm_dataset = TempNormDataset(train_df)
    norm_loader = DataLoader(norm_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0
    
    for images in norm_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_pixels += batch_samples
        
    mean /= total_pixels
    std /= total_pixels
    
    norm_stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "number_of_images": total_pixels,
        "image_size": 224,
        "split": "train",
        "dsp_preprocessing": True
    }
    
    with open(norm_json, 'w') as f:
        json.dump(norm_stats, f, indent=4)
        
    print(f"Normalization saved to {norm_json}")
    
    # 3. Report Generation and DataLoader Verification
    report = []
    report.append("Model 3 (Potato) Dataset & Preprocessing Report")
    report.append("===============================================\n")
    
    report.append("Dataset Verification:")
    report.append(f"  Total Potato images: {total_imgs}")
    report.append(f"  Train: {train_count}")
    report.append(f"  Val: {val_count}")
    report.append(f"  Test: {test_count}")
    report.append(f"  Missing files: {missing}")
    report.append(f"  Duplicate paths: {duplicates}")
    report.append(f"  Non-Potato classes: {has_non_potato}")
    report.append(f"  Grayscale/Segmented images: {has_grayscale or has_segmented}\n")
    
    report.append("Class Distribution:")
    class_stats = {}
    for cl_name, (cl_label, cl_id) in label_map.items():
        sub_df = df_potato[df_potato['label_id'] == cl_id]
        c_tot = len(sub_df)
        c_train = len(sub_df[sub_df['split'] == 'train'])
        c_val = len(sub_df[sub_df['split'] == 'val'])
        c_test = len(sub_df[sub_df['split'] == 'test'])
        class_stats[cl_id] = {'name': cl_label, 'train': c_train}
        report.append(f"  [{cl_id}] {cl_label}: Total={c_tot} | Train={c_train} | Val={c_val} | Test={c_test}")
        
    report.append(f"\nNormalization Extracted:")
    report.append(f"  Mean: {norm_stats['mean']}")
    report.append(f"  Std: {norm_stats['std']}\n")
    
    max_train = max(class_stats[k]['train'] for k in class_stats)
    min_train = min(class_stats[k]['train'] for k in class_stats)
    imbalance_ratio = max_train / min_train if min_train > 0 else float('inf')
    
    report.append("Imbalance Analysis:")
    report.append(f"  Imbalance ratio (Max/Min): {imbalance_ratio:.1f}x")
    
    report.append("\n  Standard Inverse-Frequency Balanced Weights:")
    for k in class_stats:
        w = train_count / (3 * class_stats[k]['train']) if class_stats[k]['train'] > 0 else 0
        report.append(f"    {class_stats[k]['name']}: {w:.4f}")
        
    report.append("\n  Square-Root Inverse-Frequency Weights:")
    sqrt_w = {}
    for k in class_stats:
        sqrt_w[k] = math.sqrt(train_count / class_stats[k]['train']) if class_stats[k]['train'] > 0 else 0
    mean_sqrt = sum(sqrt_w.values()) / 3
    for k in class_stats:
        val = sqrt_w[k] / mean_sqrt
        report.append(f"    {class_stats[k]['name']}: {val:.4f}")
        
    report.append("\n  Recommended Mild Weighting Strategy:")
    report.append("    Due to the severe under-representation of the 'healthy' class, applying full inverse-frequency")
    report.append("    weighting could cause drastic over-prediction of healthy leaves, similar to Model 2's potato bias.")
    report.append("    Therefore, we recommend using Square-Root Inverse-Frequency weights, which scales the healthy class")
    report.append("    weight to ~1.7 instead of ~4.8, keeping the loss stable while providing necessary minority attention.\n")
    
    # 4. DataLoader Verification
    print("Testing dataloader...")
    from src.datasets.model3_potato_dataset import create_model3_potato_dataloaders
    train_loader, val_loader, _ = create_model3_potato_dataloaders(output_csv, batch_size=64, num_workers=0)
    
    it = iter(train_loader)
    imgs, lbls = next(it)
    
    shape = list(imgs.shape)
    has_nan = torch.isnan(imgs).any().item()
    has_inf = torch.isinf(imgs).any().item()
    unique_lbls = sorted(torch.unique(lbls).tolist())
    
    report.append("DataLoader Verification (Train Batch):")
    report.append(f"  Shape: {shape} (Expected: [64, 3, 224, 224] or smaller)")
    report.append(f"  Contains NaN: {has_nan}")
    report.append(f"  Contains Inf: {has_inf}")
    report.append(f"  Labels in batch: {unique_lbls} (Subset of [0, 1, 2])")
    
    valid = (len(shape) == 4 and shape[1:] == [3, 224, 224] and not has_nan and not has_inf and all(l in [0,1,2] for l in unique_lbls))
    
    if valid:
        report.append("  Status: PASSED")
        report.append("\nSTEP 23 POTATO METADATA COMPLETE")
    else:
        report.append("  Status: FAILED")
        
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    run()
