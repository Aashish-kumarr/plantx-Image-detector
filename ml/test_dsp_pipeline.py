import os
import csv
import random
import torch
import numpy as np
import cv2
from PIL import Image

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_dsp_preprocessing import create_foreground_mask, apply_dsp_preprocessing
from torchvision.transforms import ToTensor

# Set seed for reproducibility
random.seed(42)

split_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_split.csv"
reports_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports"
samples_dir = os.path.join(reports_dir, "model1_dsp_samples")
report_txt = os.path.join(reports_dir, "model1_dsp_preprocessing_report.txt")

os.makedirs(samples_dir, exist_ok=True)

# Collect paths
pv_pool = []
caltech_pool = []
places_pool = []

with open(split_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row["source"]
        if src == "plantvillage":
            pv_pool.append(row["image_path"])
        elif src == "caltech256":
            caltech_pool.append(row["image_path"])
        elif src == "places365":
            places_pool.append(row["image_path"])

# Sample
pv_samples = random.sample(pv_pool, 20)
caltech_samples = random.sample(caltech_pool, 10)
places_samples = random.sample(places_pool, 10)

all_samples = pv_samples + caltech_samples + places_samples

report_lines = []
report_lines.append("Model 1 DSP Preprocessing Validation Report")
report_lines.append("===========================================")
report_lines.append("DSP operations implemented: RGB->HSV, Gaussian filtering, Otsu thresholding, Morphology (Open/Close), Largest Blob Extraction")
report_lines.append("HSV channel used for mask generation: S (Saturation) channel")
report_lines.append("Gaussian filter settings: Kernel Size = (5, 5), Sigma = 0 (auto)")
report_lines.append("Thresholding method: Otsu's binarization")
report_lines.append("Morphology kernel settings: Ellipse, Size = (5, 5), Iterations = 2 (Open), 2 (Close)")
report_lines.append(f"Number of images tested: {len(all_samples)}")
report_lines.append(f"Paths to saved DSP sample images: {samples_dir}")
report_lines.append("\nImage Test Results:\n")

failures = 0
to_tensor = ToTensor()

for i, path in enumerate(all_samples):
    basename = os.path.basename(path)
    prefix = f"{i:02d}_{basename}"
    
    try:
        # Load Original
        original_pil = Image.open(path).convert("RGB")
        original_np = np.array(original_pil)
        
        # 1. Mask stats and generation
        mask = create_foreground_mask(original_np)
        
        # Stats
        total_pixels = mask.size
        fg_pixels = np.count_nonzero(mask)
        fg_percentage = (fg_pixels / total_pixels) * 100
        
        # 2. Apply full pipeline
        final_pil = apply_dsp_preprocessing(original_pil)
        
        # Convert to tensor to check NaNs and Infs and Shape
        final_tensor = to_tensor(final_pil)
        shape = list(final_tensor.shape)
        is_float = final_tensor.dtype == torch.float32
        has_nan = torch.isnan(final_tensor).any().item()
        has_inf = torch.isinf(final_tensor).any().item()
        
        # Save samples
        # Save Original
        original_pil.save(os.path.join(samples_dir, f"{prefix}_original.jpg"))
        # Save Mask
        Image.fromarray(mask).save(os.path.join(samples_dir, f"{prefix}_mask.png"))
        # Save Final
        final_pil.save(os.path.join(samples_dir, f"{prefix}_final_dsp.jpg"))
        
        # Check success
        passed = (shape == [3, 224, 224]) and is_float and not has_nan and not has_inf
        
        report_lines.append(f"File: {basename}")
        report_lines.append(f"  Shape/Validity Pass: {passed}")
        report_lines.append(f"  Foreground-mask percentage: {fg_percentage:.2f}%")
        if not passed:
            report_lines.append(f"  Details - Shape: {shape}, NaNs: {has_nan}, Infs: {has_inf}")
            failures += 1
            
    except Exception as e:
        report_lines.append(f"File: {basename}")
        report_lines.append(f"  FAILED with exception: {e}")
        failures += 1

report_lines.append(f"\nTotal failures: {failures}")

with open(report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("DSP Validation completed.")
