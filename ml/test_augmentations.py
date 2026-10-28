import os
import csv
import random
import torch
import torchvision.transforms as transforms
from PIL import Image

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_transforms import get_model1_train_transform, get_model1_eval_transform

random.seed(42)
torch.manual_seed(42)

split_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_split.csv"
reports_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports"
samples_dir = os.path.join(reports_dir, "model1_augmentation_samples")
report_txt = os.path.join(reports_dir, "model1_augmentation_report.txt")

os.makedirs(samples_dir, exist_ok=True)

# 1. Collect samples (5 PV, 5 Caltech, 5 Places365)
pv_pool, caltech_pool, places_pool = [], [], []

with open(split_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row["source"]
        if src == "plantvillage": pv_pool.append(row["image_path"])
        elif src == "caltech256": caltech_pool.append(row["image_path"])
        elif src == "places365": places_pool.append(row["image_path"])

samples = random.sample(pv_pool, 5) + random.sample(caltech_pool, 5) + random.sample(places_pool, 5)

train_t = get_model1_train_transform()
eval_t = get_model1_eval_transform()

# To extract just the PIL-level transforms for saving visual samples
visual_train_t = transforms.Compose([t for t in train_t.transforms if not isinstance(t, (transforms.ToTensor, transforms.Normalize))])
visual_eval_t = transforms.Compose([t for t in eval_t.transforms if not isinstance(t, (transforms.ToTensor, transforms.Normalize))])

report = []
report.append("Model 1 Augmentation Report")
report.append("===========================")
report.append("Augmentations configured:")
report.append("  - RandomHorizontalFlip (p=0.5)")
report.append("  - RandomAffine:")
report.append("      Degrees: ±15")
report.append("      Translate: ±5% (0.05)")
report.append("      Scale: 95% - 105% (0.95-1.05)")
report.append("  - ColorJitter:")
report.append("      Brightness: ±10% (0.1)")
report.append("      Contrast: ±10% (0.1)")
report.append(f"Number of samples inspected: {len(samples)}")
report.append(f"Visual samples saved to: {samples_dir}")
report.append("\nTensor Validation Results:\n")

failures = 0
deterministic_failures = 0
variation_failures = 0

for i, path in enumerate(samples):
    basename = os.path.basename(path)
    prefix = f"{i:02d}_{basename}"
    
    try:
        img_pil = Image.open(path).convert('RGB')
        
        # --- 1. Visual Samples ---
        # Eval (Baseline DSP)
        base_img = visual_eval_t(img_pil)
        base_img.save(os.path.join(samples_dir, f"{prefix}_eval_base.jpg"))
        
        # Train Augmentations (Save 3 variations)
        for v in range(1, 4):
            aug_img = visual_train_t(img_pil)
            aug_img.save(os.path.join(samples_dir, f"{prefix}_train_aug{v}.jpg"))
            
        # --- 2. Tensor Validation ---
        eval_t1 = eval_t(img_pil)
        eval_t2 = eval_t(img_pil)
        
        train_t1 = train_t(img_pil)
        train_t2 = train_t(img_pil)
        
        shape = list(eval_t1.shape)
        is_float = eval_t1.dtype == torch.float32
        has_nan = torch.isnan(eval_t1).any().item() or torch.isnan(train_t1).any().item()
        has_inf = torch.isinf(eval_t1).any().item() or torch.isinf(train_t1).any().item()
        
        # Check determinism on eval
        is_deterministic = torch.equal(eval_t1, eval_t2)
        # Check variation on train (highly likely to be different due to RandomAffine/Flip)
        is_varying = not torch.equal(train_t1, train_t2)
        
        passed = (shape == [3, 224, 224]) and is_float and not has_nan and not has_inf
        
        report.append(f"File: {basename}")
        report.append(f"  Shape (3x224x224 expected): {shape}")
        report.append(f"  Dtype (float32 expected): {eval_t1.dtype}")
        report.append(f"  Valid Tensor (No NaN/Inf): {not (has_nan or has_inf)}")
        report.append(f"  Eval Deterministic: {is_deterministic}")
        report.append(f"  Train Varies: {is_varying}")
        
        if not passed:
            failures += 1
        if not is_deterministic:
            deterministic_failures += 1
        if not is_varying:
            variation_failures += 1
            
    except Exception as e:
        report.append(f"File: {basename} - FAILED: {e}")
        failures += 1

report.append("\nSummary:")
report.append(f"Tensor extraction failures: {failures}")
report.append(f"Determinism failures (Eval): {deterministic_failures}")
report.append(f"Variation failures (Train): {variation_failures}")

with open(report_txt, 'w', encoding='utf-8') as f:
    f.write("\n".join(report))
    
print("Validation complete.")
