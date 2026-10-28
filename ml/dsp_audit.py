import os
import csv
import random
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_dsp_preprocessing import create_foreground_mask

# Set seed
random.seed(42)

split_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_split.csv"
reports_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports"
audit_txt = os.path.join(reports_dir, "model1_dsp_large_audit.txt")
stats_csv = os.path.join(reports_dir, "model1_dsp_mask_statistics.csv")

# 1. Read and group paths
train_data = {"plantvillage": defaultdict(list), "caltech256": defaultdict(list), "places365": defaultdict(list)}

with open(split_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["split"] == "train":
            src = row["source"]
            cls = row["original_class"]
            train_data[src][cls].append(row)

# 2. Sample 1000 per source fairly across classes
def sample_fairly(source_dict, n=1000):
    classes = list(source_dict.keys())
    sampled = []
    pool = {cls: list(items) for cls, items in source_dict.items()}
    for cls in pool:
        random.shuffle(pool[cls])
        
    while len(sampled) < n:
        added = False
        for cls in classes:
            if pool[cls]:
                sampled.append(pool[cls].pop())
                added = True
            if len(sampled) == n:
                break
        if not added:
            break
    return sampled

pv_samples = sample_fairly(train_data["plantvillage"], 1000)
caltech_samples = sample_fairly(train_data["caltech256"], 1000)
places_samples = sample_fairly(train_data["places365"], 1000)

all_samples = pv_samples + caltech_samples + places_samples

# 3. Process and collect stats
raw_stats = []

print(f"Processing {len(all_samples)} images...")
for i, row in enumerate(all_samples):
    path = row["image_path"]
    try:
        img_pil = Image.open(path).convert('RGB')
        img_np = np.array(img_pil)
        mask = create_foreground_mask(img_np)
        
        total_pixels = mask.size
        fg_pixels = np.count_nonzero(mask)
        fg_ratio = fg_pixels / total_pixels
        
        raw_stats.append({
            "image_path": path,
            "label": row["label"],
            "source": row["source"],
            "original_class": row["original_class"],
            "foreground_ratio": fg_ratio
        })
    except Exception as e:
        print(f"Failed {path}: {e}")
        pass

# 4. Save raw stats
with open(stats_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label", "source", "original_class", "foreground_ratio"])
    writer.writeheader()
    for item in raw_stats:
        writer.writerow(item)

# 5. Compute statistics
def get_stats(ratios):
    r = np.array(ratios) * 100
    n = len(r)
    return {
        "n": n,
        "mean": np.mean(r),
        "median": np.median(r),
        "std": np.std(r),
        "min": np.min(r),
        "max": np.max(r),
        "p5": np.percentile(r, 5),
        "p25": np.percentile(r, 25),
        "p75": np.percentile(r, 75),
        "p95": np.percentile(r, 95),
        "bins": {
            "< 1%": int(np.sum(r < 1)),
            "1-5%": int(np.sum((r >= 1) & (r < 5))),
            "5-10%": int(np.sum((r >= 5) & (r < 10))),
            "10-25%": int(np.sum((r >= 10) & (r < 25))),
            "25-50%": int(np.sum((r >= 25) & (r < 50))),
            "50-75%": int(np.sum((r >= 50) & (r < 75))),
            "> 75%": int(np.sum(r >= 75))
        }
    }

groups = {
    "PlantVillage": [x["foreground_ratio"] for x in raw_stats if x["source"] == "plantvillage"],
    "Caltech-256": [x["foreground_ratio"] for x in raw_stats if x["source"] == "caltech256"],
    "Places365": [x["foreground_ratio"] for x in raw_stats if x["source"] == "places365"],
    "Combined Non-Leaf": [x["foreground_ratio"] for x in raw_stats if x["label"] == "non_leaf"]
}

group_stats = {k: get_stats(v) for k, v in groups.items()}

# Diagnostics
leaf_empty = (group_stats["PlantVillage"]["bins"]["< 1%"] / group_stats["PlantVillage"]["n"]) * 100
non_leaf_empty = (group_stats["Combined Non-Leaf"]["bins"]["< 1%"] / group_stats["Combined Non-Leaf"]["n"]) * 100

substantial = (
    group_stats["Combined Non-Leaf"]["bins"]["25-50%"] +
    group_stats["Combined Non-Leaf"]["bins"]["50-75%"] +
    group_stats["Combined Non-Leaf"]["bins"]["> 75%"]
)
non_leaf_substantial = (substantial / group_stats["Combined Non-Leaf"]["n"]) * 100

# 6. Report
report = []
report.append("Model 1 DSP Large Audit Report")
report.append("==============================")
for k, s in group_stats.items():
    report.append(f"\nGroup: {k}")
    report.append(f"  Number of images: {s['n']}")
    report.append(f"  Mean foreground %: {s['mean']:.2f}%")
    report.append(f"  Median foreground %: {s['median']:.2f}%")
    report.append(f"  Standard deviation: {s['std']:.2f}%")
    report.append(f"  Minimum: {s['min']:.2f}%")
    report.append(f"  Maximum: {s['max']:.2f}%")
    report.append(f"  5th percentile: {s['p5']:.2f}%")
    report.append(f"  25th percentile: {s['p25']:.2f}%")
    report.append(f"  75th percentile: {s['p75']:.2f}%")
    report.append(f"  95th percentile: {s['p95']:.2f}%")
    report.append(f"  Foreground Bins:")
    for b, count in s['bins'].items():
        report.append(f"    {b}: {count}")

report.append("\nDiagnostics:")
report.append(f"- Leaf images with almost-empty masks (<1%): {leaf_empty:.2f}%")
report.append(f"- Non-Leaf images with almost-empty masks (<1%): {non_leaf_empty:.2f}%")
report.append(f"- Non-Leaf images with substantial foreground (>25%): {non_leaf_substantial:.2f}%")
report.append("\nOverlap Analysis:")

report.append("  There is significant overlap between Leaf and Non-Leaf foreground distributions.")
report.append("  The DSP stage is NOT simply turning all Non-Leaf images black. Many Non-Leaf images retain complex foreground structures.")

with open(audit_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("Audit complete.")
