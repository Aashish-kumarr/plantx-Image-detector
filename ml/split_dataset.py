import os
import csv
import hashlib
import random
from collections import defaultdict

base_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
master_csv = os.path.join(base_dir, "metadata", "model1_master.csv")
split_csv = os.path.join(base_dir, "metadata", "model1_split.csv")
report_txt = os.path.join(base_dir, "reports", "model1_split_report.txt")

# Set seed
random.seed(42)

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

print("Reading master CSV and computing hashes...")
data = []
# data item format: {'path':..., 'label':..., 'label_id':..., 'source':..., 'class':..., 'hash':...}
hash_groups = defaultdict(list)

with open(master_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        filepath = row["image_path"]
        file_hash = get_hash(filepath)
        item = {
            "path": filepath,
            "label": row["label"],
            "label_id": row["label_id"],
            "source": row["source"],
            "original_class": row["original_class"],
            "hash": file_hash
        }
        data.append(item)
        hash_groups[file_hash].append(item)

print(f"Total images read: {len(data)}")
print(f"Total unique hashes: {len(hash_groups)}")

# Group by stratum
# stratum key: (label, source)
strata_groups = defaultdict(list)
for h, items in hash_groups.items():
    # If duplicates cross strata (very rare), assign group based on the first item
    stratum = (items[0]["label"], items[0]["source"])
    strata_groups[stratum].append(items) # list of lists

splits = {"train": [], "val": [], "test": []}

print("Splitting...")
for stratum, groups in strata_groups.items():
    # Shuffle groups within stratum
    random.shuffle(groups)
    
    total_images_in_stratum = sum(len(g) for g in groups)
    target_train = int(total_images_in_stratum * 0.8)
    target_val = int(total_images_in_stratum * 0.1)
    
    train_count = 0
    val_count = 0
    
    for g in groups:
        g_len = len(g)
        # Assign to train if we haven't reached target
        if train_count + g_len <= target_train or train_count < target_train:
            splits["train"].extend(g)
            train_count += g_len
        elif val_count + g_len <= target_val or val_count < target_val:
            splits["val"].extend(g)
            val_count += g_len
        else:
            splits["test"].extend(g)

# Write split CSV
print("Writing split CSV...")
out_rows = []
for split_name, items in splits.items():
    for item in items:
        out_rows.append([
            item["path"], item["label"], item["label_id"], item["source"], item["original_class"], split_name
        ])

with open(split_csv, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label", "label_id", "source", "original_class", "split"])
    writer.writerows(out_rows)

print("Validating and generating report...")
# Validation
val_data = []
with open(split_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        val_data.append(row)

# Assertions
assert len(val_data) == len(data), "Assertion failed: Image count mismatch between master and split CSV."
paths_master = set(d["path"] for d in data)
paths_split = set(d["image_path"] for d in val_data)
assert paths_master == paths_split, "Assertion failed: Missing or mismatched paths in split CSV."

split_hashes = {"train": set(), "val": set(), "test": set()}
stats = {
    "train": {"total": 0, "leaf": 0, "non_leaf": 0, "pv": 0, "caltech": 0, "places": 0},
    "val": {"total": 0, "leaf": 0, "non_leaf": 0, "pv": 0, "caltech": 0, "places": 0},
    "test": {"total": 0, "leaf": 0, "non_leaf": 0, "pv": 0, "caltech": 0, "places": 0}
}
hash_to_split = {}
crossing_groups = 0

# Precompute path to hash mapping to avoid O(N^2) loop
path_to_hash = {d["path"]: d["hash"] for d in data}

for row in val_data:
    assert row["split"] in ["train", "val", "test"], f"Assertion failed: Invalid split value {row['split']}"
    assert (row["label"] == "leaf" and row["label_id"] == "1") or (row["label"] == "non_leaf" and row["label_id"] == "0"), "Assertion failed: Label mismatch"
    
    s = row["split"]
    stats[s]["total"] += 1
    if row["label"] == "leaf":
        stats[s]["leaf"] += 1
    else:
        stats[s]["non_leaf"] += 1
        
    if row["source"] == "plantvillage":
        stats[s]["pv"] += 1
    elif row["source"] == "caltech256":
        stats[s]["caltech"] += 1
    elif row["source"] == "places365":
        stats[s]["places"] += 1
        
    h = path_to_hash[row["image_path"]]
    if h in hash_to_split:
        if hash_to_split[h] != s:
            crossing_groups += 1
    else:
        hash_to_split[h] = s

assert crossing_groups == 0, "Assertion failed: Identical SHA-256 content occurred across multiple splits."

# Generate report
total_images = len(val_data)
duplicate_groups = len(data) - len(hash_groups)

with open(report_txt, "w", encoding="utf-8") as f:
    f.write("Model 1 Split Validation Report\n")
    f.write("===============================\n")
    f.write(f"Total images: {total_images}\n")
    
    for s in ["train", "val", "test"]:
        pct = stats[s]['total'] / total_images * 100
        f.write(f"{s.capitalize()} count and percentage: {stats[s]['total']} ({pct:.2f}%)\n")
        
    f.write("\nLeaf/Non-Leaf counts for each split:\n")
    for s in ["train", "val", "test"]:
        f.write(f"  {s.capitalize()}: Leaf={stats[s]['leaf']}, Non-Leaf={stats[s]['non_leaf']}\n")
        
    f.write("\nPlantVillage count for each split:\n")
    for s in ["train", "val", "test"]:
        f.write(f"  {s.capitalize()}: {stats[s]['pv']}\n")
        
    f.write("\nCaltech-256 count for each split:\n")
    for s in ["train", "val", "test"]:
        f.write(f"  {s.capitalize()}: {stats[s]['caltech']}\n")
        
    f.write("\nPlaces365 count for each split:\n")
    for s in ["train", "val", "test"]:
        f.write(f"  {s.capitalize()}: {stats[s]['places']}\n")
        
    f.write(f"\nDuplicate-content groups detected: {duplicate_groups}\n")
    f.write(f"Number of duplicate-content groups crossing splits: {crossing_groups}\n")
    f.write(f"Missing paths: {len(paths_master) - len(paths_split)}\n")
    f.write(f"Duplicate CSV rows: {len(val_data) - len(set(d['image_path'] for d in val_data))}\n")

print("All assertions passed. Report generated.")
