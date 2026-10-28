import os
import csv
import hashlib
from PIL import Image

# Setup paths
base_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
pv_dir = os.path.join(base_dir, r"dataset\plantvillage dataset\color")
caltech_dir = os.path.join(base_dir, r"dataset\caltech256")
# Check if caltech is under 256_ObjectCategories
if os.path.exists(os.path.join(caltech_dir, "256_ObjectCategories")):
    caltech_dir = os.path.join(caltech_dir, "256_ObjectCategories")
places_dir = os.path.join(base_dir, r"dataset\places365_non_leaf")

metadata_dir = os.path.join(base_dir, "metadata")
reports_dir = os.path.join(base_dir, "reports")
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

csv_out = os.path.join(metadata_dir, "model1_master.csv")
report_out = os.path.join(reports_dir, "model1_metadata_report.txt")

# Excluded Caltech classes
excluded_caltech = {
    "015.bonsai-101", "025.cactus", "068.fern", "092.grapes", "103.hibiscus", 
    "118.iris", "147.mushroom", "154.palm-tree", "204.sunflower-101", "221.tomato", 
    "242.watermelon", "257.clutter", "024.butterfly", "093.grasshopper", 
    "113.hummingbird", "166.praying-mantis", "189.snail"
}

# Stats
stats = {
    "total_images": 0,
    "total_leaf": 0,
    "total_non_leaf": 0,
    "pv_count": 0,
    "caltech_count": 0,
    "places_count": 0,
    "missing_files": 0,
    "unreadable_files": 0,
    "duplicate_paths": 0,
    "duplicate_content": 0,
    "forbidden_pv_found": 0,
    "excluded_caltech_found": 0
}

all_paths = set()
all_hashes = set()
csv_rows = []

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def process_dataset(root_dir, label, label_id, source, class_check_fn=None):
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        if class_check_fn:
            class_check_fn(class_name, source)
            # If it's a skipped class (for Caltech), we don't process its images
            if source == "caltech256" and class_name in excluded_caltech:
                continue

        for file_name in os.listdir(class_path):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            file_path = os.path.join(class_path, file_name)
            
            # Additional checks for forbidden leakage
            if source == "plantvillage":
                if "grayscale" in file_path.lower() or "segmented" in file_path.lower():
                    stats["forbidden_pv_found"] += 1
                    continue
            if source == "caltech256":
                if class_name in excluded_caltech:
                    stats["excluded_caltech_found"] += 1
                    continue

            if not os.path.exists(file_path):
                stats["missing_files"] += 1
                continue
                
            if file_path in all_paths:
                stats["duplicate_paths"] += 1
                continue
                
            all_paths.add(file_path)
            
            # Verify image
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                stats["unreadable_files"] += 1
                continue
                
            # Check duplicate content
            file_hash = get_hash(file_path)
            if file_hash in all_hashes:
                stats["duplicate_content"] += 1
            all_hashes.add(file_hash)
            
            csv_rows.append([file_path, label, label_id, source, class_name])
            
            # Update stats
            stats["total_images"] += 1
            if label_id == 1:
                stats["total_leaf"] += 1
                stats["pv_count"] += 1
            else:
                stats["total_non_leaf"] += 1
                if source == "caltech256":
                    stats["caltech_count"] += 1
                elif source == "places365":
                    stats["places_count"] += 1

print("Processing PlantVillage...")
process_dataset(pv_dir, "leaf", 1, "plantvillage")

print("Processing Caltech-256...")
def caltech_check(cname, src):
    pass
process_dataset(caltech_dir, "non_leaf", 0, "caltech256", caltech_check)

print("Processing Places365...")
process_dataset(places_dir, "non_leaf", 0, "places365")

print("Writing CSV...")
with open(csv_out, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label", "label_id", "source", "original_class"])
    writer.writerows(csv_rows)

print("Generating report...")
with open(report_out, "w", encoding='utf-8') as f:
    f.write("Model 1 Metadata Validation Report\n")
    f.write("==================================\n")
    f.write(f"Total images: {stats['total_images']}\n")
    f.write(f"Total Leaf images: {stats['total_leaf']}\n")
    f.write(f"Total Non-Leaf images: {stats['total_non_leaf']}\n")
    if stats['total_images'] > 0:
        f.write(f"Leaf/Non-Leaf percentage: {stats['total_leaf']/stats['total_images']*100:.2f}% / {stats['total_non_leaf']/stats['total_images']*100:.2f}%\n")
    else:
        f.write("Leaf/Non-Leaf percentage: 0% / 0%\n")
    f.write(f"PlantVillage image count: {stats['pv_count']}\n")
    f.write(f"Caltech-256 image count: {stats['caltech_count']}\n")
    f.write(f"Places365 image count: {stats['places_count']}\n")
    f.write(f"Missing files: {stats['missing_files']}\n")
    f.write(f"Unreadable files: {stats['unreadable_files']}\n")
    f.write(f"Duplicate paths: {stats['duplicate_paths']}\n")
    f.write(f"Exact duplicate image-content count: {stats['duplicate_content']}\n")
    f.write(f"Forbidden PlantVillage versions found: {stats['forbidden_pv_found']}\n")
    f.write(f"Excluded Caltech categories found: {stats['excluded_caltech_found']}\n")

print("Done!")
