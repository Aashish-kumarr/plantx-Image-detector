import os
import shutil
import tarfile

categories_file = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\categories.txt"
val_mapping_file = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\places365_val.txt"
tar_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\places_val_256.tar"
extract_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\val_256"
output_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\places365_non_leaf"

# 1. Read categories and filter
botanical_keywords = [
    "forest", "garden", "farm", "field", "tree", "flower", "orchard", "greenhouse", 
    "bush", "grass", "botanical", "vegetation", "plant", "yard", "park", "pasture", 
    "lawn", "marsh", "swamp", "vineyard", "wood", "bamboo", "hayfield", "paddy", "campsite",
    "golf_course", "lagoon", "lake", "ocean", "river", "beach", "coast", "pond", "creek",
    "valley", "mountain", "tundra", "desert/vegetation", "snowfield", "iceberg", "glacier"
]
# Wait, water/mountain/ice are not plants, but might have vegetation or aren't man-made/indoor. 
# The user asked for "diverse indoor and man-made environments such as rooms, buildings, transportation environments, roads, industrial environments, sports environments, and other clearly non-botanical scenes."
# It's safest to exclude natural landscapes to avoid accidental plants (like grass on mountains or bushes by a river).
natural_landscape_keywords = [
    "lagoon", "lake", "ocean", "river", "beach", "coast", "pond", "creek",
    "valley", "mountain", "tundra", "desert/vegetation", "snowfield", "iceberg", "glacier",
    "canyon", "cliff", "crevasse", "badlands", "butte", "marsh", "swamp", "waterfall"
]
all_excluded_keywords = botanical_keywords + natural_landscape_keywords

kept_categories = {}
excluded_categories = []

with open(categories_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split()
        cat_name = parts[0]
        cat_idx = int(parts[1])
        
        # Check exclusion
        exclude = False
        for kw in all_excluded_keywords:
            if kw in cat_name:
                exclude = True
                break
        
        if exclude:
            excluded_categories.append(cat_name)
        else:
            # We just want ~300 max, but we'll take all safe ones (should be ~260-300).
            kept_categories[cat_idx] = cat_name

print(f"Total categories read: {len(kept_categories) + len(excluded_categories)}")
print(f"Categories kept (safe non-leaf): {len(kept_categories)}")
print(f"Categories excluded: {len(excluded_categories)}")

# Ensure we limit to 300 if we have more, to stay around 30k images
# (Places365 val has 100 images per category)
max_categories = min(len(kept_categories), 300)
kept_indices = list(kept_categories.keys())[:max_categories]

# 2. Extract tar file
print("Extracting tar file...")
with tarfile.open(tar_path, "r") as t:
    t.extractall(path=r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset")

# 3. Read mapping and copy files
print("Copying selected images to new dataset structure...")
os.makedirs(output_dir, exist_ok=True)

images_copied = 0
failed_copies = 0
cat_counts = {}

with open(val_mapping_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        img_name, cat_idx_str = line.split()
        cat_idx = int(cat_idx_str)
        
        if cat_idx in kept_indices:
            cat_name = kept_categories[cat_idx]
            # Remove leading slash and replace slashes with underscores for folder name
            clean_cat_name = cat_name.lstrip('/').replace('/', '_')
            
            cat_dir = os.path.join(output_dir, clean_cat_name)
            os.makedirs(cat_dir, exist_ok=True)
            
            src_path = os.path.join(extract_dir, img_name)
            dst_path = os.path.join(cat_dir, img_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                images_copied += 1
                cat_counts[clean_cat_name] = cat_counts.get(clean_cat_name, 0) + 1
            else:
                failed_copies += 1

# 4. Clean up original tar and extracted dir
print("Cleaning up...")
shutil.rmtree(extract_dir, ignore_errors=True)
os.remove(tar_path)

# 5. Output report
print("--- REPORT ---")
print(f"SOURCE: http://data.csail.mit.edu/places/places365/val_256.tar")
print(f"CATEGORIES SELECTED: {len(cat_counts)}")
print(f"IMAGES PER CATEGORY: ~100")
print(f"TOTAL DOWNLOADED IMAGES: {images_copied}")
print(f"FAILED/CORRUPTED: {failed_copies}")

import ctypes
def get_free_space_gb(dirname):
    free_bytes = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
    return free_bytes.value / (1024**3)

dir_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(output_dir) for filename in filenames)
print(f"DATASET SIZE ON DISK: {dir_size / (1024**2):.2f} MB")
print(f"REMAINING FREE DISK SPACE: {get_free_space_gb('c:/'):.2f} GB")
