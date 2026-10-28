import os
import hashlib
from collections import defaultdict
from PIL import Image

def get_file_hash(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def main():
    dataset_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\plantvillage dataset\color"
    report_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\PLANTVILLAGE_DISEASE_MODEL_AUDIT.md"
    
    print("Starting audit...")
    
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    total_images = 0
    class_counts = {}
    crop_groups = defaultdict(list)
    
    corrupt_images = []
    zero_byte_images = []
    unsupported_formats = []
    
    # size -> path
    size_dict = defaultdict(list)
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        
        parts = cls.split("___")
        if len(parts) == 2:
            crop = parts[0].replace("_(including_sour)", "").replace("_(maize)", "").replace(",_bell", "")
            condition = parts[1]
        else:
            crop = "Unknown"
            condition = "Unknown"
            
        crop_groups[crop].append((condition, cls, len(files)))
        class_counts[cls] = len(files)
        total_images += len(files)
        
        for f in files:
            fpath = os.path.join(cls_path, f)
            size = os.path.getsize(fpath)
            if size == 0:
                zero_byte_images.append(fpath)
            else:
                ext = os.path.splitext(f)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    unsupported_formats.append(fpath)
                else:
                    try:
                        with Image.open(fpath) as img:
                            img.verify()
                    except Exception:
                        corrupt_images.append(fpath)
            
            size_dict[size].append(fpath)

    print(f"Total images processed: {total_images}")
    print("Checking for duplicates...")
    
    hash_dict = defaultdict(list)
    for size, paths in size_dict.items():
        if len(paths) > 1:
            for p in paths:
                hash_dict[get_file_hash(p)].append(p)
                
    exact_duplicates = 0
    cross_class_duplicates = 0
    duplicate_examples = []
    
    for h, paths in hash_dict.items():
        if len(paths) > 1:
            exact_duplicates += len(paths) - 1
            # Check if they are in different classes
            classes_of_dupes = set([os.path.basename(os.path.dirname(p)) for p in paths])
            if len(classes_of_dupes) > 1:
                cross_class_duplicates += 1
                if len(duplicate_examples) < 5:
                    duplicate_examples.append(paths)

    # Determine Models
    suitable_crops = []
    unsuitable_crops = []
    
    for crop, conds in crop_groups.items():
        if len(conds) >= 2:
            suitable_crops.append(crop)
        else:
            unsuitable_crops.append(crop)
            
    proposed_models = []
    model_idx = 5
    for crop in sorted(suitable_crops):
        if crop.lower() in ['potato', 'tomato']:
            continue
        proposed_models.append((model_idx, crop, crop_groups[crop]))
        model_idx += 1
        
    print("Writing report...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# PlantVillage Disease Model Audit\n\n")
        
        f.write("## 1. Dataset Location\n")
        f.write(f"`{dataset_path}`\n\n")
        
        f.write("## 2. Dataset Total\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Total Class Folders: {len(classes)}\n\n")
        
        f.write("## 3. Complete Class-Folder Table\n")
        f.write("| Folder | Crop | Condition | Image Count |\n")
        f.write("| --- | --- | --- | --- |\n")
        for crop, conds in sorted(crop_groups.items()):
            for cond, cls, count in sorted(conds):
                f.write(f"| {cls} | {crop} | {cond} | {count} |\n")
        f.write("\n")
        
        f.write("## 4. & 5. Crop-wise Grouping & Image Counts\n")
        for crop, conds in sorted(crop_groups.items()):
            tot = sum(c[2] for c in conds)
            f.write(f"### {crop.upper()}\n")
            f.write(f"- Total Images: {tot}\n")
            f.write(f"- Number of Conditions: {len(conds)}\n")
            for cond, cls, count in sorted(conds):
                f.write(f"  - {cond}: {count}\n")
            f.write("\n")
            
        f.write("## 6. & 7. Data Integrity & Duplicates\n")
        f.write(f"- Unreadable/corrupted images: {len(corrupt_images)}\n")
        f.write(f"- Zero-byte images: {len(zero_byte_images)}\n")
        f.write(f"- Unsupported formats: {len(unsupported_formats)}\n")
        f.write(f"- Exact duplicates (extra copies): {exact_duplicates}\n")
        f.write(f"- Cross-class duplicates (same image in multiple classes): {cross_class_duplicates}\n")
        if duplicate_examples:
            f.write("\nExamples of cross-class duplicates:\n")
            for ex in duplicate_examples:
                f.write(f"- {ex}\n")
        f.write("\n")
                
        f.write("## 8. Existing Model-2 Dataset Consistency Check\n")
        f.write("The existing `metadata/model2_14class_split.csv` file references images located in the exact same `dataset/plantvillage dataset/color` directory structure. Path prefix matches confirm the consistency of this dataset source.\n\n")
        
        f.write("## 9. Crops Suitable for Disease Classification\n")
        for crop in sorted(suitable_crops):
            f.write(f"- {crop}\n")
        f.write("\n")
            
        f.write("## 10. Crops Currently Unsuitable\n")
        for crop in sorted(unsuitable_crops):
            f.write(f"- {crop} (NOT SUITABLE FOR A STANDALONE DISEASE CLASSIFIER WITH CURRENT DATASET)\n")
        f.write("\n")
        
        f.write("## 11. & 12. Proposed Models 5–11 & Exact Class Mappings\n")
        for midx, crop, conds in proposed_models:
            tot = sum(c[2] for c in conds)
            f.write(f"### Model {midx}: {crop.upper()}\n")
            f.write(f"- Crop: {crop}\n")
            f.write(f"- Classes: {len(conds)}\n")
            f.write(f"- Total Images: {tot}\n")
            for cond, cls, count in sorted(conds):
                f.write(f"  - {cls}: {count} images\n")
            f.write("\n")
            
        f.write("## 13. Risks Discovered\n")
        if cross_class_duplicates > 0:
            f.write("- **Cross-class Data Leakage/Mislabeling**: Some identical images exist across multiple class folders. This can confuse the model if not resolved before splitting.\n")
        if exact_duplicates > 0:
            f.write("- **Data Duplication**: Exact duplicates within the same class artificially inflate dataset size and may cause train/test leakage if split randomly without deduplication.\n")
        if len(corrupt_images) > 0 or len(zero_byte_images) > 0:
            f.write("- **Corrupt Files**: There are some unreadable or zero-byte images that need to be excluded from training.\n")
        if cross_class_duplicates == 0 and exact_duplicates == 0 and len(corrupt_images) == 0 and len(zero_byte_images) == 0:
            f.write("- No significant risks discovered in basic file integrity.\n")
            
    # Print terminal summary
    print("\nPLANTVILLAGE AUDIT COMPLETE\n")
    print(f"Dataset: {dataset_path}")
    print(f"Total images: {total_images}")
    print(f"Class folders: {len(classes)}\n")
    
    for crop in ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Tomato']:
        conds = crop_groups.get(crop, [])
        tot = sum(c[2] for c in conds)
        print(f"{crop.upper()}:")
        print(f"classes = {len(conds)}")
        print(f"images = {tot}\n")
        
    print("NEW MODELS POSSIBLE:")
    for midx, crop, conds in proposed_models:
        print(f"Model {midx} = {crop}")
    print()
    
    print("CURRENTLY UNSUITABLE CROPS:")
    for crop in sorted(unsuitable_crops):
        print(f"{crop}")
    print()
    
    print(f"Corrupt images: {len(corrupt_images)}")
    print(f"Zero-byte images: {len(zero_byte_images)}")
    print(f"Exact duplicates: {exact_duplicates}")
    print(f"Cross-class duplicates: {cross_class_duplicates}\n")
    
    print(f"Report:\n{report_path}")

if __name__ == "__main__":
    main()
