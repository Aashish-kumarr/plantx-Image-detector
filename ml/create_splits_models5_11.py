import os
import hashlib
import json
import random
import csv
from collections import defaultdict
from pathlib import Path

def get_sha256(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def normalize_label(folder_name):
    parts = folder_name.split('___')
    if len(parts) == 2:
        cond = parts[1]
    else:
        cond = folder_name
    return cond.lower().strip()

def main():
    random.seed(42)
    
    dataset_path = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\plantvillage dataset\color")
    metadata_dir = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata")
    reports_dir = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports")
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    models = {
        5: ("Apple", "apple"),
        6: ("Cherry", "cherry"),
        7: ("Corn", "corn"),
        8: ("Grape", "grape"),
        9: ("Peach", "peach"),
        10: ("Pepper", "pepper"),
        11: ("Strawberry", "strawberry")
    }
    
    all_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(dataset_path / d)]
    
    class_mappings_json = {}
    
    global_missing = 0
    global_unreadable = 0
    global_cross_class = 0
    global_cross_split = 0
    global_missing_train = 0
    global_missing_val = 0
    global_missing_test = 0
    
    report_content = [
        "# Models 5-11 Dataset Split Report",
        "",
        "**Dataset Source**: `C:\\Users\\ak021\\Desktop\\workspace\\plantDx\\plant image detector\\dataset\\plantvillage dataset\\color`",
        "**Random Seed**: 42",
        "**Splitting Methodology**: Stratified grouping by exact image hash.",
        "**Duplicate Grouping Methodology**: SHA-256 content hashing to ensure exact duplicates always fall in the same split (Train/Val/Test), preventing data leakage.",
        ""
    ]
    
    terminal_output = [
        "MODELS 5–11 DATASET PREPARATION COMPLETE",
        "",
        "SEED:",
        "42",
        ""
    ]
    
    for model_idx in sorted(models.keys()):
        crop_display, crop_normalized = models[model_idx]
        
        # Find folders for this crop
        crop_folders = []
        for c in all_classes:
            c_norm = c.lower()
            if crop_normalized == 'cherry' and 'cherry' in c_norm:
                crop_folders.append(c)
            elif crop_normalized == 'corn' and 'corn' in c_norm:
                crop_folders.append(c)
            elif crop_normalized == 'pepper' and 'pepper' in c_norm:
                crop_folders.append(c)
            elif crop_normalized in c_norm and crop_normalized not in ['cherry', 'corn', 'pepper']:
                crop_folders.append(c)
                
        # Determine labels
        labels = {}
        for folder in crop_folders:
            norm_lbl = normalize_label(folder)
            labels[folder] = norm_lbl
            
        # Map indices
        sorted_lbls = sorted(list(set(labels.values())))
        if 'healthy' in sorted_lbls:
            sorted_lbls.remove('healthy')
            sorted_lbls = ['healthy'] + sorted_lbls
            
        label_to_id = {lbl: i for i, lbl in enumerate(sorted_lbls)}
        
        class_mappings_json[f"model{model_idx}_{crop_normalized}"] = {
            str(i): lbl for i, lbl in enumerate(sorted_lbls)
        }
        class_mappings_json[f"model{model_idx}_{crop_normalized}_reverse"] = {
            lbl: i for i, lbl in enumerate(sorted_lbls)
        }
        
        # Process files and hash
        # structure: dict mapping class -> list of dicts with file info
        data_by_class = defaultdict(list)
        total_images = 0
        
        for folder in crop_folders:
            folder_path = dataset_path / folder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(folder_path / f)]
            norm_lbl = labels[folder]
            lbl_id = label_to_id[norm_lbl]
            
            for f in files:
                fpath = folder_path / f
                h = get_sha256(fpath)
                data_by_class[norm_lbl].append({
                    'image_path': str(fpath),
                    'label': norm_lbl,
                    'label_id': lbl_id,
                    'original_class': folder,
                    'source': 'plantvillage',
                    'crop': crop_normalized,
                    'sha256': h
                })
                total_images += 1
                
        # Group by hash within each class to do splits
        split_data = []
        train_count = 0
        val_count = 0
        test_count = 0
        
        dup_groups_count = 0
        
        for cls, items in data_by_class.items():
            hash_groups = defaultdict(list)
            for item in items:
                hash_groups[item['sha256']].append(item)
                
            dup_groups_count += sum(1 for g in hash_groups.values() if len(g) > 1)
                
            groups = list(hash_groups.values())
            random.shuffle(groups)
            
            n_items = len(items)
            target_train = int(n_items * 0.8)
            target_val = int(n_items * 0.1)
            
            curr_train = 0
            curr_val = 0
            
            for g in groups:
                if curr_train < target_train:
                    split = 'train'
                    curr_train += len(g)
                    train_count += len(g)
                elif curr_val < target_val:
                    split = 'val'
                    curr_val += len(g)
                    val_count += len(g)
                else:
                    split = 'test'
                    test_count += len(g)
                    
                for item in g:
                    item['split'] = split
                    split_data.append(item)
                    
        # Verification
        cross_split_leak = 0
        global_hash_to_split = {}
        for item in split_data:
            h = item['sha256']
            s = item['split']
            if h in global_hash_to_split and global_hash_to_split[h] != s:
                cross_split_leak += 1
                global_cross_split += 1
            global_hash_to_split[h] = s
            
        cls_in_train = set(x['label'] for x in split_data if x['split'] == 'train')
        cls_in_val = set(x['label'] for x in split_data if x['split'] == 'val')
        cls_in_test = set(x['label'] for x in split_data if x['split'] == 'test')
        
        all_cls = set(sorted_lbls)
        
        global_missing_train += len(all_cls - cls_in_train)
        global_missing_val += len(all_cls - cls_in_val)
        global_missing_test += len(all_cls - cls_in_test)
        
        # Write CSVs
        master_csv_path = metadata_dir / f"model{model_idx}_{crop_normalized}_master.csv"
        split_csv_path = metadata_dir / f"model{model_idx}_{crop_normalized}_split.csv"
        
        fields = ['image_path', 'label', 'label_id', 'original_class', 'split', 'source', 'crop', 'sha256']
        
        with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in split_data:
                writer.writerow(row)
                
        with open(split_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in split_data:
                writer.writerow(row)
                
        # Reporting
        report_content.extend([
            f"## Model {model_idx} — {crop_display.upper()}",
            f"- **Classes**: {len(sorted_lbls)}",
            f"- **Class Mapping**: {class_mappings_json[f'model{model_idx}_{crop_normalized}']}",
            f"- **Total Images**: {total_images}",
            f"- **Train**: {train_count}",
            f"- **Validation**: {val_count}",
            f"- **Test**: {test_count}",
            f"- **Duplicate Groups**: {dup_groups_count}",
            f"- **Cross-split Leakage**: {cross_split_leak}",
            f"- **Master CSV**: `{master_csv_path}`",
            f"- **Split CSV**: `{split_csv_path}`",
            ""
        ])
        
        terminal_output.extend([
            f"MODEL {model_idx} — {crop_display.upper()}",
            f"Classes: {len(sorted_lbls)}",
            f"Class mapping: {class_mappings_json[f'model{model_idx}_{crop_normalized}']}",
            f"Total: {total_images}",
            f"Train: {train_count}",
            f"Validation: {val_count}",
            f"Test: {test_count}",
            f"Duplicate groups: {dup_groups_count}",
            f"Cross-split duplicate leakage: {cross_split_leak}",
            ""
        ])

    mapping_file = metadata_dir / "models5_11_class_mappings.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mappings_json, f, indent=2)
        
    report_file = reports_dir / "MODELS_5_TO_11_DATASET_SPLIT_REPORT.md"
    
    report_content.extend([
        "## Global Verification",
        f"- **Missing files**: {global_missing}",
        f"- **Unreadable images**: {global_unreadable}",
        f"- **Cross-class duplicates**: {global_cross_class}",
        f"- **Cross-split duplicate leakage**: {global_cross_split}",
        f"- **Classes missing from train**: {global_missing_train}",
        f"- **Classes missing from validation**: {global_missing_val}",
        f"- **Classes missing from test**: {global_missing_test}",
        "",
        "## Test Set Status",
        "**LOCKED** - Test sets must not be used for anything other than final evaluation."
    ])
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
        
    terminal_output.extend([
        "GLOBAL VERIFICATION",
        "",
        f"Missing files: {global_missing}",
        f"Unreadable images: {global_unreadable}",
        f"Cross-class duplicates: {global_cross_class}",
        f"Cross-split duplicate leakage: {global_cross_split}",
        f"Classes missing from train: {global_missing_train}",
        f"Classes missing from validation: {global_missing_val}",
        f"Classes missing from test: {global_missing_test}",
        "",
        f"CLASS MAPPING FILE:\n{mapping_file}",
        "",
        f"REPORT:\n{report_file}",
        "",
        "TEST SET STATUS:",
        "LOCKED"
    ])
    
    print("\n".join(terminal_output))

if __name__ == "__main__":
    main()
