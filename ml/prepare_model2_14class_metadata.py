import os
import pandas as pd
import numpy as np
import math

def run():
    input_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model2_split.csv"
    output_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model2_14class_split.csv"
    report_file = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model2_14class_dataset_report.txt"
    
    print("Loading existing Model 2 split...")
    df = pd.read_csv(input_csv)
    
    label_map = {
        'apple': 0,
        'blueberry': 1,
        'cherry': 2,
        'corn': 3,
        'grape': 4,
        'orange': 5,
        'peach': 6,
        'pepper': 7,
        'potato': 8,
        'raspberry': 9,
        'soybean': 10,
        'squash': 11,
        'strawberry': 12,
        'tomato': 13
    }
    
    def map_crop(orig_class):
        prefix = orig_class.split('___')[0].lower()
        if 'apple' in prefix: return 'apple'
        if 'blueberry' in prefix: return 'blueberry'
        if 'cherry' in prefix: return 'cherry'
        if 'corn' in prefix: return 'corn'
        if 'grape' in prefix: return 'grape'
        if 'orange' in prefix: return 'orange'
        if 'peach' in prefix: return 'peach'
        if 'pepper' in prefix: return 'pepper'
        if 'potato' in prefix: return 'potato'
        if 'raspberry' in prefix: return 'raspberry'
        if 'soybean' in prefix: return 'soybean'
        if 'squash' in prefix: return 'squash'
        if 'strawberry' in prefix: return 'strawberry'
        if 'tomato' in prefix: return 'tomato'
        raise ValueError(f"Unknown crop prefix for: {orig_class}")

    df['label'] = df['original_class'].apply(map_crop)
    df['label_id'] = df['label'].map(label_map)
    
    # Reorder columns
    cols = ['image_path', 'label', 'label_id', 'original_class', 'split']
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]
    
    # Verifications
    print("Running verifications...")
    total_imgs = len(df)
    train_count = len(df[df['split'] == 'train'])
    val_count = len(df[df['split'] == 'val'])
    test_count = len(df[df['split'] == 'test'])
    
    print(f"Total images: {total_imgs} (Expected: 54305)")
    print(f"Train: {train_count} (Expected: 43444)")
    print(f"Val: {val_count} (Expected: 5430)")
    print(f"Test: {test_count} (Expected: 5431)")
    
    missing = sum(not os.path.exists(p) for p in df['image_path'])
    duplicates = df['image_path'].duplicated().sum()
    unique_labels = sorted(df['label_id'].unique())
    
    if missing > 0 or duplicates > 0 or len(unique_labels) != 14:
        print(f"Verification Failed! Missing: {missing}, Duplicates: {duplicates}, Labels: {len(unique_labels)}")
    else:
        print("Verification Passed: No missing files, no duplicates, exactly 14 labels.")
        
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
    
    # Report Generation
    report = []
    report.append("Model 2 (14-class) Dataset Report")
    report.append("=================================\n")
    
    report.append("Overall Verification:")
    report.append(f"  Total images: {total_imgs}")
    report.append(f"  Train: {train_count}")
    report.append(f"  Val: {val_count}")
    report.append(f"  Test: {test_count}")
    report.append(f"  Unique Crops: {len(unique_labels)}\n")
    
    report.append("Class Distribution:\n")
    
    crop_stats = {}
    for crop, crop_id in sorted(label_map.items(), key=lambda x: x[1]):
        crop_df = df[df['label'] == crop]
        crop_train = crop_df[crop_df['split'] == 'train']
        crop_val = crop_df[crop_df['split'] == 'val']
        crop_test = crop_df[crop_df['split'] == 'test']
        
        c_total = len(crop_df)
        c_train = len(crop_train)
        c_val = len(crop_val)
        c_test = len(crop_test)
        
        pct_train = (c_train / train_count) * 100 if train_count > 0 else 0
        
        crop_stats[crop] = c_train
        
        report.append(f"[{crop_id:2d}] {crop.capitalize()}:")
        report.append(f"      Total: {c_total}")
        report.append(f"      Train: {c_train} ({pct_train:.2f}% of training set)")
        report.append(f"      Val:   {c_val}")
        report.append(f"      Test:  {c_test}\n")
        
    # Imbalance Analysis
    max_crop = max(crop_stats, key=crop_stats.get)
    min_crop = min(crop_stats, key=crop_stats.get)
    max_count = crop_stats[max_crop]
    min_count = crop_stats[min_crop]
    ratio = max_count / min_count if min_count > 0 else float('inf')
    
    report.append("Imbalance Analysis:")
    report.append(f"  Largest crop class: {max_crop.capitalize()} ({max_count} train images)")
    report.append(f"  Smallest crop class: {min_crop.capitalize()} ({min_count} train images)")
    report.append(f"  Max/Min imbalance ratio: {ratio:.1f}x\n")
    
    report.append("Class Weights Analysis:")
    report.append("  (1) Standard Inverse-Frequency Balanced Weights:")
    # weight = N_total / (num_classes * N_class)
    for crop, crop_id in sorted(label_map.items(), key=lambda x: x[1]):
        w = train_count / (14 * crop_stats[crop]) if crop_stats[crop] > 0 else 0
        report.append(f"      {crop}: {w:.4f}")
        
    report.append("\n  (2) Square-Root Inverse-Frequency Weights:")
    # weight = sqrt(N_total / N_class) normalized so mean=1 or just standard
    # Let's just do standard sqrt(N_total / N_class) / num_classes? 
    # Or just sqrt( 1 / N_class ) normalized.
    sqrt_weights = {}
    for crop in label_map.keys():
        sqrt_weights[crop] = math.sqrt(train_count / crop_stats[crop]) if crop_stats[crop] > 0 else 0
    # Normalize them to average to 1
    mean_sqrt = sum(sqrt_weights.values()) / 14
    for crop in label_map.keys():
        sqrt_weights[crop] /= mean_sqrt
        
    for crop, crop_id in sorted(label_map.items(), key=lambda x: x[1]):
        report.append(f"      {crop}: {sqrt_weights[crop]:.4f}")
        
    report.append("\n  (3) Recommended Mild/Capped Weighting Strategy:")
    report.append("      Since aggressive inverse-frequency caused overprediction of minority classes like Potato,")
    report.append("      we recommend using the Square-Root Inverse-Frequency weights, or standard balanced weights")
    report.append("      capped at a maximum value (e.g., max_weight = 3.0) to prevent the loss from being dominated")
    report.append("      by extreme outliers like Soybean/Raspberry.")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    run()
