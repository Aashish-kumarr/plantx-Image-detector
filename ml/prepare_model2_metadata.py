import os
import pandas as pd
import numpy as np

def run():
    input_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_split.csv"
    output_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model2_split.csv"
    report_file = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model2_dataset_report.txt"
    
    print("Loading Model 1 split...")
    df = pd.read_csv(input_csv)
    
    # Filter only PlantVillage color images. 
    # PlantVillage paths usually contain "PlantVillage" or "color".
    # And we know non-leaf images are from Caltech and Places.
    # Let's filter by path containing "color" or "PlantVillage" and excluding "Caltech" and "places365".
    # To be safe, let's just check if "caltech" or "places365" are in the path.
    def is_pv(path):
        p = path.lower()
        return ('plantvillage' in p or 'color' in p) and ('caltech' not in p and 'places365' not in p and 'non_leaf' not in p)

    df_pv = df[df['image_path'].apply(is_pv)].copy()
    
    # Just in case, ensure it's not grayscale or segmented
    df_pv = df_pv[~df_pv['image_path'].str.contains('grayscale', case=False)]
    df_pv = df_pv[~df_pv['image_path'].str.contains('segmented', case=False)]
    
    print(f"Filtered to {len(df_pv)} PlantVillage images.")
    
    def get_model2_label(orig_class):
        if orig_class.startswith('Potato___'):
            return 'potato', 0
        elif orig_class.startswith('Tomato___'):
            return 'tomato', 1
        else:
            return 'other', 2
            
    # Apply mapping
    labels_and_ids = df_pv['original_class'].apply(get_model2_label)
    df_pv['label'] = [x[0] for x in labels_and_ids]
    df_pv['label_id'] = [x[1] for x in labels_and_ids]
    
    # Reorder columns
    cols = ['image_path', 'label', 'label_id', 'original_class', 'split']
    # If there are any other columns, append them
    other_cols = [c for c in df_pv.columns if c not in cols]
    df_pv = df_pv[cols + other_cols]
    
    # Verifications
    print("Running verifications...")
    
    # 1. Every image exists
    missing = sum(not os.path.exists(p) for p in df_pv['image_path'])
    print(f"Missing images: {missing}")
    
    # 2. No duplicates
    duplicates = df_pv['image_path'].duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # 3. Splits are exactly train, val, test
    splits = set(df_pv['split'].unique())
    print(f"Splits found: {splits}")
    
    # Save CSV
    df_pv.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
    
    # Generate Report
    report = []
    report.append("Model 2 Dataset Report")
    report.append("======================\n")
    
    report.append("Overall Counts:")
    total = len(df_pv)
    potato_total = len(df_pv[df_pv['label'] == 'potato'])
    tomato_total = len(df_pv[df_pv['label'] == 'tomato'])
    other_total = len(df_pv[df_pv['label'] == 'other'])
    
    report.append(f"  Total Model 2 images: {total}")
    report.append(f"  Total Potato: {potato_total} ({(potato_total/total)*100:.1f}%)")
    report.append(f"  Total Tomato: {tomato_total} ({(tomato_total/total)*100:.1f}%)")
    report.append(f"  Total Other: {other_total} ({(other_total/total)*100:.1f}%)\n")
    
    splits = ['train', 'val', 'test']
    for s in splits:
        df_s = df_pv[df_pv['split'] == s]
        stotal = len(df_s)
        spotato = len(df_s[df_s['label'] == 'potato'])
        stomato = len(df_s[df_s['label'] == 'tomato'])
        sother = len(df_s[df_s['label'] == 'other'])
        
        report.append(f"Per-split Counts: {s.upper()} ({stotal} images)")
        if stotal > 0:
            report.append(f"  Potato: {spotato} ({(spotato/stotal)*100:.1f}%)")
            report.append(f"  Tomato: {stomato} ({(stomato/stotal)*100:.1f}%)")
            report.append(f"  Other: {sother} ({(sother/stotal)*100:.1f}%)\n")
        else:
            report.append("  No images in this split.\n")
            
    # Class Imbalance Analysis
    report.append("Class Imbalance Analysis (Training Set):")
    df_train = df_pv[df_pv['split'] == 'train']
    train_total = len(df_train)
    counts = {
        'potato (0)': len(df_train[df_train['label'] == 'potato']),
        'tomato (1)': len(df_train[df_train['label'] == 'tomato']),
        'other (2)': len(df_train[df_train['label'] == 'other'])
    }
    
    report.append("  Training frequencies:")
    for k, v in counts.items():
        report.append(f"    {k}: {v} images")
        
    # Suggested CrossEntropy Weights
    # Weight = N_total / (num_classes * N_class)
    report.append("\n  Suggested Balanced CrossEntropy Weights:")
    num_classes = 3
    for k, v in counts.items():
        weight = train_total / (num_classes * v) if v > 0 else 0
        report.append(f"    {k}: {weight:.4f}")
        
    report.append("\n  Imbalance conclusion:")
    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count if min_count > 0 else float('inf')
    report.append(f"    The most frequent class is {ratio:.1f}x larger than the least frequent class.")
    report.append(f"    Model 2 is significantly imbalanced toward the 'other' class.")
    
    # Save Report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    run()
