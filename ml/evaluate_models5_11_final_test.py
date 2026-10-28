import os
import sys
import json
import time
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_dsp_preprocessing import DSPTransform
from src.models.model1_resnet9 import ResNet9Model1

class TestDiseaseDataset(Dataset):
    def __init__(self, csv_file, mean, std):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'test']
        self.mean = mean
        self.std = std
        
        self.transform = transforms.Compose([
            DSPTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label_id'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0,0,0))
            
        tensor = self.transform(image)
        return tensor, torch.tensor(label, dtype=torch.long), img_path

def save_confusion_matrix(cm, classes, filepath):
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df.to_csv(filepath)

def evaluate_model(manifest_entry, workspace, device, smoke_test=False):
    model_name = manifest_entry['model_name']
    ckpt_path = Path(manifest_entry['checkpoint_path'])
    num_classes = manifest_entry['num_classes']
    mean = manifest_entry['normalization_mean']
    std = manifest_entry['normalization_std']
    classes = [manifest_entry['class_mapping'][str(i)] for i in range(num_classes)]
    csv_path = workspace / "metadata" / f"{model_name}_master.csv"
    
    # Dirs
    results_dir = workspace / "results" / f"{model_name}_final_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = workspace / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    model = ResNet9Model1(in_channels=3, num_classes=num_classes).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state['model_state'])
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    # Test DataLoader
    test_ds = TestDiseaseDataset(csv_path, mean, std)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    
    test_count = len(test_ds)
    
    if smoke_test:
        # Dry run syntax checks
        all_labels = [0]
        all_preds = [0]
        all_scores = [0.9]
        all_paths = ["fake.jpg"]
        test_loss = 0.5
        correct = 1
    else:
        vrun_loss = 0.0
        correct = 0
        all_preds, all_labels, all_scores, all_paths = [], [], [], []
        
        with torch.inference_mode():
            for imgs, labels, paths in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                
                vrun_loss += loss.item()
                probs = torch.nn.functional.softmax(logits, dim=1)
                scores, preds = probs.max(1)
                
                correct += preds.eq(labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
                all_paths.extend(paths)
                
        test_loss = vrun_loss / max(1, len(test_loader))
        
    test_acc = 100. * correct / max(1, test_count if not smoke_test else 1)
    
    prec, rec, f1, sup = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(num_classes)), zero_division=0)
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Save Predictions CSV
    preds_df = pd.DataFrame({
        'filepath': all_paths,
        'true_class_index': all_labels,
        'true_class_label': [classes[l] for l in all_labels],
        'predicted_class_index': all_preds,
        'predicted_class_label': [classes[p] for p in all_preds],
        'model_score': all_scores
    })
    preds_df.to_csv(results_dir / "test_predictions.csv", index=False)
    
    # Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    save_confusion_matrix(cm, classes, results_dir / "confusion_matrix.csv")
    
    # Single Report
    report = [
        f"# {model_name} Final Test Evaluation",
        "",
        f"- Checkpoint Path: `{ckpt_path}`",
        f"- Checkpoint Stage: `{manifest_entry['checkpoint_stage']}`",
        f"- Validation Accuracy: {manifest_entry['validation_accuracy']:.2f}%",
        f"- Validation Macro-F1: {manifest_entry['validation_macro_f1']:.4f}",
        "",
        "## Test Metrics",
        f"- Test Samples: {test_count}",
        f"- Test Loss: {test_loss:.4f}",
        f"- Test Accuracy: {test_acc:.2f}%",
        f"- Correct: {correct}",
        f"- Incorrect: {(test_count if not smoke_test else 1) - correct}",
        f"- Macro Precision: {m_prec:.4f}",
        f"- Macro Recall: {m_rec:.4f}",
        f"- Macro-F1: {m_f1:.4f}",
        f"- Weighted Precision: {w_prec:.4f}",
        f"- Weighted Recall: {w_rec:.4f}",
        f"- Weighted-F1: {w_f1:.4f}",
        "",
        "## Per-Class Metrics",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|"
    ]
    for i, cls in enumerate(classes):
        report.append(f"| {cls} | {prec[i]:.4f} | {rec[i]:.4f} | {f1[i]:.4f} | {int(sup[i]) if not smoke_test else 1} |")
        
    with open(reports_dir / f"{model_name}_final_test.md", "w") as f:
        f.write("\n".join(report))
        
    return {
        "model": model_name,
        "crop": manifest_entry['crop'],
        "classes": num_classes,
        "stage": manifest_entry['checkpoint_stage'],
        "samples": test_count,
        "accuracy": test_acc,
        "m_prec": m_prec,
        "m_rec": m_rec,
        "m_f1": m_f1,
        "w_f1": w_f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke_test', action='store_true', help='Run smoke test without loading test images')
    args = parser.parse_args()
    
    workspace = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
    manifest_path = workspace / "metadata" / "models5_11_final_champions.json"
    global_report_path = workspace / "reports" / "MODELS_5_TO_11_FINAL_TEST_REPORT.md"
    consumed_marker = workspace / "metadata" / "MODELS_5_TO_11_TEST_SETS_CONSUMED.txt"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.smoke_test:
        print("Running SMOKE TEST without iterating test images.")
        
    results = []
    total_test_samples = 0
    
    for entry in manifest:
        if not args.smoke_test and entry['test_status'] == 'CONSUMED_FINAL':
            print(f"Skipping {entry['model_name']} - already CONSUMED_FINAL.")
            continue
            
        print(f"Evaluating {entry['model_name']}...")
        res = evaluate_model(entry, workspace, device, smoke_test=args.smoke_test)
        results.append(res)
        total_test_samples += res['samples']
        
        # Cleanup memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Write Global Report
    global_report = [
        "# Models 5-11 Final Test Report",
        "",
        "| Model | Crop | Classes | Stage | Test Samples | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 |",
        "|---|---|---|---|---|---|---|---|---|---|"
    ]
    for r in results:
        global_report.append(f"| {r['model']} | {r['crop']} | {r['classes']} | {r['stage']} | {r['samples']} | {r['accuracy']:.2f}% | {r['m_prec']:.4f} | {r['m_rec']:.4f} | {r['m_f1']:.4f} | {r['w_f1']:.4f} |")
        
    with open(global_report_path, 'w') as f:
        f.write("\n".join(global_report))
        
    if not args.smoke_test:
        # Update Manifest
        for entry in manifest:
            entry['test_status'] = "CONSUMED_FINAL"
            entry['test_evaluated'] = True
            entry['test_evaluation_timestamp'] = time.time()
            
            # Find result
            for r in results:
                if r['model'] == entry['model_name']:
                    entry['test_accuracy'] = r['accuracy']
                    entry['test_macro_f1'] = r['m_f1']
                    
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
            
        # Write Consumed Marker
        consumed_text = [
            "The final test sets for Models 5–11 have been evaluated.",
            "",
            "They must NOT be used for:",
            "- checkpoint selection",
            "- model refinement",
            "- hyperparameter tuning",
            "- architecture selection",
            "- augmentation selection",
            "",
            "Any future model development requires a new untouched test set."
        ]
        with open(consumed_marker, 'w') as f:
            f.write("\n".join(consumed_text))
            
    print(f"Done. Processed {total_test_samples} test samples in total.")

if __name__ == '__main__':
    main()
