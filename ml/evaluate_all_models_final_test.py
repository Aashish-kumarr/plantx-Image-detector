
import os
import sys
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
sys.path.insert(0, BASE_DIR)

from src.models.model1_resnet9 import create_model1_resnet9
from src.models.model2_14class_resnet9 import create_model2_14class_resnet9
from src.models.model3_potato_resnet9 import create_model3_potato_resnet9
from src.models.model4_tomato_resnet9 import create_model4_tomato_resnet9

from src.datasets.model1_dsp_preprocessing import apply_dsp_preprocessing
from src.datasets.model1_transforms import get_model1_eval_transform
from src.datasets.model2_transforms import get_model2_eval_transform
from src.datasets.model3_potato_transforms import get_model3_eval_transform
from src.datasets.model4_tomato_transforms import get_model4_eval_transform

# ---------------------------------------------------------------------------
# Path remapping helper
# ---------------------------------------------------------------------------
OLD_PREFIXES = [
    r"C:\Users\ak021\Desktop\plantDx\plant image detector",
    r"c:\Users\ak021\Desktop\plant image detector",
]
NEW_PREFIX = BASE_DIR

def remap_image_path(raw_path: str) -> str:
    for old in OLD_PREFIXES:
        if raw_path.lower().startswith(old.lower()):
            return NEW_PREFIX + raw_path[len(old):]
    return raw_path

# ---------------------------------------------------------------------------
# Dataset definition for evaluation
# ---------------------------------------------------------------------------
class PathRemappedTestDataset(Dataset):
    def __init__(self, csv_path: str, transform):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == "test"].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = remap_image_path(str(row["image_path"]))
        label_id = int(row["label_id"])
        label_str = str(row["label"])
        pil_img = Image.open(img_path).convert("RGB")
        dsp_pil = apply_dsp_preprocessing(pil_img)
        tensor_img = self.transform(dsp_pil)
        return tensor_img, label_id, label_str, img_path

# ---------------------------------------------------------------------------
# Configuration / Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(BASE_DIR, "results", "final_test_evaluation")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "model1": {
        "name": "Model 1 ? Leaf vs Non-Leaf",
        "ckpt_path": os.path.join(BASE_DIR, "checkpoints", "model1", "best_model1.pth"),
        "csv_path": os.path.join(BASE_DIR, "metadata", "model1_split.csv"),
        "num_classes": 2,
        "expected_count": 11110,
        "val_acc": 96.68,
        "val_f1": 0.9667,
        "state_key": "model_state_dict",
        "arch_fn": create_model1_resnet9,
        "transform_fn": get_model1_eval_transform,
        "class_names": ["non_leaf", "leaf"],
    },
    "model2": {
        "name": "Model 2 ? 14-Class Crop Classifier",
        "ckpt_path": os.path.join(BASE_DIR, "checkpoints", "model2_versions", "model2_14class_refine_v1_f1_0.9335_acc_95.01.pth"),
        "csv_path": os.path.join(BASE_DIR, "metadata", "model2_14class_split.csv"),
        "num_classes": 14,
        "expected_count": 5431,
        "val_acc": 95.01,
        "val_f1": 0.9335,
        "state_key": "model_state",
        "arch_fn": create_model2_14class_resnet9,
        "transform_fn": get_model2_eval_transform,
        "class_names": [
            "apple", "blueberry", "cherry", "corn", "grape", "orange",
            "peach", "pepper", "potato", "raspberry", "soybean",
            "squash", "strawberry", "tomato"
        ],
    },
    "model3": {
        "name": "Model 3 ? Potato Disease Classifier",
        "ckpt_path": os.path.join(BASE_DIR, "checkpoints", "model3_versions", "model3_potato_refine_v1_f1_0.9194_acc_93.67.pth"),
        "csv_path": os.path.join(BASE_DIR, "metadata", "model3_potato_split.csv"),
        "num_classes": 3,
        "expected_count": 210,
        "val_acc": 93.67,
        "val_f1": 0.9194,
        "state_key": "model_state",
        "arch_fn": create_model3_potato_resnet9,
        "transform_fn": get_model3_eval_transform,
        "class_names": ["healthy", "early_blight", "late_blight"],
    },
    "model4": {
        "name": "Model 4 ? Tomato Disease Classifier",
        "ckpt_path": os.path.join(BASE_DIR, "checkpoints", "model4_versions", "model4_tomato_refine_v1_f1_0.9131_acc_93.47.pth"),
        "csv_path": os.path.join(BASE_DIR, "metadata", "model4_tomato_split.csv"),
        "num_classes": 10,
        "expected_count": 1865,
        "val_acc": 93.47,
        "val_f1": 0.9131,
        "state_key": "model_state",
        "arch_fn": create_model4_tomato_resnet9,
        "transform_fn": get_model4_eval_transform,
        "class_names": [
            "healthy", "early_blight", "late_blight", "bacterial_spot",
            "leaf_mold", "septoria_leaf_spot", "spider_mites",
            "target_spot", "tomato_mosaic_virus", "tomato_yellow_leaf_curl_virus"
        ],
    },
}

# ---------------------------------------------------------------------------
# Dry Verification
# ---------------------------------------------------------------------------
def dry_verify():
    print("=" * 68)
    print("  PLANTDX FINAL TEST EVALUATION ? DRY VERIFICATION")
    print("=" * 68)

    all_ok = True
    for key, cfg in MODEL_CONFIGS.items():
        print(f"\n--- Verifying {cfg['name']} ---")
        
        # Checkpoint existence
        if not os.path.exists(cfg["ckpt_path"]):
            print(f"  [FAIL] Checkpoint not found: {cfg['ckpt_path']}")
            all_ok = False
            continue
        print(f"  [OK] Checkpoint exists: {os.path.basename(cfg['ckpt_path'])}")
        
        # Checkpoint load + strict model match
        ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")
        state_key = cfg["state_key"]
        if state_key not in ckpt:
            print(f"  [FAIL] Key '{state_key}' not in checkpoint. Available keys: {list(ckpt.keys())}")
            all_ok = False
            continue
            
        model = cfg["arch_fn"]()
        try:
            model.load_state_dict(ckpt[state_key], strict=True)
            print(f"  [OK] Strict state_dict load succeeded (output dim = {cfg['num_classes']})")
        except Exception as e:
            print(f"  [FAIL] Strict load failed: {e}")
            all_ok = False
            continue
            
        # Test split count
        if not os.path.exists(cfg["csv_path"]):
            print(f"  [FAIL] Split CSV not found: {cfg['csv_path']}")
            all_ok = False
            continue
            
        df = pd.read_csv(cfg["csv_path"])
        test_cnt = len(df[df["split"] == "test"])
        if test_cnt != cfg["expected_count"]:
            print(f"  [FAIL] Test count mismatch: found {test_cnt}, expected {cfg['expected_count']}")
            all_ok = False
            continue
        print(f"  [OK] Test count matches exactly ({test_cnt} samples)")

    print("\n" + "=" * 68)
    if all_ok:
        print("ALL DRY-RUN VERIFICATIONS PASSED SUCCESSFULLY!")
        print("Ready to run full evaluation from PowerShell.")
    else:
        print("DRY-RUN VERIFICATION FAILED. Review errors above.")
    print("=" * 68)
    return all_ok

# ---------------------------------------------------------------------------
# Full Evaluation Execution
# ---------------------------------------------------------------------------
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning Final Test Evaluation on Device: {device}\n")

    summary_metrics = {}

    for key, cfg in MODEL_CONFIGS.items():
        print("=" * 68)
        print(f"  Evaluating {cfg['name']}")
        print("=" * 68)

        # Load checkpoint & model
        ckpt = torch.load(cfg["ckpt_path"], map_location=device)
        model = cfg["arch_fn"]().to(device)
        model.load_state_dict(ckpt[cfg["state_key"]], strict=True)
        model.eval()

        # Dataloader
        ds = PathRemappedTestDataset(cfg["csv_path"], cfg["transform_fn"]())
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

        all_preds = []
        all_labels = []
        all_confidences = []
        all_paths = []
        all_label_strs = []
        running_loss = 0.0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        with torch.inference_mode():
            for imgs, label_ids, label_strs, paths in loader:
                imgs = imgs.to(device)
                label_ids_dev = label_ids.to(device)
                
                logits = model(imgs)
                loss = criterion(logits, label_ids_dev)
                running_loss += loss.item() * imgs.size(0)
                
                probs = torch.softmax(logits, dim=1)
                confs, preds = torch.max(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_ids.numpy())
                all_confidences.extend(confs.cpu().numpy())
                all_paths.extend(paths)
                all_label_strs.extend(label_strs)
                total_samples += imgs.size(0)

        test_loss = running_loss / total_samples
        test_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100.0

        # Metrics
        mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        wei_p, wei_r, wei_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        pc_p, pc_r, pc_f1, pc_sup = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=range(cfg["num_classes"]), zero_division=0
        )

        class_names = cfg["class_names"]

        print(f"Test Samples : {total_samples}")
        print(f"Test Loss    : {test_loss:.4f}")
        print(f"Test Acc     : {test_acc:.2f}% (Val Champion: {cfg['val_acc']:.2f}%)")
        print(f"Macro F1     : {mac_f1:.4f} (Val Champion: {cfg['val_f1']:.4f})")
        print(f"Weighted F1  : {wei_f1:.4f}")
        print("\nPer-Class Breakdown:")
        for i, cname in enumerate(class_names):
            print(f"  {cname:<35} P={pc_p[i]:.4f} R={pc_r[i]:.4f} F1={pc_f1[i]:.4f} (N={pc_sup[i]})")

        # Save predictions CSV
        pred_csv_path = os.path.join(RESULTS_DIR, f"{key}_predictions.csv")
        pred_df = pd.DataFrame({
            "image_path": all_paths,
            "true_label": all_labels,
            "true_class": [class_names[i] for i in all_labels],
            "predicted_label": all_preds,
            "predicted_class": [class_names[i] for i in all_preds],
            "confidence": [float(c) for c in all_confidences],
            "correct": [int(t == p) for t, p in zip(all_labels, all_preds)]
        })
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"\nSaved predictions -> {pred_csv_path}")

        # Save Confusion Matrices (Raw & Normalized)
        cm = confusion_matrix(all_labels, all_preds, labels=range(cfg["num_classes"]))
        
        # Raw CM
        disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names)-2)))
        disp_raw.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
        plt.title(f"{cfg['name']} ? Final Test Set Confusion Matrix (Raw Counts)")
        plt.tight_layout()
        cm_raw_path = os.path.join(RESULTS_DIR, f"{key}_confusion_matrix.png")
        plt.savefig(cm_raw_path)
        plt.close()

        # Normalized CM
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names)-2)))
        disp_norm.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format=".2f")
        plt.title(f"{cfg['name']} ? Final Test Set Confusion Matrix (Normalized)")
        plt.tight_layout()
        cm_norm_path = os.path.join(RESULTS_DIR, f"{key}_confusion_matrix_normalized.png")
        plt.savefig(cm_norm_path)
        plt.close()

        # Store in json summary
        per_class_dict = {}
        for i, cname in enumerate(class_names):
            per_class_dict[cname] = {
                "precision": float(pc_p[i]),
                "recall": float(pc_r[i]),
                "f1": float(pc_f1[i]),
                "support": int(pc_sup[i])
            }

        summary_metrics[key] = {
            "checkpoint": cfg["ckpt_path"],
            "test_samples": total_samples,
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "val_accuracy": float(cfg["val_acc"]),
            "acc_gap": float(test_acc - cfg["val_acc"]),
            "macro_precision": float(mac_p),
            "macro_recall": float(mac_r),
            "macro_f1": float(mac_f1),
            "val_macro_f1": float(cfg["val_f1"]),
            "f1_gap": float(mac_f1 - cfg["val_f1"]),
            "weighted_f1": float(wei_f1),
            "per_class_metrics": per_class_dict,
            "confusion_matrix": cm.tolist()
        }

    # Save JSON summary
    json_path = os.path.join(RESULTS_DIR, "final_test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\nSaved final test metrics JSON -> {json_path}")

    # Generate MASTER REPORT markdown
    generate_master_report(summary_metrics)

def generate_master_report(metrics):
    md = []
    md.append("# PlantDx Final Test Evaluation")
    md.append("\nThis document contains the official final test set evaluation results for all 4 trained/refined models in the PlantDx pipeline.")
    md.append("All evaluations were performed deterministically on held-out, previously untouched test splits.")

    md.append("\n## Summary Table")
    md.append("| Model | Task | Test Samples | Val Acc | Test Acc | Acc Gap | Val Macro-F1 | Test Macro-F1 | F1 Gap |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    
    for key, cfg in MODEL_CONFIGS.items():
        m = metrics[key]
        md.append(
            f"| {key.upper()} | {cfg['name']} | {m['test_samples']:,} | "
            f"{m['val_accuracy']:.2f}% | {m['accuracy']:.2f}% | {m['acc_gap']:+.2f}% | "
            f"{m['val_macro_f1']:.4f} | {m['macro_f1']:.4f} | {m['f1_gap']:+.4f} |"
        )

    for key, cfg in MODEL_CONFIGS.items():
        m = metrics[key]
        md.append(f"\n## {cfg['name']} ({key.upper()}) Results")
        md.append(f"- **Evaluated Checkpoint**: {m['checkpoint']}")
        md.append(f"- **Test Samples**: {m['test_samples']:,}")
        md.append(f"- **Test Accuracy**: {m['accuracy']:.2f}% (Validation: {m['val_accuracy']:.2f}%, Gap: {m['acc_gap']:+.2f}%)")
        md.append(f"- **Test Macro-F1**: {m['macro_f1']:.4f} (Validation: {m['val_macro_f1']:.4f}, Gap: {m['f1_gap']:+.4f})")
        md.append(f"- **Test Weighted-F1**: {m['weighted_f1']:.4f}")
        md.append(f"- **Test Loss**: {m['loss']:.4f}")

        md.append("\n### Per-Class Metrics")
        md.append("| Class Name | Precision | Recall | F1 Score | Support |")
        md.append("|---|---|---|---|---|")
        for cname, pcm in m["per_class_metrics"].items():
            md.append(f"| {cname} | {pcm['precision']:.4f} | {pcm['recall']:.4f} | {pcm['f1']:.4f} | {pcm['support']:,} |")

        # Top Confusions
        cm = np.array(m["confusion_matrix"])
        cnames = cfg["class_names"]
        confusions = []
        for i in range(len(cnames)):
            for j in range(len(cnames)):
                if i != j and cm[i, j] > 0:
                    confusions.append((cnames[i], cnames[j], int(cm[i, j])))
        confusions.sort(key=lambda x: x[2], reverse=True)

        md.append("\n### Top Confusion Pairs")
        if confusions:
            for true_c, pred_c, cnt in confusions[:10]:
                md.append(f"- Actual {true_c} -> Predicted {pred_c}: **{cnt}** images")
        else:
            md.append("- No misclassifications observed.")

    md.append("\n## Final Champion Checkpoints Summary")
    for key, cfg in MODEL_CONFIGS.items():
        md.append(f"- **{key.upper()}**: {cfg['ckpt_path']}")

    report_md_path = os.path.join(REPORTS_DIR, "FINAL_TEST_EVALUATION.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved master report markdown -> {report_md_path}")

if __name__ == "__main__":
    if dry_verify():
        if "--run" in sys.argv:
            run_evaluation()
