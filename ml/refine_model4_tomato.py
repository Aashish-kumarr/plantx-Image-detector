
import os
import sys
import csv
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
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

from src.models.model4_tomato_resnet9 import create_model4_tomato_resnet9
from src.datasets.model1_dsp_preprocessing import apply_dsp_preprocessing
from src.datasets.model4_tomato_transforms import get_model4_train_transform, get_model4_eval_transform

# ---------------------------------------------------------------------------
# Path-remapping (CSV may contain old base dir before workspace move)
# ---------------------------------------------------------------------------
OLD_PREFIXES = [
    r"C:\Users\ak021\Desktop\plantDx\plant image detector",
    r"c:\Users\ak021\Desktop\plant image detector",
    r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector",
]
NEW_PREFIX = BASE_DIR

def remap_image_path(raw_path: str) -> str:
    for old in OLD_PREFIXES:
        if raw_path.lower().startswith(old.lower()):
            return NEW_PREFIX + raw_path[len(old):]
    return raw_path

class PathRemappedModel4Dataset(Dataset):
    def __init__(self, csv_path: str, split_name: str, transform):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split_name].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = remap_image_path(str(row["image_path"]))
        label = int(row["label_id"])
        pil_img = Image.open(img_path).convert("RGB")
        dsp_pil = apply_dsp_preprocessing(pil_img)
        return self.transform(dsp_pil), label

def build_model4_dataloaders(csv_path: str, batch_size: int, num_workers: int):
    train_ds = PathRemappedModel4Dataset(csv_path, "train", get_model4_train_transform())
    val_ds   = PathRemappedModel4Dataset(csv_path, "val",   get_model4_eval_transform())
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIGINAL_CKPT   = os.path.join(BASE_DIR, "checkpoints", "model4_tomato", "best_model4_tomato.pth")
CSV_PATH        = os.path.join(BASE_DIR, "metadata", "model4_tomato_split.csv")

CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints", "model4_tomato_refine")
RESULTS_DIR     = os.path.join(BASE_DIR, "results",     "model4_tomato_refine")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports",     "model4_tomato_refine")

BEST_CKPT_PATH  = os.path.join(CHECKPOINT_DIR, "best_model4_tomato_refine.pth")
LAST_CKPT_PATH  = os.path.join(CHECKPOINT_DIR, "last_model4_tomato_refine.pth")
HISTORY_FILE    = os.path.join(RESULTS_DIR,    "training_history.csv")
CM_FILE         = os.path.join(RESULTS_DIR,    "confusion_matrix.png")
REPORT_FILE     = os.path.join(REPORTS_DIR,    "model4_refinement_report.txt")

# Guard-rails
EXPECTED_EPOCH  = 4
EXPECTED_F1     = 0.7661

# Training config
MAX_REFINE_EPOCHS = 7
REFINE_LR         = 2e-4
WEIGHT_DECAY      = 1e-4
BATCH_SIZE        = 64
NUM_WORKERS       = 0
ES_PATIENCE       = 3

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "healthy",
    "early_blight",
    "late_blight",
    "bacterial_spot",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites",
    "target_spot",
    "tomato_mosaic_virus",
    "tomato_yellow_leaf_curl_virus",
]
NUM_CLASSES = len(CLASS_NAMES)

# Verified class weights from original checkpoint
CLASS_WEIGHTS = torch.tensor([
    0.9151,   # healthy
    1.1519,   # early_blight
    0.8384,   # late_blight
    0.7980,   # bacterial_spot
    1.1880,   # leaf_mold
    0.8757,   # septoria_leaf_spot
    0.8896,   # spider_mites
    0.9689,   # target_spot
    1.8759,   # tomato_mosaic_virus
    0.4986,   # tomato_yellow_leaf_curl_virus
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"REFINE EPOCH {epoch}/{MAX_REFINE_EPOCHS} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / total, correct / total * 100.0

@torch.no_grad()
def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f"REFINE EPOCH {epoch}/{MAX_REFINE_EPOCHS} [val]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    val_loss = running_loss / total
    val_acc  = correct / total * 100.0
    mac_prec, mac_rec, mac_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(NUM_CLASSES), zero_division=0
    )
    return val_loss, val_acc, mac_prec, mac_rec, mac_f1, pc_prec, pc_rec, pc_f1, all_labels, all_preds

def print_epoch_metrics(epoch, t_loss, t_acc, v_loss, v_acc, m_prec, m_rec, m_f1, pc_prec, pc_rec, pc_f1, lr):
    print(f"\n[REFINE EPOCH {epoch}/{MAX_REFINE_EPOCHS}]")
    print(f"  LR         : {lr:.2e}")
    print(f"  Train Loss : {t_loss:.4f} | Train Acc : {t_acc:.2f}%")
    print(f"  Val Loss   : {v_loss:.4f} | Val Acc   : {v_acc:.2f}%")
    print(f"  Macro  P={m_prec:.4f}  R={m_rec:.4f}  F1={m_f1:.4f}")
    print(f"\n  Per-Class Metrics:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<35} P={pc_prec[i]:.4f}  R={pc_rec[i]:.4f}  F1={pc_f1[i]:.4f}")
    # Bottom 3
    ranked = sorted(range(NUM_CLASSES), key=lambda i: pc_f1[i])
    print(f"\n  Weakest classes this epoch:")
    for rank, idx in enumerate(ranked[:3], 1):
        print(f"    {rank}. {CLASS_NAMES[idx]:<35} F1={pc_f1[idx]:.4f}")

def write_history_row(epoch, t_loss, t_acc, v_loss, v_acc, m_prec, m_rec, m_f1, lr, epoch_time, pc_prec, pc_rec, pc_f1):
    file_exists = os.path.exists(HISTORY_FILE)
    headers = [
        "refine_epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall",
        "val_macro_f1", "learning_rate", "epoch_time_seconds",
    ]
    for name in CLASS_NAMES:
        headers += [f"{name}_precision", f"{name}_recall", f"{name}_f1"]
    row = [epoch, t_loss, t_acc, v_loss, v_acc, m_prec, m_rec, m_f1, lr, epoch_time]
    for i in range(NUM_CLASSES):
        row += [pc_prec[i], pc_rec[i], pc_f1[i]]
    with open(HISTORY_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

def save_checkpoint(path, model, optimizer, scaler, scheduler,
                    refine_epoch, best_f1, val_acc, val_loss, ckpt_source):
    torch.save({
        "refine_epoch":          refine_epoch,
        "source_original_epoch": EXPECTED_EPOCH,
        "model_state":           model.state_dict(),
        "optimizer_state":       optimizer.state_dict(),
        "scaler_state":          scaler.state_dict(),
        "scheduler_state":       scheduler.state_dict(),
        "current_lr":            optimizer.param_groups[0]["lr"],
        "best_val_macro_f1":     best_f1,
        "validation_accuracy":   val_acc,
        "validation_loss":       val_loss,
        "class_mapping":         {c: i for i, c in enumerate(CLASS_NAMES)},
        "class_weights":         CLASS_WEIGHTS.tolist(),
        "normalization":         ckpt_source.get("normalization"),
        "architecture":          ckpt_source.get("architecture"),
    }, path)

def generate_cm_and_report(model, val_loader, criterion, device, best_f1, best_acc, best_epoch):
    # Reload the best checkpoint
    best_ckpt = torch.load(BEST_CKPT_PATH, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    _, _, m_prec, m_rec, m_f1, pc_prec, pc_rec, pc_f1, all_labels, all_preds = validate_epoch(
        model, val_loader, criterion, device, epoch="best"
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Model 4 - Tomato Disease (Refinement Best)")
    plt.tight_layout()
    plt.savefig(CM_FILE)
    plt.close()
    print(f"Confusion matrix saved to {CM_FILE}")

    # Top-5 confusions
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusions.append({"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "count": int(cm[i, j])})
    confusions.sort(key=lambda x: x["count"], reverse=True)

    # Report
    lines = [
        "Model 4 - Tomato Disease Refinement Report",
        "=" * 48,
        "",
        "--- Baseline: Original Best Epoch 4 ---",
        f"Val Accuracy : 80.73%",
        f"Val Macro-F1 : 0.7661",
        "",
        f"--- Best Refinement Epoch: {best_epoch} ---",
        f"Val Accuracy   : {best_acc:.2f}%",
        f"Val Macro-P    : {m_prec:.4f}",
        f"Val Macro-R    : {m_rec:.4f}",
        f"Val Macro-F1   : {m_f1:.4f}",
        f"Val Loss       : {best_ckpt['validation_loss']:.4f}",
        "",
        f"Absolute F1 improvement : {m_f1 - 0.7661:+.4f}",
        f"Is refinement better?   : {'YES' if m_f1 > 0.7661 else 'NO (failed to beat baseline)'}",
        "",
        "--- Per-Class Metrics (Best Refinement) ---",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:<35} P={pc_prec[i]:.4f}  R={pc_rec[i]:.4f}  F1={pc_f1[i]:.4f}")
    lines.append("")
    lines.append("--- Top 5 Confusion Pairs ---")
    for c in confusions[:5]:
        lines.append(f"  actual {c['true']:<35} -> predicted {c['pred']:<35} : {c['count']}")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {REPORT_FILE}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_refine():
    print("=" * 68)
    print("  Model 4 - Tomato Disease Classifier  |  Refinement Training")
    print("=" * 68)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")

    # Create directories
    for d in [CHECKPOINT_DIR, RESULTS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------------
    # LOAD + VALIDATE CHECKPOINT
    # -----------------------------------------------------------------------
    if not os.path.exists(ORIGINAL_CKPT):
        raise FileNotFoundError(f"Original checkpoint not found: {ORIGINAL_CKPT}")

    print(f"\nLoading : {ORIGINAL_CKPT}")
    ckpt = torch.load(ORIGINAL_CKPT, map_location="cpu")

    ckpt_epoch = ckpt.get("epoch", -1)
    ckpt_f1    = ckpt.get("best_macro_f1", 0.0)
    print(f"  Checkpoint epoch     : {ckpt_epoch}")
    print(f"  Checkpoint Macro-F1  : {ckpt_f1:.4f}")

    if ckpt_epoch != EXPECTED_EPOCH:
        raise RuntimeError(f"GUARD-RAIL FAILED: checkpoint epoch={ckpt_epoch}, expected={EXPECTED_EPOCH}")
    if abs(ckpt_f1 - EXPECTED_F1) > 0.002:
        raise RuntimeError(f"GUARD-RAIL FAILED: checkpoint F1={ckpt_f1:.4f}, expected~={EXPECTED_F1}")
    print("  Guard-rails PASSED.")

    # -----------------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------------
    model = create_model4_tomato_resnet9().to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model strict-load OK | params={total_params:,}")

    # -----------------------------------------------------------------------
    # LOSS, OPTIMIZER, SCALER, SCHEDULER
    # -----------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=REFINE_LR, weight_decay=WEIGHT_DECAY)
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print("  Optimizer state restored.")
    for pg in optimizer.param_groups:
        pg["lr"] = REFINE_LR
    print(f"  Optimizer LR forced to {REFINE_LR}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
        print("  AMP scaler state restored.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )

    # -----------------------------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------------------------
    print(f"\nBuilding dataloaders from {CSV_PATH} ...")
    train_loader, val_loader = build_model4_dataloaders(CSV_PATH, BATCH_SIZE, NUM_WORKERS)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # -----------------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------------
    best_f1          = EXPECTED_F1
    best_acc         = 80.73
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(1, MAX_REFINE_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"  REFINE EPOCH {epoch}/{MAX_REFINE_EPOCHS}")
        print(f"{'='*50}")

        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        v_loss, v_acc, m_prec, m_rec, m_f1, pc_prec, pc_rec, pc_f1, all_labels, all_preds = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print_epoch_metrics(epoch, t_loss, t_acc, v_loss, v_acc, m_prec, m_rec, m_f1, pc_prec, pc_rec, pc_f1, current_lr)

        write_history_row(epoch, t_loss, t_acc, v_loss, v_acc, m_prec, m_rec, m_f1, current_lr, epoch_time, pc_prec, pc_rec, pc_f1)

        scheduler.step(v_loss)
        next_lr = optimizer.param_groups[0]["lr"]
        print(f"  Scheduler step done. Next LR: {next_lr:.2e}")

        # Save last
        save_checkpoint(LAST_CKPT_PATH, model, optimizer, scaler, scheduler,
                        epoch, m_f1 if m_f1 > best_f1 else best_f1, v_acc, v_loss, ckpt)

        if m_f1 > best_f1:
            best_f1 = m_f1
            best_acc = v_acc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(BEST_CKPT_PATH, model, optimizer, scaler, scheduler,
                            epoch, best_f1, v_acc, v_loss, ckpt)
            print(f"  *** NEW BEST (F1={best_f1:.4f}) saved to best checkpoint ***")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience {patience_counter}/{ES_PATIENCE}")
            if patience_counter >= ES_PATIENCE:
                print("\n  Early stopping triggered.")
                break

    print("\n" + "="*68)
    print("  Refinement training finished.")
    print(f"  Best refine epoch : {best_epoch}  |  Best F1 : {best_f1:.4f}")

    if best_epoch > 0 and os.path.exists(BEST_CKPT_PATH):
        generate_cm_and_report(model, val_loader, criterion, device, best_f1, best_acc, best_epoch)
    else:
        print("  Baseline was NOT beaten. No best refinement checkpoint saved.")

if __name__ == "__main__":
    run_refine()
