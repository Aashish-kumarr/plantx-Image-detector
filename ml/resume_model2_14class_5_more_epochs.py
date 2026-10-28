r"""
Resume Model 2 - 14-Class Crop Classifier
==========================================
Resumes from Epoch 2 checkpoint and trains Epochs 3-7 (5 additional epochs).

Usage (from PowerShell):
    cd "C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
    python resume_model2_14class_5_more_epochs.py
"""

import os
import sys
import csv
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
sys.path.insert(0, BASE_DIR)

from src.models.model2_14class_resnet9 import create_model2_14class_resnet9
from src.datasets.model1_dsp_preprocessing import apply_dsp_preprocessing
from src.datasets.model2_transforms import get_model2_train_transform, get_model2_eval_transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Path-remapping dataset
# The CSV stores paths under the old location:
#   C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\...
# The data now lives at:
#   C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\...
# We remap transparently here without modifying any source file.
# ---------------------------------------------------------------------------
OLD_PATH_PREFIX = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"
NEW_PATH_PREFIX = BASE_DIR   # r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector"

def remap_image_path(raw_path: str) -> str:
    """Replace old base prefix with current BASE_DIR (case-insensitive match)."""
    if raw_path.lower().startswith(OLD_PATH_PREFIX.lower()):
        return NEW_PATH_PREFIX + raw_path[len(OLD_PATH_PREFIX):]
    return raw_path

class PathRemappedModel2Dataset(Dataset):
    """Reads model2_14class_split.csv and remaps stale image paths on the fly."""
    def __init__(self, csv_path: str, split_name: str, transform):
        df = pd.read_csv(csv_path)
        self.df        = df[df["split"] == split_name].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_path  = remap_image_path(str(row["image_path"]))
        label     = int(row["label_id"])
        pil_img   = Image.open(img_path).convert("RGB")
        dsp_pil   = apply_dsp_preprocessing(pil_img)
        return self.transform(dsp_pil), label

def build_model2_dataloaders(csv_path: str, batch_size: int, num_workers: int):
    """Return (train_loader, val_loader) with path-remapped dataset."""
    train_ds = PathRemappedModel2Dataset(csv_path, "train", get_model2_train_transform())
    val_ds   = PathRemappedModel2Dataset(csv_path, "val",   get_model2_eval_transform())
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESUME_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "model2_14class", "best_model2_14class.pth")
CHECKPOINT_DIR    = os.path.join(BASE_DIR, "checkpoints", "model2_14class")
RESULTS_DIR       = os.path.join(BASE_DIR, "results",     "model2_14class")
REPORTS_DIR       = os.path.join(BASE_DIR, "reports")

HISTORY_FILE          = os.path.join(RESULTS_DIR, "training_history.csv")
PER_CLASS_HISTORY_FILE= os.path.join(RESULTS_DIR, "training_history_per_class.csv")
CONFUSION_MATRIX_FILE = os.path.join(RESULTS_DIR, "model2_14class_confusion_matrix.png")
REPORT_FILE           = os.path.join(REPORTS_DIR,  "model2_14class_7epoch_report.txt")

RESUME_FROM_EPOCH  = 2        # already completed
NEXT_EPOCH         = 3        # first epoch we will train
FINAL_EPOCH        = 7        # last epoch we will train
TOTAL_EPOCHS       = FINAL_EPOCH   # used for progress display

RESUME_LR          = 5e-4
WEIGHT_DECAY       = 1e-4
BATCH_SIZE         = 64
NUM_WORKERS        = 0

KNOWN_BEST_F1      = 0.7415   # Epoch-2 best — guard rail
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "apple", "blueberry", "cherry", "corn", "grape", "orange", "peach",
    "pepper", "potato", "raspberry", "soybean", "squash", "strawberry", "tomato",
]

CLASS_WEIGHTS = torch.tensor([
    0.8399, 1.2224, 1.0980, 0.7703, 0.7421, 0.6420, 0.9224,
    0.9540, 1.0271, 2.4479, 0.6691, 1.1054, 1.2047, 0.3546,
], dtype=torch.float32)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(REPORTS_DIR,    exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path, model, optimizer, scaler, device):
    """Load checkpoint; return resumed_epoch and best_macro_f1."""
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])

    resumed_epoch = ckpt.get("epoch", RESUME_FROM_EPOCH)
    best_f1       = ckpt.get("best_macro_f1", KNOWN_BEST_F1)

    print(f"  Resumed from epoch : {resumed_epoch}")
    print(f"  Stored best F1     : {best_f1:.4f}")

    # Forcefully set LR to RESUME_LR for all param groups
    # (keeps momentum/state from optimizer, just adjusts step size)
    for pg in optimizer.param_groups:
        pg["lr"] = RESUME_LR
    print(f"  LR override to     : {RESUME_LR:.1e}  (optimizer momentum/state preserved)")

    return resumed_epoch, best_f1


def train_epoch(model, loader, criterion, optimizer, scaler, epoch, total_epochs, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)

    pbar = tqdm(
        total=n_batches,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=True,
        dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA={remaining} | {postfix}",
    )
    pbar.set_postfix(loss="0.000", acc="0.0%", lr=f"{RESUME_LR:.1e}")

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted  = logits.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{running_loss / (i + 1):.4f}",
                acc=f"{100. * correct / total:.1f}%",
                lr=f"{cur_lr:.1e}",
            )
        pbar.update(1)

    pbar.close()
    return running_loss / n_batches, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    n_batches = len(loader)

    with torch.inference_mode():
        pbar = tqdm(
            total=n_batches,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}<{remaining}",
        )
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted  = logits.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.update(1)
        pbar.close()

    val_loss = running_loss / n_batches
    val_acc  = 100.0 * correct / total

    label_ids = list(range(14))
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=label_ids, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return val_loss, val_acc, macro_prec, macro_rec, macro_f1, prec, rec, f1, all_labels, all_preds


def save_checkpoint(model, optimizer, scaler, epoch, best_f1, is_best):
    state = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "best_macro_f1":   best_f1,
        "label_mapping":   {k: i for i, k in enumerate(CLASSES)},
        "normalization":   {"mean": [0.1257, 0.1572, 0.0856],
                            "std":  [0.1620, 0.1985, 0.1149]},
        "class_weights":   CLASS_WEIGHTS.tolist(),
        "architecture":    "ResNet-9 (num_classes=14)",
    }
    last_path = os.path.join(CHECKPOINT_DIR, "last_model2_14class.pth")
    torch.save(state, last_path)
    print(f"  -> Saved last checkpoint  : {last_path}")

    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "best_model2_14class.pth")
        torch.save(state, best_path)
        print(f"  -> Saved BEST checkpoint  : {best_path}  (F1 improved)")


def append_history(epoch, train_loss, train_acc, val_loss, val_acc,
                   macro_prec, macro_rec, macro_f1, lr, epoch_time,
                   prec, rec, f1):
    # Main history row
    with open(HISTORY_FILE, "a", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            epoch, train_loss, train_acc,
            val_loss, val_acc, macro_prec, macro_rec, macro_f1,
            lr, epoch_time,
        ])

    # Per-class history row
    with open(PER_CLASS_HISTORY_FILE, "a", newline="") as fh:
        writer = csv.writer(fh)
        row = [epoch]
        for i in range(14):
            row.extend([prec[i], rec[i], f1[i]])
        writer.writerow(row)


def generate_confusion_matrix(model, val_loader, criterion, device, title_suffix=""):
    """Run one validation pass and save the confusion matrix."""
    _, _, _, _, _, _, _, _, all_labels, all_preds = validate(
        model, val_loader, criterion, device
    )
    cm   = confusion_matrix(all_labels, all_preds, labels=list(range(14)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(14, 14))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Model 2 (14-Class) Validation Confusion Matrix{title_suffix}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close()
    print(f"  -> Confusion matrix saved: {CONFUSION_MATRIX_FILE}")


# ---------------------------------------------------------------------------
# Main resume logic
# ---------------------------------------------------------------------------

def run_resume():
    print("=" * 68)
    print("  Model 2  — 14-Class Crop Classifier  |  Resume Epochs 3-7")
    print("=" * 68)
    print(f"Device          : {DEVICE}")
    print(f"Resume from     : Epoch {RESUME_FROM_EPOCH}")
    print(f"Next epoch      : {NEXT_EPOCH}")
    print(f"Final epoch     : {FINAL_EPOCH}")
    print(f"Resume LR       : {RESUME_LR:.1e}")
    print(f"Checkpoint      : {RESUME_CHECKPOINT}")
    print()

    # Verify checkpoint exists
    if not os.path.exists(RESUME_CHECKPOINT):
        raise FileNotFoundError(
            f"Resume checkpoint not found: {RESUME_CHECKPOINT}\n"
            "Run the 2-epoch training first."
        )

    # Build dataloaders using path-remapping dataset
    # (CSV stores old paths; remap_image_path() fixes them transparently)
    CSV_PATH = os.path.join(BASE_DIR, "metadata", "model2_14class_split.csv")
    print("Building dataloaders ...")
    print(f"  CSV path      : {CSV_PATH}")
    print(f"  Path remap    : {OLD_PATH_PREFIX}")
    print(f"            ->  : {NEW_PATH_PREFIX}")
    train_loader, val_loader = build_model2_dataloaders(
        csv_path=CSV_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}\n")

    # Build model and optimiser
    model     = create_model2_14class_resnet9().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=RESUME_LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))

    # Restore full checkpoint state
    resumed_epoch, best_f1 = load_checkpoint(
        RESUME_CHECKPOINT, model, optimizer, scaler, DEVICE
    )

    # Confirm the guard-rail
    assert resumed_epoch == RESUME_FROM_EPOCH, (
        f"Expected to resume from epoch {RESUME_FROM_EPOCH}, "
        f"but checkpoint says epoch {resumed_epoch}."
    )
    print(f"\nResume guard-rail OK: checkpoint epoch = {resumed_epoch}, best F1 = {best_f1:.4f}\n")

    # Scheduler — initialised fresh after restoring optimizer
    # (original 2-epoch run had no scheduler; we add it here)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    print(f"Scheduler       : ReduceLROnPlateau (min, factor=0.5, patience=1, min_lr=1e-6)\n")

    best_epoch   = resumed_epoch
    total_start  = time.time()

    # ------------------------------------------------------------------
    # Training loop — Epochs 3 to 7
    # ------------------------------------------------------------------
    for epoch in range(NEXT_EPOCH, FINAL_EPOCH + 1):
        epoch_start = time.time()

        print(f"\n{'='*50}")
        print(f"  EPOCH {epoch}/{TOTAL_EPOCHS}")
        print(f"{'='*50}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, TOTAL_EPOCHS, DEVICE
        )

        val_loss, val_acc, macro_prec, macro_rec, macro_f1, prec, rec, f1, all_labels, all_preds = validate(
            model, val_loader, criterion, DEVICE
        )

        epoch_time = time.time() - epoch_start
        cur_lr     = optimizer.param_groups[0]["lr"]

        # Scheduler step (monitors val_loss)
        scheduler.step(val_loss)
        next_lr = optimizer.param_groups[0]["lr"]

        is_best = macro_f1 > best_f1
        if is_best:
            best_f1    = macro_f1
            best_epoch = epoch

        save_checkpoint(model, optimizer, scaler, epoch, best_f1, is_best)
        append_history(
            epoch, train_loss, train_acc,
            val_loss, val_acc, macro_prec, macro_rec, macro_f1,
            cur_lr, epoch_time, prec, rec, f1,
        )

        # Console summary
        print(f"\n  Epoch {epoch} Summary")
        print(f"  Time            : {epoch_time:.1f}s")
        print(f"  LR this epoch   : {cur_lr:.2e}  |  next epoch: {next_lr:.2e}")
        print(f"  Train Loss      : {train_loss:.4f}  |  Train Acc : {train_acc:.2f}%")
        print(f"  Val   Loss      : {val_loss:.4f}  |  Val   Acc : {val_acc:.2f}%")
        print(f"  Macro F1        : {macro_f1:.4f}  |  Best F1   : {best_f1:.4f} (Epoch {best_epoch})")
        print(f"  {'*** NEW BEST ***' if is_best else '(no improvement)'}")

        # Highlighted crops
        potato_idx = CLASSES.index("potato")
        tomato_idx = CLASSES.index("tomato")
        rasp_idx   = CLASSES.index("raspberry")
        print(
            f"\n  Potato   -> Prec: {prec[potato_idx]:.4f} | Rec: {rec[potato_idx]:.4f} | F1: {f1[potato_idx]:.4f}  "
            f"(Ep2: Prec=0.3362 Rec=0.8914 F1=0.4882)"
        )
        print(
            f"  Tomato   -> Prec: {prec[tomato_idx]:.4f} | Rec: {rec[tomato_idx]:.4f} | F1: {f1[tomato_idx]:.4f}  "
            f"(Ep2: Prec=0.9175 Rec=0.7264 F1=0.8108)"
        )
        print(
            f"  Raspberry-> Prec: {prec[rasp_idx]:.4f}  | Rec: {rec[rasp_idx]:.4f}  | F1: {f1[rasp_idx]:.4f}"
        )

        print(f"\n  Per-class F1:")
        for i, cls in enumerate(CLASSES):
            marker = " <--" if cls in ("potato", "tomato", "raspberry") else ""
            print(f"    {cls:<12} -> Prec: {prec[i]:.4f} | Rec: {rec[i]:.4f} | F1: {f1[i]:.4f}{marker}")

    # ------------------------------------------------------------------
    # Post-training: confusion matrix from best checkpoint
    # ------------------------------------------------------------------
    total_time = time.time() - total_start

    print(f"\n{'='*68}")
    print(f"  Training complete  |  Best Epoch: {best_epoch}  |  Best F1: {best_f1:.4f}")
    print(f"  Total training time: {total_time:.1f}s")
    print(f"{'='*68}\n")

    print("Generating confusion matrix from best checkpoint ...")
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model2_14class.pth")
    best_ckpt      = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    generate_confusion_matrix(model, val_loader, criterion, DEVICE, f" — Best Epoch {best_epoch}")

    # ------------------------------------------------------------------
    # Final 7-epoch report
    # ------------------------------------------------------------------
    print("Writing 7-epoch report ...")
    write_report(best_epoch, best_f1, total_time)

    print(f"\nAll done. Report: {REPORT_FILE}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(best_epoch, best_f1, total_time):
    df_hist = pd.read_csv(HISTORY_FILE)
    df_pc   = pd.read_csv(PER_CLASS_HISTORY_FILE)

    lines = []
    lines.append("=" * 70)
    lines.append("  Model 2 — 14-Class Crop Classifier  |  7-Epoch Training Report")
    lines.append("=" * 70)
    lines.append(f"Best Epoch         : {best_epoch}")
    lines.append(f"Best Val Macro-F1  : {best_f1:.4f}")
    lines.append(f"Total Training Time: {total_time:.1f}s")
    lines.append(f"Checkpoint         : {os.path.join(CHECKPOINT_DIR, 'best_model2_14class.pth')}")
    lines.append(f"Confusion Matrix   : {CONFUSION_MATRIX_FILE}")
    lines.append("")

    # Val loss and LR progression table
    lines.append("Val-Loss & LR Progression")
    lines.append("-" * 50)
    lines.append(f"  {'Epoch':>5}  {'Val Loss':>10}  {'Val Acc%':>8}  {'MacroF1':>8}  {'LR':>10}")
    for _, row in df_hist.iterrows():
        marker = "  <-- BEST" if int(row["epoch"]) == best_epoch else ""
        lines.append(
            f"  {int(row['epoch']):>5}  {row['val_loss']:>10.4f}  "
            f"{row['val_accuracy']:>8.2f}  {row['val_macro_f1']:>8.4f}  "
            f"{row['learning_rate']:>10.2e}{marker}"
        )
    lines.append("")

    # Per-epoch detail
    lines.append("Per-Epoch Detail")
    lines.append("=" * 70)
    for _, row in df_hist.iterrows():
        epoch = int(row["epoch"])
        pc    = df_pc[df_pc["epoch"] == epoch].iloc[0]
        marker = "  *** BEST ***" if epoch == best_epoch else ""
        lines.append(f"\n--- Epoch {epoch}{marker} ---")
        lines.append(f"  LR       : {row['learning_rate']:.2e}")
        lines.append(f"  Train    : Loss={row['train_loss']:.4f}  Acc={row['train_accuracy']:.2f}%")
        lines.append(f"  Val      : Loss={row['val_loss']:.4f}  Acc={row['val_accuracy']:.2f}%")
        lines.append(
            f"  Macro    : Prec={row['val_macro_precision']:.4f}  "
            f"Rec={row['val_macro_recall']:.4f}  F1={row['val_macro_f1']:.4f}"
        )
        lines.append(f"  Time     : {row['epoch_time_seconds']:.1f}s")
        lines.append("  Per-class:")
        for cls in CLASSES:
            marker2 = " <--" if cls in ("potato", "tomato", "raspberry") else ""
            p = pc[f"{cls}_precision"]
            r = pc[f"{cls}_recall"]
            f = pc[f"{cls}_f1"]
            lines.append(f"    {cls:<12} Prec={p:.4f}  Rec={r:.4f}  F1={f:.4f}{marker2}")

    # Potato & Tomato across epochs
    lines.append("\n")
    lines.append("=" * 70)
    lines.append("Potato Metrics Across All 7 Epochs")
    lines.append("-" * 50)
    lines.append(f"  {'Epoch':>5}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
    for _, row in df_pc.iterrows():
        epoch = int(row["epoch"])
        lines.append(
            f"  {epoch:>5}  {row['potato_precision']:>8.4f}  "
            f"{row['potato_recall']:>8.4f}  {row['potato_f1']:>8.4f}"
        )

    lines.append("")
    lines.append("Tomato Metrics Across All 7 Epochs")
    lines.append("-" * 50)
    lines.append(f"  {'Epoch':>5}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
    for _, row in df_pc.iterrows():
        epoch = int(row["epoch"])
        lines.append(
            f"  {epoch:>5}  {row['tomato_precision']:>8.4f}  "
            f"{row['tomato_recall']:>8.4f}  {row['tomato_f1']:>8.4f}"
        )

    lines.append("")
    lines.append("RESUME EPOCHS 3-7 COMPLETE")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_resume()
    except Exception:
        print("\n--- TRAINING FAILED ---")
        traceback.print_exc()
        sys.exit(1)
