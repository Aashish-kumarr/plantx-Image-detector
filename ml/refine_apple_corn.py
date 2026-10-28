"""
refine_apple_corn.py

PlantDx – Refinement launcher for Model 5 (Apple) and Model 7 (Corn).

Usage (start OR resume – same command):
    python refine_apple_corn.py

Safe interruption: Ctrl+C cleanly stops the queue.
Completed models are skipped automatically on re-run.
"""

import os
import sys
import json
import time
import gc
import traceback
import csv
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.generic_disease_dataset import GenericDiseaseDataset
from src.models.model1_resnet9 import ResNet9Model1

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
CONFIG_FILE = WORKSPACE / "metadata" / "models5_11_training_config.json"

REFINE_MODELS = [
    {
        "model_number": 5,
        "model_name": "model5_apple",
        "source_checkpoint": WORKSPACE / "checkpoints" / "model5_apple" / "best_model.pth",
        "refine_dir":        WORKSPACE / "checkpoints" / "model5_apple_refine",
        "log_dir":           WORKSPACE / "logs" / "model5_apple_refine",
        "results_dir":       WORKSPACE / "results" / "model5_apple_refine",
        "expected_epoch": 2,
        "expected_f1":    0.7583,
        "expected_acc":   81.01,
        "expected_classes": 4,
    },
    {
        "model_number": 7,
        "model_name": "model7_corn",
        "source_checkpoint": WORKSPACE / "checkpoints" / "model7_corn" / "best_model.pth",
        "refine_dir":        WORKSPACE / "checkpoints" / "model7_corn_refine",
        "log_dir":           WORKSPACE / "logs" / "model7_corn_refine",
        "results_dir":       WORKSPACE / "results" / "model7_corn_refine",
        "expected_epoch": 5,
        "expected_f1":    0.7620,
        "expected_acc":   80.73,
        "expected_classes": 4,
    },
]

REFINE_LR       = 2e-4
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 7
PATIENCE        = 3
BATCH_SIZE      = 64
TOL             = 0.005       # floating-point tolerance for guardrails


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def atomic_save(state: dict, path: Path):
    dir_name = path.parent
    with NamedTemporaryFile(delete=False, dir=str(dir_name), suffix=".pth") as tmp:
        torch.save(state, tmp.name)
        tmp_name = tmp.name
    shutil.move(tmp_name, str(path))


def load_config(model_name: str) -> dict:
    with open(CONFIG_FILE) as f:
        configs = json.load(f)
    for c in configs:
        if c["model_name"] == model_name:
            return c
    raise KeyError(f"Config for {model_name} not found.")


def cleanup_cuda(model=None, optimizer=None, scaler=None):
    for obj in (model, optimizer, scaler):
        if obj is not None:
            del obj
    gc.collect()
    torch.cuda.empty_cache()


def bar(pct: float, width: int = 22) -> str:
    filled = int(pct * width)
    return "█" * filled + "─" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail
# ─────────────────────────────────────────────────────────────────────────────
def verify_source_checkpoint(rm: dict) -> dict:
    ckpt_path = rm["source_checkpoint"]
    print(f"\n  Loading {ckpt_path} ...")
    state = torch.load(str(ckpt_path), map_location="cpu")

    issues = []
    if state.get("num_classes") != rm["expected_classes"]:
        issues.append(f"num_classes {state.get('num_classes')} != {rm['expected_classes']}")
    if abs(state.get("best_macro_f1", -1) - rm["expected_f1"]) > TOL:
        issues.append(f"best_macro_f1 {state.get('best_macro_f1')} ~ {rm['expected_f1']} (tol={TOL})")
    if abs(state.get("validation_accuracy", -1) - rm["expected_acc"]) > TOL * 100:
        issues.append(f"validation_accuracy {state.get('validation_accuracy')} ~ {rm['expected_acc']}")

    if issues:
        print(f"  GUARDRAIL FAILED for {rm['model_name']}:")
        for i in issues:
            print(f"    ✗ {i}")
        raise RuntimeError(f"Checkpoint guardrail failed: {issues}")

    print(f"  GUARDRAIL PASSED")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Refinement state
# ─────────────────────────────────────────────────────────────────────────────
def completion_path(rm: dict) -> Path:
    return rm["refine_dir"] / "refine_complete.json"


def is_complete(rm: dict) -> bool:
    return completion_path(rm).exists()


def write_completion(rm: dict, src_f1: float, best_ref_f1: float, best_ref_acc: float,
                     best_ref_epoch: int, stop_reason: str, duration: float):
    preferred = "REFINED" if best_ref_f1 > src_f1 else "ORIGINAL"
    marker = {
        "model": rm["model_name"],
        "status": "COMPLETED",
        "source_epoch": rm["expected_epoch"],
        "source_macro_f1": src_f1,
        "best_refine_epoch": best_ref_epoch,
        "best_refine_macro_f1": best_ref_f1,
        "best_refine_val_accuracy": best_ref_acc,
        "delta": round(best_ref_f1 - src_f1, 6),
        "preferred_champion": preferred,
        "stop_reason": stop_reason,
        "duration_seconds": round(duration, 1),
        "best_refined_checkpoint": str(rm["refine_dir"] / "best_refined_model.pth"),
    }
    with open(str(completion_path(rm)), "w") as f:
        json.dump(marker, f, indent=4)
    return marker


def load_last_refine(rm: dict):
    last_path = rm["refine_dir"] / "last_refine_checkpoint.pth"
    if last_path.exists():
        return torch.load(str(last_path), map_location="cpu")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(all_labels, all_preds, num_classes, classes):
    labels_list = list(range(num_classes))
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=labels_list, zero_division=0)
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0)

    print("\n  Per-class metrics:")
    for i, cls in enumerate(classes):
        print(f"    {cls:40s}  P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}  sup={int(sup[i])}")
    print(f"\n  Macro  → P={m_prec:.4f}  R={m_rec:.4f}  F1={m_f1:.4f}")
    print(f"  Weighted F1 = {w_f1:.4f}")

    return m_f1, w_f1, prec, rec, f1, sup


# ─────────────────────────────────────────────────────────────────────────────
# Core refinement engine
# ─────────────────────────────────────────────────────────────────────────────
def refine_model(rm: dict, device: str = "cuda"):
    cfg    = load_config(rm["model_name"])
    src_state = verify_source_checkpoint(rm)
    src_f1 = src_state["best_macro_f1"]

    rm["refine_dir"].mkdir(parents=True, exist_ok=True)
    rm["log_dir"].mkdir(parents=True, exist_ok=True)
    rm["results_dir"].mkdir(parents=True, exist_ok=True)

    num_classes = cfg["num_classes"]
    classes = [cfg["class_mapping"][str(i)] for i in range(num_classes)]
    cw = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_ds = GenericDiseaseDataset(
        cfg["train_csv"], is_train=True,
        mean=cfg["normalization_mean"], std=cfg["normalization_std"])
    val_ds = GenericDiseaseDataset(
        cfg["train_csv"], is_train=False,
        mean=cfg["normalization_mean"], std=cfg["normalization_std"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ResNet9Model1(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(src_state["model_state"])

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=REFINE_LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5)

    # Optionally restore optimizer/scaler
    if "optimizer_state" in src_state:
        try:
            optimizer.load_state_dict(src_state["optimizer_state"])
            for pg in optimizer.param_groups:
                pg["lr"] = REFINE_LR
            print("  Optimizer state restored, LR overridden to", REFINE_LR)
        except Exception:
            print("  Optimizer state incompatible – using fresh optimizer at LR", REFINE_LR)
    if "scaler_state" in src_state:
        try:
            scaler.load_state_dict(src_state["scaler_state"])
            print("  Scaler state restored.")
        except Exception:
            print("  Scaler state incompatible – using fresh scaler.")

    # ── Resume from last refine checkpoint if available ──────────────────────
    start_epoch   = 1
    best_f1       = -1.0
    best_epoch    = -1
    es_counter    = 0
    last_val_acc  = 0.0

    last_state = load_last_refine(rm)
    if last_state is not None:
        model.load_state_dict(last_state["model_state"])
        optimizer.load_state_dict(last_state["optimizer_state"])
        scaler.load_state_dict(last_state["scaler_state"])
        scheduler.load_state_dict(last_state["scheduler_state"])
        start_epoch = last_state["refine_epoch"] + 1
        best_f1     = last_state["best_refined_macro_f1"]
        best_epoch  = last_state["best_refine_epoch"]
        es_counter  = last_state["early_stopping_counter"]
        print(f"  Resumed from refine epoch {last_state['refine_epoch']} (best F1={best_f1:.4f})")

    # ── History CSV ──────────────────────────────────────────────────────────
    hist_path = rm["log_dir"] / "training_history.csv"
    if start_epoch == 1 or not hist_path.exists():
        with open(str(hist_path), "w", newline="") as f:
            csv.writer(f).writerow([
                "refine_epoch", "train_loss", "train_accuracy",
                "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1",
                "learning_rate", "epoch_duration_seconds", "best_so_far", "patience"
            ])

    total_start = time.time()
    stop_reason = "max_epochs"

    for refine_epoch in range(start_epoch, MAX_EPOCHS + 1):
        ep_start = time.time()
        print(f"\n{'='*60}")
        print(f"{rm['model_name'].upper()}  |  REFINE EPOCH {refine_epoch}/{MAX_EPOCHS}")
        print(f"{'='*60}")
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Current LR: {lr_now:.2e}")

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        run_loss, correct, total_imgs = 0.0, 0, 0
        n_batches = len(train_loader)
        t_loop = time.time()

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss    += loss.item()
            _, pred      = logits.max(1)
            total_imgs  += labels.size(0)
            correct     += pred.eq(labels).sum().item()

            if (i + 1) % max(1, n_batches // 5) == 0 or (i + 1) == n_batches:
                pct     = (i + 1) / n_batches
                elapsed = time.time() - t_loop
                eta     = (elapsed / (i + 1)) * (n_batches - i - 1)
                print(f"\r  [{bar(pct)}] {pct*100:5.1f}%  "
                      f"batch {i+1}/{n_batches}  "
                      f"loss={run_loss/(i+1):.4f}  "
                      f"acc={100.*correct/total_imgs:.2f}%  "
                      f"elapsed={int(elapsed)}s  ETA={int(eta)}s")

        train_loss = run_loss / n_batches
        train_acc  = 100. * correct / total_imgs

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        vrun_loss, vcorrect, vtotal = 0.0, 0, 0
        all_preds, all_labels = [], []
        vn = len(val_loader)
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(imgs)
                    vloss  = criterion(logits, labels)
                vrun_loss += vloss.item()
                _, pred = logits.max(1)
                vtotal   += labels.size(0)
                vcorrect += pred.eq(labels).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = vrun_loss / vn
        val_acc  = 100. * vcorrect / vtotal

        m_f1, w_f1, prec, rec, f1, sup = compute_metrics(all_labels, all_preds, num_classes, classes)

        scheduler.step(m_f1)
        lr_after = optimizer.param_groups[0]["lr"]
        ep_dur   = time.time() - ep_start

        is_best = m_f1 > best_f1
        if is_best:
            best_f1      = m_f1
            best_epoch   = refine_epoch
            last_val_acc = val_acc
            es_counter   = 0
        else:
            es_counter += 1

        # ── Save last checkpoint ─────────────────────────────────────────────
        ckpt_base = {
            "source_checkpoint": str(rm["source_checkpoint"]),
            "source_epoch":      rm["expected_epoch"],
            "source_val_acc":    rm["expected_acc"],
            "source_macro_f1":   src_f1,
            "refine_epoch":      refine_epoch,
            "model_state":       {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state":   optimizer.state_dict(),
            "scaler_state":      scaler.state_dict(),
            "scheduler_state":   scheduler.state_dict(),
            "current_lr":        lr_after,
            "best_refined_macro_f1":     best_f1,
            "best_refine_epoch":         best_epoch,
            "early_stopping_counter":    es_counter,
            "validation_accuracy":       val_acc,
            "validation_macro_f1":       m_f1,
            "validation_weighted_f1":    w_f1,
            "class_mapping":     cfg["class_mapping"],
            "label_to_index":    cfg["label_to_index"],
            "normalization": {
                "mean": cfg["normalization_mean"],
                "std":  cfg["normalization_std"],
            },
            "class_weights":  cfg["class_weights"],
            "architecture":   cfg["architecture"],
            "num_classes":    num_classes,
            "image_size":     cfg["image_size"],
            "training_configuration": cfg,
        }
        atomic_save(ckpt_base, rm["refine_dir"] / "last_refine_checkpoint.pth")
        if is_best:
            atomic_save(ckpt_base, rm["refine_dir"] / "best_refined_model.pth")
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                str(rm["results_dir"] / "validation_confusion_matrix.csv"))

        # ── Log history ──────────────────────────────────────────────────────
        with open(str(hist_path), "a", newline="") as f:
            csv.writer(f).writerow([
                refine_epoch, train_loss, train_acc, val_loss, val_acc,
                m_f1, w_f1, lr_after, ep_dur, best_f1, es_counter
            ])

        # ── Print epoch summary ──────────────────────────────────────────────
        print(f"\n  ── Epoch {refine_epoch} Summary ─────────────────")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")
        print(f"  Val Macro-F1: {m_f1:.4f}  Weighted-F1: {w_f1:.4f}")
        print(f"  LR: {lr_now:.2e} → {lr_after:.2e}")
        print(f"  Best Refined F1: {best_f1:.4f}  (Epoch {best_epoch})")
        print(f"  Early-stop patience: {es_counter}/{PATIENCE}")
        print(f"  Epoch duration: {ep_dur:.1f}s")

        if es_counter >= PATIENCE:
            print(f"\n  Early stopping triggered after refine epoch {refine_epoch}.")
            stop_reason = "early_stopping"
            break

    duration = time.time() - total_start
    marker = write_completion(rm, src_f1, best_f1, last_val_acc, best_epoch, stop_reason, duration)

    print(f"\n{'='*60}")
    print(f"  {rm['model_name'].upper()} REFINEMENT COMPLETE")
    print(f"  Source F1:  {src_f1:.4f}")
    print(f"  Refined F1: {best_f1:.4f}  (Δ {best_f1 - src_f1:+.4f})")
    print(f"  Preferred champion: {marker['preferred_champion']}")
    print(f"{'='*60}")

    cleanup_cuda(model, optimizer, scaler)
    return marker


# ─────────────────────────────────────────────────────────────────────────────
# Master launcher
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    queue_file = WORKSPACE / "checkpoints" / "models5_7_refine_queue_state.json"

    def save_queue(state):
        with open(str(queue_file), "w") as f:
            json.dump(state, f, indent=4)

    state = {
        "queue": [rm["model_name"] for rm in REFINE_MODELS],
        "completed": [],
        "current": None,
        "interrupted": None,
        "failed": [],
        "start_timestamp": time.time(),
    }
    if queue_file.exists():
        with open(str(queue_file)) as f:
            saved = json.load(f)
        state["completed"] = saved.get("completed", [])
        state["failed"]    = saved.get("failed", [])

    print("\nREFINEMENT QUEUE")
    for rm in REFINE_MODELS:
        if rm["model_name"] in state["completed"]:
            status = "[✓]"
        elif rm["model_name"] in state["failed"]:
            status = "[x]"
        else:
            status = "[ ]"
        print(f"  {status} {rm['model_name']}")

    markers = []
    try:
        for rm in REFINE_MODELS:
            mname = rm["model_name"]
            if mname in state["completed"]:
                print(f"\n  Skipping {mname} (already complete).")
                continue
            if is_complete(rm):
                print(f"\n  Skipping {mname} (completion marker exists).")
                state["completed"].append(mname)
                save_queue(state)
                continue

            state["current"] = mname
            save_queue(state)

            try:
                marker = refine_model(rm, device=device)
                markers.append(marker)
                state["completed"].append(mname)
                state["current"] = None
                save_queue(state)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                print(f"\n  ERROR refining {mname}: {e}")
                print(tb)
                log_file = rm["log_dir"] / "refinement_error.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(tb)
                state["failed"].append(mname)
                state["current"] = None
                save_queue(state)
                cleanup_cuda()

    except KeyboardInterrupt:
        interrupted = state.get("current")
        state["interrupted"] = interrupted
        state["current"] = None
        save_queue(state)
        print("\n============================================================")
        print("TRAINING INTERRUPTED BY USER")
        print(f"  Active model : {interrupted}")
        print("  Resume available: YES  (re-run same command)")
        print("============================================================")
        cleanup_cuda()
        sys.exit(0)

    # ── Final comparison report ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    report_lines = [
        "# Models 5 & 7 Refinement Report",
        "",
        "## Source Checkpoints",
    ]
    for rm in REFINE_MODELS:
        report_lines += [
            f"### {rm['model_name']}",
            f"- Source epoch   : {rm['expected_epoch']}",
            f"- Source Val Acc : {rm['expected_acc']}%",
            f"- Source Macro-F1: {rm['expected_f1']}",
            "",
        ]

    report_lines += ["## Refinement Results", ""]

    for m in markers:
        delta = m["delta"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {m['model']}")
        print(f"    Original F1 : {m['source_macro_f1']:.4f}")
        print(f"    Refined  F1 : {m['best_refine_macro_f1']:.4f}")
        print(f"    Δ           : {sign}{delta:.4f}")
        print(f"    Champion    : {m['preferred_champion']}")
        print()
        report_lines += [
            f"### {m['model']}",
            f"- Original Macro-F1 : {m['source_macro_f1']:.4f}",
            f"- Best Refined F1   : {m['best_refine_macro_f1']:.4f}",
            f"- Delta             : {sign}{delta:.4f}",
            f"- Best Refine Epoch : {m['best_refine_epoch']}",
            f"- Stop Reason       : {m['stop_reason']}",
            f"- Duration (s)      : {m['duration_seconds']}",
            f"- Preferred Champion: **{m['preferred_champion']}**",
            "",
        ]

    report_lines += [
        "## Test Set Status",
        "Test sets remain **LOCKED**. No test data was used during refinement.",
    ]

    report_path = WORKSPACE / "reports" / "MODELS_5_AND_7_REFINEMENT_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    print(f"  Report saved: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
