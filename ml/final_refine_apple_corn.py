"""
final_refine_apple_corn.py

PlantDx – FINAL low-LR refinement stage for Model 5 (Apple) and Model 7 (Corn).

Source: best_refined_model.pth from the previous refinement stage.
LR forced to 1e-4. Max 5 epochs. Early-stop patience 3 on Val Macro-F1.

Usage (start OR resume – same command):
    python final_refine_apple_corn.py

Safe interruption: Ctrl+C stops cleanly. Same command resumes.
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
WORKSPACE   = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
CONFIG_FILE = WORKSPACE / "metadata" / "models5_11_training_config.json"

FINAL_LR     = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 5
PATIENCE     = 3
BATCH_SIZE   = 64
TOL          = 0.005   # guardrail floating-point tolerance

REFINE_MODELS = [
    {
        "model_number"  : 5,
        "cfg_name"      : "model5_apple",
        "source_ckpt"   : WORKSPACE / "checkpoints" / "model5_apple_refine"  / "best_refined_model.pth",
        "final_dir"     : WORKSPACE / "checkpoints" / "model5_apple_final_refine",
        "log_dir"       : WORKSPACE / "logs"        / "model5_apple_final_refine",
        "results_dir"   : WORKSPACE / "results"     / "model5_apple_final_refine",
        "prev_f1"       : 0.8907,
        "prev_acc"      : 90.19,
        "prev_epoch"    : 4,
        "prev_classes"  : 4,
    },
    {
        "model_number"  : 7,
        "cfg_name"      : "model7_corn",
        "source_ckpt"   : WORKSPACE / "checkpoints" / "model7_corn_refine"   / "best_refined_model.pth",
        "final_dir"     : WORKSPACE / "checkpoints" / "model7_corn_final_refine",
        "log_dir"       : WORKSPACE / "logs"        / "model7_corn_final_refine",
        "results_dir"   : WORKSPACE / "results"     / "model7_corn_final_refine",
        "prev_f1"       : 0.8841,
        "prev_acc"      : 90.62,
        "prev_epoch"    : 7,
        "prev_classes"  : 4,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def atomic_save(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".pth") as tmp:
        torch.save(state, tmp.name)
        tmp_name = tmp.name
    shutil.move(tmp_name, str(path))


def load_config(cfg_name: str) -> dict:
    with open(CONFIG_FILE) as f:
        configs = json.load(f)
    for c in configs:
        if c["model_name"] == cfg_name:
            return c
    raise KeyError(f"Config for {cfg_name} not found.")


def cleanup_cuda(*objs):
    for o in objs:
        if o is not None:
            del o
    gc.collect()
    torch.cuda.empty_cache()


def bar(pct: float, width: int = 24) -> str:
    filled = int(pct * width)
    return "█" * filled + "─" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail
# ─────────────────────────────────────────────────────────────────────────────

def verify_source(rm: dict) -> dict:
    path = rm["source_ckpt"]
    print(f"\n  Loading {path} ...")
    state = torch.load(str(path), map_location="cpu")

    issues = []
    stored_f1  = state.get("best_refined_macro_f1", state.get("validation_macro_f1", -1))
    stored_acc = state.get("validation_accuracy", -1)
    stored_cls = state.get("num_classes", -1)

    if stored_cls != rm["prev_classes"]:
        issues.append(f"num_classes {stored_cls} != {rm['prev_classes']}")
    if abs(stored_f1 - rm["prev_f1"]) > TOL:
        issues.append(f"best_refined_macro_f1 {stored_f1:.4f} vs expected {rm['prev_f1']:.4f} (tol={TOL})")
    if abs(stored_acc - rm["prev_acc"]) > TOL * 100:
        issues.append(f"validation_accuracy {stored_acc:.4f} vs expected {rm['prev_acc']:.4f}")

    if issues:
        print(f"  GUARDRAIL FAILED for {rm['cfg_name']}:")
        for i in issues:
            print(f"      {i}")
        raise RuntimeError(f"Guardrail failed: {issues}")

    print(f"  GUARDRAIL PASSED  (F1={stored_f1:.4f}  Acc={stored_acc:.2f}%)")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Completion
# ─────────────────────────────────────────────────────────────────────────────

def completion_path(rm: dict) -> Path:
    return rm["final_dir"] / "final_refine_complete.json"


def is_complete(rm: dict) -> bool:
    return completion_path(rm).exists()


def write_completion(rm: dict, prev_f1: float, best_f1: float, best_acc: float,
                     best_ep: int, stop_reason: str, duration: float) -> dict:
    preferred = "FINAL-REFINED" if best_f1 > prev_f1 else "PREVIOUS-REFINED"
    marker = {
        "model"                     : rm["cfg_name"],
        "status"                    : "COMPLETED",
        "source_checkpoint"         : str(rm["source_ckpt"]),
        "source_prev_macro_f1"      : prev_f1,
        "best_final_refine_epoch"   : best_ep,
        "best_final_refine_macro_f1": best_f1,
        "best_final_refine_val_acc" : best_acc,
        "delta"                     : round(best_f1 - prev_f1, 6),
        "preferred_champion"        : preferred,
        "stop_reason"               : stop_reason,
        "duration_seconds"          : round(duration, 1),
        "best_final_checkpoint"     : str(rm["final_dir"] / "best_final_refined_model.pth"),
    }
    completion_path(rm).write_text(json.dumps(marker, indent=4))
    return marker


def load_last(rm: dict):
    p = rm["final_dir"] / "last_final_refine_checkpoint.pth"
    return torch.load(str(p), map_location="cpu") if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, num_classes, classes):
    ll = list(range(num_classes))
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=ll, zero_division=0)
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    _, _, w_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0)
    print("\n  Per-class metrics:")
    for i, cls in enumerate(classes):
        print(f"    {cls:45s}  P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}  sup={int(sup[i])}")
    print(f"\n  Macro  → P={m_prec:.4f}  R={m_rec:.4f}  F1={m_f1:.4f}")
    print(f"  Weighted F1 = {w_f1:.4f}")
    return m_f1, w_f1, prec, rec, f1, sup


# ─────────────────────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────────────────────

def final_refine(rm: dict, device: str = "cuda"):
    cfg = load_config(rm["cfg_name"])
    src_state = verify_source(rm)
    prev_f1 = src_state.get("best_refined_macro_f1",
                             src_state.get("validation_macro_f1", rm["prev_f1"]))

    for d in (rm["final_dir"], rm["log_dir"], rm["results_dir"]):
        d.mkdir(parents=True, exist_ok=True)

    num_classes = cfg["num_classes"]
    classes = [cfg["class_mapping"][str(i)] for i in range(num_classes)]
    cw = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)

    # Datasets
    train_ds = GenericDiseaseDataset(
        cfg["train_csv"], is_train=True,
        mean=cfg["normalization_mean"], std=cfg["normalization_std"])
    val_ds = GenericDiseaseDataset(
        cfg["train_csv"], is_train=False,
        mean=cfg["normalization_mean"], std=cfg["normalization_std"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = ResNet9Model1(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(src_state["model_state"])

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=FINAL_LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5)

    # Restore optimizer / scaler where compatible
    if "optimizer_state" in src_state:
        try:
            optimizer.load_state_dict(src_state["optimizer_state"])
            for pg in optimizer.param_groups:
                pg["lr"] = FINAL_LR
            print(f"  Optimizer state restored; LR forced to {FINAL_LR:.0e}")
        except Exception:
            print(f"  Optimizer state incompatible – fresh optimizer at {FINAL_LR:.0e}")
    if "scaler_state" in src_state:
        try:
            scaler.load_state_dict(src_state["scaler_state"])
            print("  Scaler state restored.")
        except Exception:
            print("  Scaler state incompatible – fresh scaler.")

    # Resume from last final-refine checkpoint if present
    start_epoch  = 1
    best_f1      = -1.0
    best_epoch   = -1
    es_counter   = 0
    last_val_acc = 0.0

    last_state = load_last(rm)
    if last_state is not None:
        model.load_state_dict(last_state["model_state"])
        optimizer.load_state_dict(last_state["optimizer_state"])
        scaler.load_state_dict(last_state["scaler_state"])
        scheduler.load_state_dict(last_state["scheduler_state"])
        start_epoch  = last_state["final_refine_epoch"] + 1
        best_f1      = last_state["best_final_macro_f1"]
        best_epoch   = last_state["best_final_epoch"]
        es_counter   = last_state["early_stopping_counter"]
        last_val_acc = last_state.get("validation_accuracy", 0.0)
        print(f"  Resumed from final-refine epoch {last_state['final_refine_epoch']} (best F1={best_f1:.4f})")

    # History CSV
    hist_path = rm["log_dir"] / "training_history.csv"
    if start_epoch == 1 or not hist_path.exists():
        with open(str(hist_path), "w", newline="") as f:
            csv.writer(f).writerow([
                "final_refine_epoch", "train_loss", "train_accuracy",
                "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1",
                "learning_rate", "epoch_duration_seconds", "best_so_far", "patience"
            ])

    total_start = time.time()
    stop_reason = "max_epochs"

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        ep_start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*60}")
        print(f"{rm['cfg_name'].upper()}  |  FINAL-REFINE EPOCH {epoch}/{MAX_EPOCHS}  |  LR={lr_now:.2e}")
        print(f"{'='*60}")

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        run_loss, correct, total_imgs = 0.0, 0, 0
        n_batches = len(train_loader)
        t0 = time.time()

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss   += loss.item()
            _, pred     = logits.max(1)
            total_imgs += labels.size(0)
            correct    += pred.eq(labels).sum().item()

            if (i + 1) % max(1, n_batches // 5) == 0 or (i + 1) == n_batches:
                pct     = (i + 1) / n_batches
                elapsed = time.time() - t0
                eta     = (elapsed / (i + 1)) * (n_batches - i - 1)
                print(f"  [{bar(pct)}] {pct*100:5.1f}%  "
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

        m_f1, w_f1, prec, rec, f1, sup = compute_metrics(
            all_labels, all_preds, num_classes, classes)

        scheduler.step(m_f1)
        lr_after = optimizer.param_groups[0]["lr"]
        ep_dur   = time.time() - ep_start

        is_best = m_f1 > best_f1
        if is_best:
            best_f1      = m_f1
            best_epoch   = epoch
            last_val_acc = val_acc
            es_counter   = 0
        else:
            es_counter += 1

        # ── Save checkpoints ─────────────────────────────────────────────────
        ckpt = {
            "source_checkpoint"         : str(rm["source_ckpt"]),
            "source_prev_macro_f1"      : prev_f1,
            "source_prev_val_acc"       : rm["prev_acc"],
            "final_refine_epoch"        : epoch,
            "model_state"               : {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state"           : optimizer.state_dict(),
            "scaler_state"              : scaler.state_dict(),
            "scheduler_state"           : scheduler.state_dict(),
            "current_lr"                : lr_after,
            "best_final_macro_f1"       : best_f1,
            "best_final_epoch"          : best_epoch,
            "early_stopping_counter"    : es_counter,
            "validation_accuracy"       : val_acc,
            "validation_macro_f1"       : m_f1,
            "validation_weighted_f1"    : w_f1,
            "class_mapping"             : cfg["class_mapping"],
            "label_to_index"            : cfg["label_to_index"],
            "normalization"             : {
                "mean": cfg["normalization_mean"],
                "std" : cfg["normalization_std"],
            },
            "class_weights"             : cfg["class_weights"],
            "architecture"              : cfg["architecture"],
            "num_classes"               : num_classes,
            "image_size"                : cfg["image_size"],
            "training_configuration"    : cfg,
        }
        atomic_save(ckpt, rm["final_dir"] / "last_final_refine_checkpoint.pth")
        if is_best:
            atomic_save(ckpt, rm["final_dir"] / "best_final_refined_model.pth")
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                str(rm["results_dir"] / "validation_confusion_matrix.csv"))

        # History
        with open(str(hist_path), "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, train_acc, val_loss, val_acc,
                m_f1, w_f1, lr_after, ep_dur, best_f1, es_counter
            ])

        # Epoch summary
        print(f"\n  ── Epoch {epoch} Summary ──────────────────────────────────")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")
        print(f"  Val Macro-F1: {m_f1:.4f}  Weighted-F1: {w_f1:.4f}")
        print(f"  LR: {lr_now:.2e} → {lr_after:.2e}")
        print(f"  Best Final-Refine F1: {best_f1:.4f}  (Epoch {best_epoch})")
        print(f"  Patience: {es_counter}/{PATIENCE}")
        print(f"  Epoch duration: {ep_dur:.1f}s")

        if es_counter >= PATIENCE:
            print(f"\n  Early stopping triggered after epoch {epoch}.")
            stop_reason = "early_stopping"
            break

    duration = time.time() - total_start
    marker = write_completion(rm, prev_f1, best_f1, last_val_acc, best_epoch, stop_reason, duration)

    print(f"\n{'='*60}")
    print(f"  {rm['cfg_name'].upper()} FINAL-REFINEMENT COMPLETE")
    print(f"  Previous refined F1 : {prev_f1:.4f}")
    print(f"  Final-refine best F1: {best_f1:.4f}  (Δ {best_f1 - prev_f1:+.4f})")
    print(f"  Preferred champion  : {marker['preferred_champion']}")
    print(f"{'='*60}\n")

    cleanup_cuda(model, optimizer, scaler)
    return marker


# ─────────────────────────────────────────────────────────────────────────────
# Master launcher
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    queue_file = WORKSPACE / "checkpoints" / "models5_7_final_refine_queue.json"

    def save_q(state):
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        queue_file.write_text(json.dumps(state, indent=4))

    state = {"completed": [], "failed": [], "current": None, "interrupted": None}
    if queue_file.exists():
        with open(str(queue_file)) as f:
            saved = json.load(f)
        state["completed"] = saved.get("completed", [])
        state["failed"]    = saved.get("failed", [])

    print("FINAL-REFINEMENT QUEUE")
    for rm in REFINE_MODELS:
        n = rm["cfg_name"]
        if n in state["completed"]:   sym = "[done]"
        elif n in state["failed"]:    sym = "[fail]"
        else:                         sym = "[    ]"
        print(f"  {sym} {n}")

    markers = []
    try:
        for rm in REFINE_MODELS:
            n = rm["cfg_name"]
            if n in state["completed"] or is_complete(rm):
                print(f"\n  Skipping {n} (complete).")
                if n not in state["completed"]:
                    state["completed"].append(n)
                    save_q(state)
                continue

            state["current"] = n
            save_q(state)

            try:
                marker = final_refine(rm, device=device)
                markers.append(marker)
                state["completed"].append(n)
                state["current"] = None
                save_q(state)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                print(f"\n  ERROR refining {n}: {e}")
                rm["log_dir"].mkdir(parents=True, exist_ok=True)
                (rm["log_dir"] / "error.log").write_text(tb)
                state["failed"].append(n)
                state["current"] = None
                save_q(state)
                cleanup_cuda()

    except KeyboardInterrupt:
        interrupted = state.get("current")
        state["interrupted"] = interrupted
        state["current"] = None
        save_q(state)
        print("\n============================================================")
        print("TRAINING INTERRUPTED BY USER")
        print(f"  Active model  : {interrupted}")
        print("  Resume: YES – re-run same command")
        print("============================================================")
        cleanup_cuda()
        sys.exit(0)

    # ── Champion comparison report ───────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL CHAMPION DECISION")
    print("="*60)

    lines = [
        "# Models 5 & 7 Final Refinement Report",
        "",
        "## Settings",
        f"- Starting LR : {FINAL_LR:.0e}",
        f"- Max epochs  : {MAX_EPOCHS}",
        f"- Early-stop patience: {PATIENCE}",
        f"- Scheduler   : ReduceLROnPlateau (max, factor=0.5, patience=1, min_lr=1e-5)",
        f"- AMP         : enabled",
        "",
        "## Source Checkpoints",
    ]
    for rm in REFINE_MODELS:
        lines += [
            f"### {rm['cfg_name']}",
            f"- Source : `{rm['source_ckpt']}`",
            f"- Source Val Acc  : {rm['prev_acc']}%",
            f"- Source Macro-F1 : {rm['prev_f1']}",
            "",
        ]

    lines += ["## Refinement Results", ""]
    for m in markers:
        delta = m["delta"]
        sign  = "+" if delta >= 0 else ""
        line_apple = (
            f"APPLE\n"
            f"  Previous refined F1 : {m['source_prev_macro_f1']:.4f}\n"
            f"  Final-refine best F1: {m['best_final_refine_macro_f1']:.4f}\n"
            f"  Delta               : {sign}{delta:.4f}\n"
            f"  Final champion      : {m['preferred_champion']}"
        ) if "apple" in m["model"] else None
        line_corn = (
            f"CORN\n"
            f"  Previous refined F1 : {m['source_prev_macro_f1']:.4f}\n"
            f"  Final-refine best F1: {m['best_final_refine_macro_f1']:.4f}\n"
            f"  Delta               : {sign}{delta:.4f}\n"
            f"  Final champion      : {m['preferred_champion']}"
        ) if "corn" in m["model"] else None

        for txt in [line_apple, line_corn]:
            if txt:
                print("\n" + txt)

        lines += [
            f"### {m['model']}",
            f"- Previous Macro-F1 : {m['source_prev_macro_f1']:.4f}",
            f"- Best Final F1     : {m['best_final_refine_macro_f1']:.4f}",
            f"- Delta             : {sign}{delta:.4f}",
            f"- Best Epoch        : {m['best_final_refine_epoch']}",
            f"- Stop Reason       : {m['stop_reason']}",
            f"- Duration (s)      : {m['duration_seconds']}",
            f"- **Preferred champion: {m['preferred_champion']}**",
            "",
        ]

    lines += [
        "## Test Set Status",
        "Test sets remain **LOCKED**. Zero test images were accessed during this stage.",
    ]

    report_path = WORKSPACE / "reports" / "MODELS_5_AND_7_FINAL_REFINEMENT_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
