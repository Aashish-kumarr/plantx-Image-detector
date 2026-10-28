import os
import time
import torch
import traceback
import copy

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model1_resnet9 import create_model1_resnet9
from src.datasets.model1_dataset import create_model1_dataloaders
from src.training.model1_trainer import Model1Trainer

def run_benchmark():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model1_epoch1_benchmark.txt"
    report = []
    
    report.append("Model 1 Epoch 1 Benchmark Report")
    report.append("================================")
    
    history_file = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\results\model1\training_history.csv"
    if os.path.exists(history_file):
        os.remove(history_file)
    
    try:
        print("Preparing dataloaders...")
        train_loader, val_loader, _ = create_model1_dataloaders(num_workers=0, prefetch_factor=None, persistent_workers=False)
        
        print("Initializing model...")
        model = create_model1_resnet9()
        
        print("Setting up trainer...")
        trainer = Model1Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            weight_decay=1e-4,
            patience=5,
            device='cuda'
        )
        
        torch.cuda.reset_peak_memory_stats()
        
        print("\nStarting 1-Epoch Benchmark (Real Training)...")
        
        # Start timing
        start_time = time.time()
        
        # Run 1 real epoch
        train_loss, train_acc = trainer.train_epoch(epoch=1, num_epochs=1, dry_run=False)
        train_phase_dur = time.time() - start_time
        
        val_start = time.time()
        val_loss, val_acc, val_prec, val_rec, val_f1 = trainer.validate(dry_run=False)
        val_phase_dur = time.time() - val_start
        
        total_dur = time.time() - start_time
        
        # Manually step scheduler since we decomposed the trainer.train() loop to time it
        trainer.scheduler.step(val_loss)
        
        # Save checkpoints and history manually for the first epoch
        trainer.save_checkpoint(epoch=1, is_best=True, is_dry_run=False)
        
        import csv
        with open(trainer.history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                1, train_loss, train_acc, 
                val_loss, val_acc, val_prec, 
                val_rec, val_f1, trainer.optimizer.param_groups[0]['lr'], 
                total_dur, total_dur
            ])
            
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        mem_res = torch.cuda.memory_reserved() / (1024**2)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_res = torch.cuda.max_memory_reserved() / (1024**2)
        gpu_name = torch.cuda.get_device_name(0)
        
        total_train_imgs = len(train_loader.dataset)
        imgs_per_sec = total_train_imgs / train_phase_dur
        
        report.append("\nConfiguration Details:")
        report.append("  Model: ResNet-9 from scratch")
        report.append("  Loss: CrossEntropyLoss")
        report.append("  Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")
        report.append("  AMP: Enabled (autocast + GradScaler)")
        report.append("  Batch size: 64")
        report.append("  DataLoader: num_workers=0, pin_memory=False")
        
        report.append("\nPerformance Metrics:")
        report.append(f"  Train Loss: {train_loss:.4f}")
        report.append(f"  Train Accuracy: {train_acc:.2f}%")
        report.append(f"  Validation Loss: {val_loss:.4f}")
        report.append(f"  Validation Accuracy: {val_acc:.2f}%")
        report.append(f"  Validation Precision: {val_prec:.4f}")
        report.append(f"  Validation Recall: {val_rec:.4f}")
        report.append(f"  Validation Macro-F1: {val_f1:.4f}")
        
        report.append("\nTimings:")
        report.append(f"  Train phase duration: {train_phase_dur:.1f}s")
        report.append(f"  Validation phase duration: {val_phase_dur:.1f}s")
        report.append(f"  Complete epoch duration: {total_dur:.1f}s")
        report.append(f"  Training throughput: {imgs_per_sec:.1f} images/sec")
        
        report.append("\nHardware & Memory:")
        report.append(f"  GPU Name: {gpu_name}")
        report.append(f"  Current Allocated VRAM: {mem_alloc:.2f} MB")
        report.append(f"  Current Reserved VRAM: {mem_res:.2f} MB")
        report.append(f"  Peak Allocated VRAM: {peak_alloc:.2f} MB")
        report.append(f"  Peak Reserved VRAM: {peak_res:.2f} MB")
        
        report.append("\nCheckpoint Status:")
        report.append("  checkpoints/model1/last_model1.pth: Saved")
        report.append("  checkpoints/model1/best_model1.pth: Saved")
        
        report.append("\nSTEP 17 ONE-EPOCH BENCHMARK COMPLETE")
        
    except Exception as e:
        report.append(f"\nBenchmark FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nSTEP 17 FAILED")
        print(f"Failed with error: {e}")
        
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"\nBenchmark complete. Report saved to {report_out}")

if __name__ == "__main__":
    run_benchmark()
