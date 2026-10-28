import os
import torch
import traceback
import pandas as pd

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model2_resnet9 import create_model2_resnet9
from src.datasets.model2_dataset import create_model2_dataloaders
from src.training.model2_trainer import Model2Trainer

def run_training():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model2_training_2epoch_report.txt"
    report = []
    
    report.append("Model 2 Training Report (2 Epochs)")
    report.append("==================================\n")
    
    try:
        print("Preparing dataloaders...")
        train_loader, val_loader, _ = create_model2_dataloaders(num_workers=0, batch_size=64)
        
        print("Initializing fresh Model 2 architecture...")
        model = create_model2_resnet9()
        
        print("Setting up trainer...")
        trainer = Model2Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            weight_decay=1e-4,
            patience=5,
            device='cuda'
        )
        
        print("\nStarting Model 2 Training (2 Epochs)...")
        total_time = trainer.train(num_epochs=2)
        
        print("\nTraining completed successfully! Generating report...")
        
        # Read history for the report
        df = pd.read_csv(trainer.history_file)
        
        report.append(f"Total Training Time: {total_time:.1f} seconds")
        report.append(f"Best Validation Macro-F1: {trainer.best_metric:.4f}")
        
        report.append("\nEpoch Metrics:\n")
        
        for idx, row in df.iterrows():
            report.append(f"--- Epoch {int(row['epoch'])} ---")
            report.append(f"Time: {row['epoch_time']:.1f}s | LR: {row['learning_rate']:.1e}")
            report.append(f"Train Loss: {row['train_loss']:.4f} | Train Acc: {row['train_accuracy']:.2f}%")
            report.append(f"Val Loss: {row['val_loss']:.4f} | Val Acc: {row['val_accuracy']:.2f}%")
            report.append(f"Val Macro-F1: {row['val_macro_f1']:.4f}")
            report.append(f"  Potato -> Prec: {row['potato_precision']:.4f} | Rec: {row['potato_recall']:.4f} | F1: {row['potato_f1']:.4f}")
            report.append(f"  Tomato -> Prec: {row['tomato_precision']:.4f} | Rec: {row['tomato_recall']:.4f} | F1: {row['tomato_f1']:.4f}")
            report.append(f"  Other  -> Prec: {row['other_precision']:.4f} | Rec: {row['other_recall']:.4f} | F1: {row['other_f1']:.4f}\n")
            
        report.append("Checkpoints:")
        report.append("  Saved checkpoints/model2/last_model2.pth")
        report.append("  Saved checkpoints/model2/best_model2.pth (based on Val Macro-F1)")
        report.append("  Saved results/model2/model2_confusion_matrix.png")
        
        report.append("\nSTEP 21 COMPLETE")
        
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        print(f"\nFinal Summary:")
        print(f"Total time: {total_time:.1f}s")
        print(f"Best Val Macro-F1: {trainer.best_metric:.4f}")
        print(f"Report saved to: {report_out}")
        print("Checkpoints and confusion matrix saved to results/model2 and checkpoints/model2.")
        
    except Exception as e:
        report.append(f"\nTraining FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nSTEP 21 FAILED")
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    run_training()
