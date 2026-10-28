import os
import torch
import traceback
import pandas as pd

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model2_14class_resnet9 import create_model2_14class_resnet9
from src.datasets.model2_14class_dataset import create_model2_14class_dataloaders
from src.training.model2_14class_trainer import Model2_14ClassTrainer

def run_training():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model2_14class_training_2epoch_report.txt"
    report = []
    
    report.append("Model 2 (14-Class) Training Report (2 Epochs)")
    report.append("=============================================\n")
    
    try:
        print("Preparing dataloaders...")
        train_loader, val_loader, _ = create_model2_14class_dataloaders(num_workers=0, batch_size=64)
        
        print("Initializing fresh Model 2 14-class architecture...")
        model = create_model2_14class_resnet9()
        
        print("Setting up trainer...")
        trainer = Model2_14ClassTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            weight_decay=1e-4,
            device='cuda'
        )
        
        print("\nStarting Model 2 Training (2 Epochs)...")
        total_time = trainer.train(num_epochs=2)
        
        print("\nTraining completed successfully! Generating report...")
        
        # Read history for the report
        df = pd.read_csv(trainer.history_file)
        df_per_class = pd.read_csv(trainer.per_class_history_file)
        
        for idx, row in df.iterrows():
            epoch = int(row['epoch'])
            pc_row = df_per_class[df_per_class['epoch'] == epoch].iloc[0]
            
            report.append(f"--- Epoch {epoch} ---")
            report.append(f"Time: {row['epoch_time_seconds']:.1f}s | LR: {row['learning_rate']:.1e}")
            report.append(f"Train Loss: {row['train_loss']:.4f} | Train Acc: {row['train_accuracy']:.2f}%")
            report.append(f"Val Loss: {row['val_loss']:.4f} | Val Acc: {row['val_accuracy']:.2f}%")
            report.append(f"Val Macro-F1: {row['val_macro_f1']:.4f}\n")
            
            report.append("  Per-Class Metrics:")
            for c in trainer.classes:
                prec = pc_row[f"{c}_precision"]
                rec = pc_row[f"{c}_recall"]
                f1 = pc_row[f"{c}_f1"]
                report.append(f"    {c.capitalize():<12} -> Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            report.append("\n")
            
        report.append("Checkpoints:")
        report.append("  Saved checkpoints/model2_14class/last_model2_14class.pth")
        report.append("  Saved checkpoints/model2_14class/best_model2_14class.pth (based on Val Macro-F1)")
        report.append("  Saved results/model2_14class/model2_14class_confusion_matrix.png")
        
        report.append(f"\nBest Epoch: {trainer.best_epoch}")
        report.append(f"Best Validation Macro-F1: {trainer.best_metric:.4f}")
        report.append(f"Total Training Time: {total_time:.1f} seconds")
        report.append("Best checkpoint: checkpoints/model2_14class/best_model2_14class.pth")
        
        report.append("\nSTEP 23 COMPLETE")
        
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        print(f"\nFinal Summary:")
        print(f"Total Training Time: {total_time:.1f} seconds")
        print(f"Best Val Macro-F1: {trainer.best_metric:.4f}")
        print(f"Report saved to: {report_out}")
        print("Checkpoints and confusion matrix saved to results/model2_14class and checkpoints/model2_14class.")
        
    except Exception as e:
        report.append(f"\nTraining FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nSTEP 23 FAILED")
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    run_training()
