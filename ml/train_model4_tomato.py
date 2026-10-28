import os
import torch
import traceback
import pandas as pd

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model4_tomato_resnet9 import create_model4_tomato_resnet9
from src.datasets.model4_tomato_dataset import create_model4_tomato_dataloaders
from src.training.model4_tomato_trainer import Model4TomatoTrainer

def run_training():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model4_tomato_training_report.txt"
    report = []
    
    report.append("Model 4 (Tomato) Training Report")
    report.append("================================\n")
    
    try:
        print("Preparing dataloaders...")
        train_loader, val_loader, _ = create_model4_tomato_dataloaders(num_workers=0, batch_size=64)
        
        print("Initializing fresh Model 4 Tomato architecture...")
        model = create_model4_tomato_resnet9()
        
        print("Setting up trainer...")
        trainer = Model4TomatoTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            weight_decay=1e-4,
            patience_early_stopping=2,
            device='cuda'
        )
        
        print("\nStarting Model 4 Training (Max 5 Epochs)...")
        total_time, early_stopped = trainer.train(num_epochs=5)
        
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
                report.append(f"    {c:<30} -> Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            report.append("\n")
            
        report.append("Summary:")
        
        best_row = df[df['epoch'] == trainer.best_epoch].iloc[0]
        best_pc_row = df_per_class[df_per_class['epoch'] == trainer.best_epoch].iloc[0]
        
        report.append(f"  Best Epoch: {trainer.best_epoch}")
        report.append(f"  Best Validation Accuracy: {best_row['val_accuracy']:.2f}%")
        report.append(f"  Best Validation Macro-F1: {best_row['val_macro_f1']:.4f}")
        
        report.append("\n  Best Epoch Per-Class P/R/F1:")
        for c in trainer.classes:
            prec = best_pc_row[f"{c}_precision"]
            rec = best_pc_row[f"{c}_recall"]
            f1 = best_pc_row[f"{c}_f1"]
            report.append(f"    {c:<30} -> Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        report.append(f"\n  Total training time: {total_time:.1f} seconds")
        report.append(f"  Early stopping occurred: {early_stopped}")
        report.append(f"  Final learning rate: {df.iloc[-1]['learning_rate']:.1e}")
        report.append("  Best checkpoint path: checkpoints/model4_tomato/best_model4_tomato.pth")
        
        report.append("\nSTEP 29 COMPLETE")
        
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        print(f"\nFinal Summary:")
        print(f"Total Training Time: {total_time:.1f} seconds")
        print(f"Best Val Macro-F1: {trainer.best_metric:.4f}")
        print(f"Report saved to: {report_out}")
        print("Checkpoints and confusion matrix saved to results/model4_tomato and checkpoints/model4_tomato.")
        
    except Exception as e:
        report.append(f"\nTraining FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nSTEP 29 FAILED")
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    run_training()
