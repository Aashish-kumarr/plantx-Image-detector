import os
import torch
import traceback
import pandas as pd

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.models.model3_potato_resnet9 import create_model3_potato_resnet9
from src.datasets.model3_potato_dataset import create_model3_potato_dataloaders
from src.training.model3_potato_trainer import Model3PotatoTrainer

def run_training():
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model3_potato_training_report.txt"
    report = []
    
    report.append("Model 3 (Potato) Training Report")
    report.append("================================\n")
    
    try:
        print("Preparing dataloaders...")
        train_loader, val_loader, _ = create_model3_potato_dataloaders(num_workers=0, batch_size=64)
        
        print("Initializing fresh Model 3 Potato architecture...")
        model = create_model3_potato_resnet9()
        
        print("Setting up trainer...")
        trainer = Model3PotatoTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            weight_decay=1e-4,
            patience_early_stopping=3,
            device='cuda'
        )
        
        print("\nStarting Model 3 Training (Max 10 Epochs)...")
        total_time, early_stopped = trainer.train(num_epochs=10)
        
        print("\nTraining completed successfully! Generating report...")
        
        # Read history for the report
        df = pd.read_csv(trainer.history_file)
        
        for idx, row in df.iterrows():
            epoch = int(row['epoch'])
            
            report.append(f"--- Epoch {epoch} ---")
            report.append(f"Time: {row['epoch_time_seconds']:.1f}s | LR: {row['learning_rate']:.1e}")
            report.append(f"Train Loss: {row['train_loss']:.4f} | Train Acc: {row['train_accuracy']:.2f}%")
            report.append(f"Val Loss: {row['val_loss']:.4f} | Val Acc: {row['val_accuracy']:.2f}%")
            report.append(f"Val Macro-F1: {row['val_macro_f1']:.4f}\n")
            
            report.append("  Per-Class Metrics:")
            report.append(f"    Healthy      -> Prec: {row['healthy_precision']:.4f} | Rec: {row['healthy_recall']:.4f} | F1: {row['healthy_f1']:.4f}")
            report.append(f"    Early Blight -> Prec: {row['early_blight_precision']:.4f} | Rec: {row['early_blight_recall']:.4f} | F1: {row['early_blight_f1']:.4f}")
            report.append(f"    Late Blight  -> Prec: {row['late_blight_precision']:.4f} | Rec: {row['late_blight_recall']:.4f} | F1: {row['late_blight_f1']:.4f}")
            report.append("\n")
            
        report.append("Summary:")
        
        best_row = df[df['epoch'] == trainer.best_epoch].iloc[0]
        report.append(f"  Best Epoch: {trainer.best_epoch}")
        report.append(f"  Best Validation Accuracy: {best_row['val_accuracy']:.2f}%")
        report.append(f"  Best Validation Macro-F1: {best_row['val_macro_f1']:.4f}")
        report.append(f"    Healthy      -> Prec: {best_row['healthy_precision']:.4f} | Rec: {best_row['healthy_recall']:.4f} | F1: {best_row['healthy_f1']:.4f}")
        report.append(f"    Early Blight -> Prec: {best_row['early_blight_precision']:.4f} | Rec: {best_row['early_blight_recall']:.4f} | F1: {best_row['early_blight_f1']:.4f}")
        report.append(f"    Late Blight  -> Prec: {best_row['late_blight_precision']:.4f} | Rec: {best_row['late_blight_recall']:.4f} | F1: {best_row['late_blight_f1']:.4f}")
        
        report.append(f"\n  Total training time: {total_time:.1f} seconds")
        report.append(f"  Early stopping occurred: {early_stopped}")
        report.append(f"  Final learning rate: {df.iloc[-1]['learning_rate']:.1e}")
        report.append("  Best checkpoint path: checkpoints/model3_potato/best_model3_potato.pth")
        
        report.append("\nSTEP 26 COMPLETE")
        
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        print(f"\nFinal Summary:")
        print(f"Total Training Time: {total_time:.1f} seconds")
        print(f"Best Val Macro-F1: {trainer.best_metric:.4f}")
        print(f"Report saved to: {report_out}")
        print("Checkpoints and confusion matrix saved to results/model3_potato and checkpoints/model3_potato.")
        
    except Exception as e:
        report.append(f"\nTraining FAILED with exception:\n{traceback.format_exc()}")
        report.append("\nSTEP 26 FAILED")
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    run_training()
