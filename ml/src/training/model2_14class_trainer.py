import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class Model2_14ClassTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3, weight_decay=1e-4, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.classes = [
            'apple', 'blueberry', 'cherry', 'corn', 'grape', 'orange', 'peach', 
            'pepper', 'potato', 'raspberry', 'soybean', 'squash', 'strawberry', 'tomato'
        ]
        
        # Square-Root Inverse-Frequency Weights
        class_weights = torch.tensor([
            0.8399, 1.2224, 1.0980, 0.7703, 0.7421, 0.6420, 0.9224, 
            0.9540, 1.0271, 2.4479, 0.6691, 1.1054, 1.2047, 0.3546
        ], dtype=torch.float32, device=self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.checkpoint_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model2_14class"
        self.results_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\results\model2_14class"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.history_file = os.path.join(self.results_dir, "training_history.csv")
        self.per_class_history_file = os.path.join(self.results_dir, "training_history_per_class.csv")
        
        # Initialize history file if it doesn't exist
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "train_accuracy", 
                    "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1", 
                    "learning_rate", "epoch_time_seconds"
                ])
                
        if not os.path.exists(self.per_class_history_file):
            with open(self.per_class_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["epoch"]
                for c in self.classes:
                    header.extend([f"{c}_precision", f"{c}_recall", f"{c}_f1"])
                writer.writerow(header)
                
        self.best_metric = -1.0
        self.best_epoch = -1
        
    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        total_batches = len(self.train_loader)
        
        pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}/{num_epochs}", leave=True, dynamic_ncols=True,
                    bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA={remaining} | {postfix}")
        pbar.set_postfix(loss="0.000", acc="0.0%", lr="1.0e-3")
        
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                pbar.set_postfix(
                    loss=f"{running_loss / (i + 1):.4f}", 
                    acc=f"{100. * correct / total:.1f}%", 
                    lr=f"{current_lr:.1e}"
                )
            
            pbar.update(1)
            
        pbar.close()
        
        epoch_loss = running_loss / total_batches
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
        
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        total_batches = len(self.val_loader)
        
        with torch.inference_mode():
            pbar = tqdm(total=total_batches, desc="Validation", leave=False, dynamic_ncols=True, 
                        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}<{remaining}")
            
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                running_loss += loss.item()
                _, predicted = logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.update(1)
                
            pbar.close()
                
        val_loss = running_loss / total_batches
        val_acc = 100. * correct / total
        
        labels_list = list(range(14))
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=labels_list, average=None, zero_division=0)
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        
        return val_loss, val_acc, macro_prec, macro_rec, macro_f1, prec, rec, f1, all_labels, all_preds
        
    def save_checkpoint(self, epoch, is_best):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'best_macro_f1': self.best_metric,
            'label_mapping': {k: i for i, k in enumerate(self.classes)},
            'normalization': {
                'mean': [0.1257, 0.1572, 0.0856],
                'std': [0.1620, 0.1985, 0.1149]
            },
            'class_weights': [
                0.8399, 1.2224, 1.0980, 0.7703, 0.7421, 0.6420, 0.9224, 
                0.9540, 1.0271, 2.4479, 0.6691, 1.1054, 1.2047, 0.3546
            ],
            'architecture': "ResNet-9 (num_classes=14)"
        }
        
        last_path = os.path.join(self.checkpoint_dir, "last_model2_14class.pth")
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model2_14class.pth")
            torch.save(state, best_path)
            
    def generate_confusion_matrix(self, all_labels, all_preds):
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(14)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.xticks(rotation=45, ha='right')
        plt.title('Model 2 (14-Class) Validation Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "model2_14class_confusion_matrix.png"))
        plt.close()
        
    def train(self, num_epochs=2):
        start_time = time.time()
        
        # Clear history if it exists for a fresh run
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        if os.path.exists(self.per_class_history_file):
            os.remove(self.per_class_history_file)
            
        with open(self.history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_accuracy", 
                "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1", 
                "learning_rate", "epoch_time_seconds"
            ])
            
        with open(self.per_class_history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["epoch"]
            for c in self.classes:
                header.extend([f"{c}_precision", f"{c}_recall", f"{c}_f1"])
            writer.writerow(header)
                
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch, num_epochs)
            val_loss, val_acc, macro_prec, macro_rec, macro_f1, prec, rec, f1, all_labels, all_preds = self.validate()
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            is_best = macro_f1 > self.best_metric
            if is_best:
                self.best_metric = macro_f1
                self.best_epoch = epoch
                
            self.save_checkpoint(epoch, is_best)
            
            with open(self.history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, train_loss, train_acc, 
                    val_loss, val_acc, macro_prec, macro_rec, macro_f1, 
                    current_lr, epoch_time
                ])
                
            with open(self.per_class_history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [epoch]
                for i in range(14):
                    row.extend([prec[i], rec[i], f1[i]])
                writer.writerow(row)
                
            if epoch == num_epochs:
                self.generate_confusion_matrix(all_labels, all_preds)
                
        total_time = time.time() - start_time
        
        return total_time
