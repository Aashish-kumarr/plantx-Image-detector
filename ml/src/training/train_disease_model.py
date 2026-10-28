import os
import time
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

class DiseaseModelTrainer:
    def __init__(self, config, train_loader, val_loader, device='cuda', run_smoke_test=False):
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.model_name = config['model_name']
        self.num_classes = config['num_classes']
        self.classes = [config['class_mapping'][str(i)] for i in range(self.num_classes)]
        
        self.checkpoint_dir = Path(rf"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\{self.model_name}")
        self.log_dir = Path(rf"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\logs\{self.model_name}")
        self.results_dir = Path(rf"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\results\{self.model_name}")
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.log_dir / "training_history.csv"
        
        # Build Model
        from src.models.model1_resnet9 import ResNet9Model1
        self.model = ResNet9Model1(in_channels=3, num_classes=self.num_classes).to(self.device)
        
        # Loss and Weights
        class_weights = torch.tensor(config['class_weights'], dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
        self.scaler = torch.amp.GradScaler('cuda')
        
        # State
        self.start_epoch = 1
        self.best_metric = -1.0
        self.best_epoch = -1
        self.early_stopping_counter = 0
        self.early_stopped = False
        
        # Resume if last_checkpoint exists and not smoke testing
        if not run_smoke_test:
            self._load_last_checkpoint()
            
    def _load_last_checkpoint(self):
        last_path = self.checkpoint_dir / "last_checkpoint.pth"
        if last_path.exists():
            print(f"Resuming from {last_path}")
            state = torch.load(last_path)
            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            self.scaler.load_state_dict(state['scaler_state'])
            if 'scheduler_state' in state:
                self.scheduler.load_state_dict(state['scheduler_state'])
            self.start_epoch = state['epoch'] + 1
            self.best_metric = state['best_macro_f1']
            self.best_epoch = state.get('best_epoch', state['epoch'])
            self.early_stopping_counter = state.get('early_stopping_counter', 0)
        else:
            if not self.history_file.exists():
                with open(self.history_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = [
                        "epoch", "train_loss", "train_accuracy", 
                        "val_loss", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1", "val_weighted_f1",
                        "learning_rate", "epoch_duration_seconds", "best_so_far"
                    ]
                    writer.writerow(header)
                    
    def _save_checkpoint(self, epoch, is_best, val_acc, val_f1, val_weighted_f1):
        state = {
            'epoch': epoch,
            'model_number': self.config['model_number'],
            'model_name': self.model_name,
            'crop': self.config['crop'],
            'model_state': self.model_state_dict_safe(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'best_macro_f1': self.best_metric,
            'best_epoch': self.best_epoch,
            'early_stopping_counter': self.early_stopping_counter,
            'validation_accuracy': val_acc,
            'validation_macro_f1': val_f1,
            'validation_weighted_f1': val_weighted_f1,
            'class_mapping': self.config['class_mapping'],
            'label_to_index': self.config['label_to_index'],
            'normalization': {
                'mean': self.config['normalization_mean'],
                'std': self.config['normalization_std']
            },
            'class_weights': self.config['class_weights'],
            'architecture': self.config['architecture'],
            'num_classes': self.num_classes,
            'image_size': self.config['image_size'],
            'training_configuration': self.config
        }
        
        last_path = self.checkpoint_dir / "last_checkpoint.pth"
        self._atomic_save(state, last_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            self._atomic_save(state, best_path)
            
    def model_state_dict_safe(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def _atomic_save(self, state, path):
        dir_name = os.path.dirname(path)
        with NamedTemporaryFile(delete=False, dir=dir_name, suffix=".pth") as temp_file:
            torch.save(state, temp_file.name)
            temp_name = temp_file.name
        shutil.move(temp_name, path)

    def train_epoch(self, epoch, num_epochs, smoke_test=False):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        total_batches = len(self.train_loader)
        start_time = time.time()
        
        print(f"\n============================================================")
        print(f"{self.model_name.upper()}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"============================================================")
        
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            
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
            
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                pct = (i + 1) / total_batches
                bar_len = 20
                filled = int(pct * bar_len)
                bar = '█' * filled + '-' * (bar_len - filled)
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (total_batches - (i + 1))
                print(f"Training:\n[{bar}] {pct*100:.1f}%")
                print(f"Batch: {i+1}/{total_batches}")
                print(f"Loss: {running_loss/(i+1):.4f} | Accuracy: {100.*correct/total:.2f}%")
                print(f"Elapsed: {int(elapsed)}s | Epoch ETA: {int(eta)}s\n")
                
            if smoke_test: break
            
        return running_loss / max(1, (i+1)), 100. * correct / total, time.time() - start_time

    def validate(self, smoke_test=False):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        
        total_batches = len(self.val_loader)
        
        with torch.inference_mode():
            for i, (images, labels) in enumerate(self.val_loader):
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
                
                if smoke_test: break
                
        val_loss = running_loss / max(1, (i+1))
        val_acc = 100. * correct / total
        
        prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(self.num_classes)), zero_division=0)
        m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        
        return val_loss, val_acc, m_prec, m_rec, m_f1, w_f1, prec, rec, f1, support, all_labels, all_preds

    def save_confusion_matrix(self, labels, preds, filename="validation_confusion_matrix.csv"):
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        df = pd.DataFrame(cm, index=self.classes, columns=self.classes)
        df.to_csv(self.results_dir / filename)
        
    def train(self, max_epochs=10):
        start_time = time.time()
        
        if self.start_epoch > max_epochs:
            print(f"Model already reached max epochs ({max_epochs}).")
            self._write_completion("max_epochs", time.time() - start_time)
            return True
            
        for epoch in range(self.start_epoch, max_epochs + 1):
            train_loss, train_acc, epoch_dur = self.train_epoch(epoch, max_epochs)
            val_loss, val_acc, m_prec, m_rec, m_f1, w_f1, prec, rec, f1, support, all_lbl, all_prd = self.validate()
            
            self.scheduler.step(m_f1)
            lr = self.optimizer.param_groups[0]['lr']
            
            is_best = m_f1 > self.best_metric
            if is_best:
                self.best_metric = m_f1
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                self.save_confusion_matrix(all_lbl, all_prd)
            else:
                self.early_stopping_counter += 1
                
            self._save_checkpoint(epoch, is_best, val_acc, m_f1, w_f1)
            
            with open(self.history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, train_loss, train_acc, val_loss, val_acc, m_prec, m_rec, m_f1, w_f1, lr, epoch_dur, self.best_metric
                ])
                
            print("After validation:\n")
            print(f"Epoch {epoch} Results")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.2f}%\n")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.2f}%")
            print(f"Val Macro-F1: {m_f1:.4f}")
            print(f"Val Weighted-F1: {w_f1:.4f}\n")
            print(f"Best Macro-F1: {self.best_metric:.4f}")
            print(f"Best Epoch: {self.best_epoch}\n")
            
            if self.early_stopping_counter >= 3:
                print(f"Early stopping triggered after {epoch} epochs.")
                self.early_stopped = True
                self._write_completion("early_stopping", time.time() - start_time)
                return True
                
        self._write_completion("max_epochs", time.time() - start_time)
        return True
        
    def _write_completion(self, reason, duration):
        marker_path = self.checkpoint_dir / "training_complete.json"
        with open(marker_path, 'w') as f:
            json.dump({
                "model": self.model_name,
                "crop": self.config['crop'],
                "status": "COMPLETED",
                "epochs_completed": self.start_epoch - 1 if hasattr(self, 'start_epoch') else 10,
                "best_epoch": self.best_epoch,
                "best_macro_f1": self.best_metric,
                "best_validation_accuracy": getattr(self, 'last_val_acc', 0.0), # Simplification
                "stop_reason": reason,
                "duration_seconds": duration,
                "best_checkpoint_path": str(self.checkpoint_dir / "best_model.pth")
            }, f, indent=4)
