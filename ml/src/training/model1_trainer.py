import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
from sklearn.metrics import precision_recall_fscore_support

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Model1Trainer:
    def __init__(self, model, train_loader, val_loader, 
                 lr=1e-3, weight_decay=1e-4, 
                 patience=5, device='cuda',
                 checkpoint_dir=r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model1",
                 results_dir=r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\results\model1"):
        
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.early_stopping_patience = patience
        self.best_val_f1 = -1.0
        self.epochs_without_improvement = 0
        self.start_epoch = 1
        
        self.history_file = os.path.join(self.results_dir, 'training_history.csv')
        self._init_history()

    def _init_history(self):
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_accuracy', 
                    'val_loss', 'val_accuracy', 'val_precision', 
                    'val_recall', 'val_f1', 'learning_rate', 
                    'epoch_time_seconds', 'total_elapsed_seconds'
                ])

    def save_checkpoint(self, epoch, is_best=False, is_dry_run=False):
        if is_dry_run:
            return
            
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'label_mapping': {'non_leaf': 0, 'leaf': 1}
        }
        
        last_path = os.path.join(self.checkpoint_dir, 'last_model1.pth')
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model1.pth')
            torch.save(state, best_path)

    def load_checkpoint(self, path):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
            self.scaler.load_state_dict(state['scaler_state_dict'])
            self.start_epoch = state['epoch'] + 1
            self.best_val_f1 = state['best_val_f1']
            print(f"Loaded checkpoint from {path} (Epoch {state['epoch']}, Best F1: {state['best_val_f1']:.4f})")
            return True
        return False

    def train_epoch(self, epoch, num_epochs, dry_run=False, dry_run_batches=5):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        total_batches = len(self.train_loader)
        if dry_run:
            total_batches = min(dry_run_batches, total_batches)
            
        pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}/{num_epochs}", leave=True, dynamic_ncols=True,
                    bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | ETA={remaining} | {postfix}")
        pbar.set_postfix(loss="0.000", acc="0.0%", lr="1.0e-3")
        
        for i, (images, labels) in enumerate(self.train_loader):
            if dry_run and i >= dry_run_batches:
                break
                
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
            
            cur_loss = running_loss / (i + 1)
            cur_acc = 100. * correct / total
            cur_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'loss': f"{cur_loss:.4f}",
                'acc': f"{cur_acc:.1f}%",
                'lr': f"{cur_lr:.1e}"
            })
            pbar.update(1)
            
        pbar.close()
        return running_loss / total_batches, 100. * correct / total

    def validate(self, dry_run=False, dry_run_batches=5):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        total_batches = len(self.val_loader)
        if dry_run:
            total_batches = min(dry_run_batches, total_batches)
            
        with torch.inference_mode():
            pbar = tqdm(total=total_batches, desc="Validation", leave=False, dynamic_ncols=True, 
                        bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}<{remaining}")
            
            for i, (images, labels) in enumerate(self.val_loader):
                if dry_run and i >= dry_run_batches:
                    break
                    
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
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        return val_loss, val_acc, precision, recall, f1

    def train(self, num_epochs, dry_run=False, dry_run_batches=5):
        start_time_total = time.time()
        print(f"{'DRY RUN — NOT A REAL TRAINING EPOCH' if dry_run else 'Starting Training'}")
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            start_time_epoch = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch, num_epochs, dry_run, dry_run_batches)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(dry_run, dry_run_batches)
            
            if not dry_run:
                self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time_epoch
            total_time = time.time() - start_time_total
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.1e}")
            print(f"  Epoch Duration: {epoch_time:.1f}s | Total Elapsed: {total_time:.1f}s")
            
            is_best = val_f1 > self.best_val_f1
            if is_best:
                print(f"  *** New Best Validation F1: {val_f1:.4f} (improved from {self.best_val_f1:.4f}) ***")
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
            self.save_checkpoint(epoch, is_best, is_dry_run=dry_run)
            
            if not dry_run:
                with open(self.history_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, train_loss, train_acc, 
                        val_loss, val_acc, val_prec, 
                        val_rec, val_f1, self.optimizer.param_groups[0]['lr'], 
                        epoch_time, total_time
                    ])
                    
            if not dry_run and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break
                
            if dry_run:
                return train_loss, val_f1 # Return some metrics to assert in the script
