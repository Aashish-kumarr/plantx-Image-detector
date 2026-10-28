import os
import sys
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory

sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.datasets.generic_disease_dataset import GenericDiseaseDataset
from src.training.train_disease_model import DiseaseModelTrainer

def main():
    workspace = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
    config_file = workspace / "metadata" / "models5_11_training_config.json"
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
        
    cfg = configs[0]  # Just use Apple (Model 5) for smoke test
    
    print("Running Smoke Test...")
    
    train_dataset = GenericDiseaseDataset(cfg['train_csv'], True, cfg['normalization_mean'], cfg['normalization_std'])
    val_dataset = GenericDiseaseDataset(cfg['train_csv'], False, cfg['normalization_mean'], cfg['normalization_std'])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    
    with TemporaryDirectory() as temp_dir:
        # Override checkpoint dir to temp for smoke test
        trainer = DiseaseModelTrainer(cfg, train_loader, val_loader, run_smoke_test=True)
        trainer.checkpoint_dir = Path(temp_dir)
        trainer.results_dir = Path(temp_dir)
        trainer.log_dir = Path(temp_dir)
        
        # Test 1 batch train
        train_loss, train_acc, _ = trainer.train_epoch(1, 1, smoke_test=True)
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
            print("NaN/Inf in train loss!")
            
        # Test 1 batch val
        val_loss, val_acc, m_prec, m_rec, m_f1, w_f1, prec, rec, f1, supp, _, _ = trainer.validate(smoke_test=True)
        if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
            print("NaN/Inf in val loss!")
            
        # Test Save Checkpoint
        trainer.best_metric = 0.99
        trainer._save_checkpoint(1, True, val_acc, m_f1, w_f1)
        
        # Test Load Checkpoint
        new_trainer = DiseaseModelTrainer(cfg, train_loader, val_loader, run_smoke_test=True)
        new_trainer.checkpoint_dir = Path(temp_dir)
        new_trainer._load_last_checkpoint()
        
        if new_trainer.start_epoch != 2:
            print("Resume state failed: start_epoch")
        if new_trainer.best_metric != 0.99:
            print("Resume state failed: best_metric")
            
    # Cleanup CUDA
    del trainer
    del new_trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Smoke test complete.")

if __name__ == "__main__":
    main()
