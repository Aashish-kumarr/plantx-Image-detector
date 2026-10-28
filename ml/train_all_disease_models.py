import os
import sys
import json
import time
import traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import gc

sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from src.datasets.generic_disease_dataset import GenericDiseaseDataset
from src.training.train_disease_model import DiseaseModelTrainer

def load_queue_state(state_file, queue_list):
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {
        "queue": queue_list,
        "completed_models": [],
        "current_model": None,
        "failed_models": [],
        "interrupted_model": None,
        "start_timestamp": time.time(),
        "last_update_timestamp": time.time()
    }

def save_queue_state(state_file, state):
    state["last_update_timestamp"] = time.time()
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    workspace = Path(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
    config_file = workspace / "metadata" / "models5_11_training_config.json"
    state_file = workspace / "checkpoints" / "models5_11_training_queue_state.json"
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
        
    config_dict = {cfg['model_name']: cfg for cfg in configs}
    queue_list = [cfg['model_name'] for cfg in configs]
    
    state = load_queue_state(state_file, queue_list)
    
    print("============================================================")
    print("MODEL QUEUE")
    for m in queue_list:
        if m in state['completed_models']:
            print(f"[✓] {m}")
        elif m == state['current_model']:
            print(f"[▶] {m}")
        elif m in state['failed_models']:
            print(f"[x] {m}")
        else:
            print(f"[ ] {m}")
    print("============================================================\n")

    try:
        for model_name in queue_list:
            if model_name in state['completed_models']:
                continue
                
            state['current_model'] = model_name
            state['interrupted_model'] = None
            save_queue_state(state_file, state)
            
            cfg = config_dict[model_name]
            
            # Check for completion marker just in case
            marker = workspace / "checkpoints" / model_name / "training_complete.json"
            if marker.exists():
                print(f"Skipping {model_name}, already marked complete.")
                state['completed_models'].append(model_name)
                state['current_model'] = None
                save_queue_state(state_file, state)
                continue
            
            print(f"Loading datasets for {model_name}...")
            train_dataset = GenericDiseaseDataset(cfg['train_csv'], True, cfg['normalization_mean'], cfg['normalization_std'])
            val_dataset = GenericDiseaseDataset(cfg['train_csv'], False, cfg['normalization_mean'], cfg['normalization_std'])
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
            
            trainer = DiseaseModelTrainer(cfg, train_loader, val_loader)
            
            try:
                success = trainer.train(max_epochs=10)
                if success:
                    state['completed_models'].append(model_name)
                    state['current_model'] = None
                    save_queue_state(state_file, state)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                traceback.print_exc()
                with open(workspace / "logs" / model_name / "training.log", "a") as lf:
                    lf.write(traceback.format_exc())
                state['failed_models'].append(model_name)
                state['current_model'] = None
                save_queue_state(state_file, state)
            
            # Cleanup
            del trainer
            del train_loader
            del val_loader
            del train_dataset
            del val_dataset
            cleanup_cuda()
            
        print("\nAll models processed.")
        
        # Summary report
        summary = ["# Models 5-11 Training Summary\n"]
        for model_name in queue_list:
            marker = workspace / "checkpoints" / model_name / "training_complete.json"
            if marker.exists():
                with open(marker, 'r') as mf:
                    md = json.load(mf)
                summary.append(f"## {model_name}")
                summary.append(f"- Status: {md.get('status')}")
                summary.append(f"- Epochs Completed: {md.get('epochs_completed')}")
                summary.append(f"- Best Epoch: {md.get('best_epoch')}")
                summary.append(f"- Best Macro-F1: {md.get('best_macro_f1')}")
                summary.append(f"- Stop Reason: {md.get('stop_reason')}\n")
        
        with open(workspace / "reports" / "MODELS_5_TO_11_TRAINING_SUMMARY.md", "w") as f:
            f.write("\n".join(summary))

    except KeyboardInterrupt:
        state['interrupted_model'] = state['current_model']
        save_queue_state(state_file, state)
        print("\n============================================================")
        print("TRAINING INTERRUPTED BY USER")
        print("============================================================")
        print(f"Active model: {state['interrupted_model']}")
        print("Resume available: YES")
        cleanup_cuda()
        sys.exit(0)

if __name__ == "__main__":
    main()
