import time
import torch
import cv2
import csv
import traceback
import os
import psutil
from torch.utils.data import DataLoader

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_dataset import Model1Dataset
from src.datasets.model1_transforms import get_model1_train_transform

class WorkerInit:
    def __init__(self, cv_threads):
        self.cv_threads = cv_threads
    
    def __call__(self, worker_id):
        import cv2
        if self.cv_threads > 0:
            cv2.setNumThreads(self.cv_threads)

def run_benchmarks():
    csv_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_split.csv"
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model1_dataloader_optimization.txt"
    batch_size = 64
    batches_to_measure = 50
    warmup_batches = 2
    
    train_transform = get_model1_train_transform()
    train_dataset = Model1Dataset(csv_path, split_name='train', transform=train_transform)
    
    configs = [
        # num_workers, prefetch, persist, cv_threads
        (4, 2, False, 0),
        (4, 2, False, 1),
        (6, 2, True, 1),
        (8, 2, True, 0),
        (8, 2, True, 1),
        (10, 2, True, 1),
        (12, 2, True, 1),
        (8, 4, True, 1),
        (10, 4, True, 1),
        (12, 4, True, 1),
    ]
    
    results = []
    
    print("Starting DataLoader Optimization Benchmark...")
    print(f"Total system RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    for w, p, persist, cv in configs:
        print(f"\nTesting: workers={w}, prefetch={p}, persist={persist}, cv_threads={'auto' if cv==0 else cv}")
        
        try:
            init_fn = WorkerInit(cv) if cv > 0 else None
            
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=w,
                prefetch_factor=p,
                persistent_workers=persist,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=init_fn
            )
            
            iterator = iter(loader)
            
            # Warm up
            for _ in range(warmup_batches):
                next(iterator)
                
            # Measure
            start_time = time.time()
            valid_tensors = True
            
            for _ in range(batches_to_measure):
                images, labels = next(iterator)
                if torch.isnan(images).any() or torch.isinf(images).any():
                    valid_tensors = False
                    
            total_time = time.time() - start_time
            
            avg_time = total_time / batches_to_measure
            imgs_sec = batch_size / avg_time
            
            print(f"  Stable: YES | {imgs_sec:.1f} imgs/s | {avg_time:.3f} s/batch")
            
            results.append({
                "w": w,
                "p": p,
                "persist": persist,
                "cv": cv,
                "time": avg_time,
                "imgs_sec": imgs_sec,
                "stable": "Yes" if valid_tensors else "Corrupted"
            })
            
            # Delete iterator and loader to free resources immediately
            del iterator
            del loader
            
        except Exception as e:
            err_msg = str(e).split('\n')[0]
            print(f"  Stable: NO ({err_msg})")
            results.append({
                "w": w,
                "p": p,
                "persist": persist,
                "cv": cv,
                "time": float('inf'),
                "imgs_sec": 0,
                "stable": f"No ({err_msg})"
            })
            
    # Generate Report
    report = []
    report.append("Model 1 DataLoader Optimization Benchmark")
    report.append("=========================================")
    report.append(f"Batch Size: {batch_size}")
    report.append(f"Batches measured per config: {batches_to_measure}")
    report.append(f"Warmup batches: {warmup_batches}\n")
    
    # Table header
    header = f"{'Workers':<8} | {'Prefetch':<8} | {'Persist':<8} | {'CV_Thr':<8} | {'s/batch':<8} | {'imgs/sec':<8} | {'Stable'}"
    report.append(header)
    report.append("-" * len(header))
    
    best_config = None
    best_speed = -1
    
    for r in results:
        w_str = str(r['w'])
        p_str = str(r['p'])
        persist_str = str(r['persist'])
        cv_str = "Auto" if r['cv'] == 0 else str(r['cv'])
        
        if r['stable'] == "Yes":
            time_str = f"{r['time']:.3f}"
            speed_str = f"{r['imgs_sec']:.1f}"
            if r['imgs_sec'] > best_speed:
                best_speed = r['imgs_sec']
                best_config = r
        else:
            time_str = "N/A"
            speed_str = "N/A"
            
        row = f"{w_str:<8} | {p_str:<8} | {persist_str:<8} | {cv_str:<8} | {time_str:<8} | {speed_str:<8} | {r['stable']}"
        report.append(row)
        
    report.append("\nConclusion:")
    if best_config:
        report.append(f"Fastest stable configuration achieved {best_speed:.1f} imgs/sec.")
        report.append(f"Recommended Settings: num_workers={best_config['w']}, prefetch_factor={best_config['p']}, " 
                      f"persistent_workers={best_config['persist']}, cv2.setNumThreads({best_config['cv']})")
    else:
        report.append("No stable configuration found!")
        
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"\nBenchmark finished. Best speed: {best_speed:.1f} imgs/s.")
    print(f"Results saved to {report_out}")

if __name__ == "__main__":
    run_benchmarks()
