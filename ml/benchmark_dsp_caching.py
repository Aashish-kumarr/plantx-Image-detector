import os
import csv
import time
import random
import shutil
import psutil
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")
from src.datasets.model1_dsp_preprocessing import apply_dsp_preprocessing
from src.datasets.model1_transforms import load_normalization

def get_disk_free_gb(path="."):
    usage = psutil.disk_usage(os.path.abspath(path))
    return usage.free / (1024 ** 3)

class CachedDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # Dummy label
        return img, torch.tensor(1, dtype=torch.long)

def run_benchmark():
    random.seed(42)
    split_csv = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_split.csv"
    cache_dir = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\cache\model1_dsp_benchmark"
    report_out = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\reports\model1_dsp_cache_benchmark.txt"
    norm_json = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model1_normalization.json"
    
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    
    png_dir = os.path.join(cache_dir, "png")
    webp_dir = os.path.join(cache_dir, "webp")
    os.makedirs(png_dir)
    os.makedirs(webp_dir)
    
    # Select 1000 images
    pv, caltech, places = [], [], []
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] == "plantvillage": pv.append(row["image_path"])
            elif row["source"] == "caltech256": caltech.append(row["image_path"])
            elif row["source"] == "places365": places.append(row["image_path"])
            
    samples = random.sample(pv, 400) + random.sample(caltech, 300) + random.sample(places, 300)
    
    report = []
    report.append("Model 1 DSP Cache Benchmark Report")
    report.append("==================================")
    
    free_gb = get_disk_free_gb()
    report.append(f"Available disk space: {free_gb:.2f} GB\n")
    
    print("Generating caches...")
    start_time = time.time()
    
    png_paths = []
    webp_paths = []
    integrity_failures = 0
    
    for i, path in enumerate(samples):
        basename = f"img_{i:04d}"
        try:
            original = Image.open(path).convert('RGB')
            # Produce DSP RGB image
            dsp_img = apply_dsp_preprocessing(original)
            
            # Save PNG
            png_path = os.path.join(png_dir, f"{basename}.png")
            dsp_img.save(png_path, "PNG")
            png_paths.append(png_path)
            
            # Save Lossless WebP
            webp_path = os.path.join(webp_dir, f"{basename}.webp")
            dsp_img.save(webp_path, "WEBP", lossless=True)
            webp_paths.append(webp_path)
            
            # Integrity check on first 10
            if i < 10:
                dsp_np = np.array(dsp_img)
                png_np = np.array(Image.open(png_path).convert('RGB'))
                webp_np = np.array(Image.open(webp_path).convert('RGB'))
                if not np.array_equal(dsp_np, png_np) or not np.array_equal(dsp_np, webp_np):
                    integrity_failures += 1
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    cache_time = time.time() - start_time
    report.append(f"Cache-generation speed: {len(samples)} images in {cache_time:.2f}s ({len(samples)/cache_time:.1f} imgs/s)")
    report.append(f"Integrity check failures (lossless verification): {integrity_failures}")
    
    def measure_dir(d):
        total_size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        return total_size
        
    png_size_mb = measure_dir(png_dir) / (1024 * 1024)
    webp_size_mb = measure_dir(webp_dir) / (1024 * 1024)
    
    TOTAL_DS = 111087
    
    report.append("\nStorage Formats Tested:")
    report.append("  Format: PNG")
    report.append(f"    Size per 1000 images: {png_size_mb:.2f} MB")
    report.append(f"    Average size per image: {(png_size_mb/1000)*1024:.2f} KB")
    report.append(f"    Projected size for {TOTAL_DS} images: {(png_size_mb / 1000 * TOTAL_DS) / 1024:.2f} GB")
    
    report.append("  Format: Lossless WebP")
    report.append(f"    Size per 1000 images: {webp_size_mb:.2f} MB")
    report.append(f"    Average size per image: {(webp_size_mb/1000)*1024:.2f} KB")
    report.append(f"    Projected size for {TOTAL_DS} images: {(webp_size_mb / 1000 * TOTAL_DS) / 1024:.2f} GB")
    
    projected_best = min(png_size_mb, webp_size_mb) / 1000 * TOTAL_DS / 1024
    if free_gb - projected_best < 5.0:
        report.append(f"\nWARNING: Projected cache size leaves less than 5GB free. NOT RECOMMENDED to cache entire dataset.")
    else:
        report.append(f"\nDisk check: Projected cache size is safe (>5GB remaining).")
        
    # Throughput benchmark
    mean, std = load_normalization(norm_json)
    cached_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # We will test webp throughput
    # Just replicate the paths so we have enough batches (10 * 64 = 640 images)
    ds = CachedDataset(webp_paths * 5, transform=cached_train_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    
    print("Running Dataloader speed test...")
    start_loader = time.time()
    iterator = iter(loader)
    for i in range(10):
        next(iterator)
    loader_time = time.time() - start_loader
    
    imgs_per_sec = (10 * 64) / loader_time
    old_imgs_per_sec = 49.7
    
    report.append("\nPerformance:")
    report.append(f"  Current on-the-fly throughput: ~{old_imgs_per_sec:.1f} imgs/s")
    report.append(f"  Cached DataLoader throughput: {imgs_per_sec:.1f} imgs/s (tested on WebP)")
    speedup = imgs_per_sec / old_imgs_per_sec
    report.append(f"  Estimated speed improvement: {speedup:.1f}x faster")
    
    if webp_size_mb < png_size_mb:
        report.append("\nRecommended cache format: Lossless WebP (smaller size, fast loading).")
    else:
        report.append("\nRecommended cache format: PNG (smaller size, fast loading).")
        
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print("Benchmark complete!")

if __name__ == "__main__":
    run_benchmark()
