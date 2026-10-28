import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

import sys
sys.path.append(r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector")

from PIL import Image
from src.datasets.model1_dsp_preprocessing import apply_dsp_preprocessing

class Model2NormDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == 'train'].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        # Apply DSP
        pil_img = Image.open(img_path).convert('RGB')
        dsp_pil = apply_dsp_preprocessing(pil_img)
        img_rgb = np.array(dsp_pil)
        # Convert to float [0, 1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return img_tensor

def compute_normalization():
    csv_path = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model2_split.csv"
    output_json = r"C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model2_normalization.json"
    
    print("Initializing Model 2 Normalization Dataset...")
    dataset = Model2NormDataset(csv_path)
    
    loader = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Total training images to process: {len(dataset)}")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    for images in tqdm(loader, desc="Computing Normalization"):
        # images: [B, 3, H, W]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1) # [B, 3, H*W]
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
        
    mean /= total_images
    std /= total_images
    
    norm_stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "number_of_images": total_images,
        "image_size": 224,
        "split": "train",
        "dsp_preprocessing": True
    }
    
    with open(output_json, 'w') as f:
        json.dump(norm_stats, f, indent=4)
        
    print(f"Saved {output_json}")
    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")

if __name__ == "__main__":
    compute_normalization()
