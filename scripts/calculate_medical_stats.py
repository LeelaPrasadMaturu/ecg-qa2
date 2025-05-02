# ecg_vqa_system/scripts/calculate_medical_stats.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.data.clinical_transforms import ClinicalTransforms  # Use your actual transforms

class ECGStatsDataset(Dataset):
    """Medical-grade dataset for calculating image statistics"""
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.endswith('.png')
        ]
        # Use validation transforms (no augmentation)
        self.transform = ClinicalTransforms(img_size=512).val  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        return self.transform(img)

def calculate_medical_stats(image_dir, batch_size=32, num_workers=4):
    """Calculate dataset statistics following clinical standards"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = ECGStatsDataset(image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, 
                       num_workers=num_workers, shuffle=False)

    print(f"Calculating statistics for {len(dataset)} clinical ECG images...")

    # Medical-grade precision calculation
    pixel_sum = torch.tensor(0.0, device=device)
    pixel_sq_sum = torch.tensor(0.0, device=device)
    num_pixels = 0

    for batch in tqdm(loader, desc="Processing ECG Images"):
        batch = batch.to(device)
        pixel_sum += batch.sum()
        pixel_sq_sum += (batch ** 2).sum()
        num_pixels += batch.numel()

    # Final calculations with double precision
    mean = (pixel_sum / num_pixels).cpu().double()
    std = torch.sqrt(
        (pixel_sq_sum / num_pixels) - (mean ** 2)
    ).cpu().double()

    return mean.item(), std.item()

if __name__ == "__main__":
    image_dir = "clinical_data/ecgImages"  
    mean, std = calculate_medical_stats(image_dir)
    
    print("\nClinical Dataset Statistics:")
    print(f"Mean: {mean:.4f}")
    print(f"Std Dev: {std:.4f}")
    print(f"Recommended normalization: T.Normalize([{mean:.4f}], [{std:.4f}])")