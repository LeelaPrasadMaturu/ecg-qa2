# ecg_vqa_system/src/data/clinical_transforms.py
import torchvision.transforms as T
from src.configs.clinical_config import ClinicalConfig

class ClinicalTransforms:
    def __init__(self, img_size=512):
        config = ClinicalConfig()
        
        self.train = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.RandomChoice([
                T.RandomAffine(degrees=0, translate=(0.03,0)), 
                T.RandomAffine(degrees=0, scale=(0.97, 1.03))
            ]),
            T.RandomAutocontrast(),
            T.Lambda(lambda x: x.convert('L')),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([config.dataset_mean], [config.dataset_std])
        ])
        
        self.val = T.Compose([
            T.Lambda(lambda x: x.convert('L')),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([config.dataset_mean], [config.dataset_std])
        ])