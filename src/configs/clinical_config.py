# ecg_vqa_system/src/configs/clinical_config.py
from transformers import AutoTokenizer

class ClinicalConfig:
    def __init__(self):
        self.dataset_mean = 2.2539  
        self.dataset_std = 0.7426
        self.warmup_steps = 1000
        self.img_size = 512
        self.batch_size = 32
        self.lr = 3e-5
        self.epochs = 20
        self.model_save_path = "clinical_models/saved_models/best_model.pt"
        self.tokenizer = "monologg/biobert_v1.1_pubmed"