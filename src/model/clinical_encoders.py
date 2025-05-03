# ecg_vqa_system/src/model/clinical_encoders.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import AutoModel

class MedicalImageEncoder(nn.Module):
    def __init__(self, encoder_name='efficientnet-b0'):
        super().__init__()
        self.base = EfficientNet.from_pretrained(encoder_name, in_channels=1) 
        self.base._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.base.extract_features(x)
        x = self.adaptive_pool(x)  
        return x.flatten(1)

class ClinicalTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.biobert = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
        self.projection = nn.Linear(768, 512)  # Proper dimension projection
        
    def forward(self, input_ids, attention_mask=None):
        # Pass attention_mask to handle padding properly
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        return self.projection(pooled)
