# ecg_vqa_system/src/model/multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalCrossAttention(nn.Module):
    def __init__(self, img_dim=1280, txt_dim=768):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, 256)
        self.txt_proj = nn.Linear(txt_dim, 256)
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)  # Critical fix
        
    def forward(self, img_feats, txt_feats):
        query = self.img_proj(img_feats).unsqueeze(1)  
        key = self.txt_proj(txt_feats).unsqueeze(1)    
        attn_output, _ = self.attention(query, key, key)
        return attn_output.squeeze(1)


class DiagnosticGate(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):  
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),  
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, features):
        gate = self.gate(features)
        transformed_features = self.feature_transform(features)
        return 0.7*features + 0.3*transformed + (gate * features * transformed)


