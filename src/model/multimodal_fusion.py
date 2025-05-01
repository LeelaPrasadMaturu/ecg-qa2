# ecg_vqa_system/src/model/multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# In src/model/multimodal_fusion.py
class MedicalCrossAttention(nn.Module):
    def __init__(self, img_dim=1792, txt_dim=512):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, 512)  # EfficientNet-B4 features
        self.txt_proj = nn.Linear(txt_dim, 512)  # BioBERT projected features
        self.attention = nn.MultiheadAttention(512, 8)
        
    def forward(self, img_feats, txt_feats):
        query = self.img_proj(img_feats).unsqueeze(0)  # [1, B, 512]
        key = self.txt_proj(txt_feats).unsqueeze(0)    # [1, B, 512]
        value = key
        attn_output, _ = self.attention(query, key, value)
        return attn_output.squeeze(0)



# In src/model/multimodal_fusion.py
class DiagnosticGate(nn.Module):
    def __init__(self, input_dim=512):  # Changed from 1024 to match actual features
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),  # Input dimension matches fusion output
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        gate = self.gate(features)
        return features * gate + features   

