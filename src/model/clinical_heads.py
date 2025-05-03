# ecg_vqa_system/src/model/clinical_heads.py
import torch.nn as nn

class ClinicalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=-1)  # Better for numerical stability
        )

    def forward(self, x):
        return self.head(x)
