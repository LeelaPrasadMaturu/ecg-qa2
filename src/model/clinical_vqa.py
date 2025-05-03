# ecg_vqa_system/src/model/clinical_vqa.py
import torch
import torch.nn as nn
from .clinical_encoders import MedicalImageEncoder, ClinicalTextEncoder
from .multimodal_fusion import MedicalCrossAttention, DiagnosticGate
from .clinical_heads import ClinicalClassifier
    


class ClinicalVQAModel(nn.Module):
    def __init__(self, image_encoder='efficientnet-b0'): 
        super().__init__()
        self.image_encoder = MedicalImageEncoder(encoder_name=image_encoder)
        self.text_encoder = ClinicalTextEncoder()
        self.cross_attn = MedicalCrossAttention()
        self.diagnostic_gate = DiagnosticGate(input_dim=512)
        self.classifier = ClinicalClassifier()

    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_encoder(images)
        txt_features = self.text_encoder(input_ids, attention_mask)
        fused_features = self.cross_attn(img_features, txt_features)
        gated_features = self.diagnostic_gate(fused_features)
        return self.classifier(gated_features)

