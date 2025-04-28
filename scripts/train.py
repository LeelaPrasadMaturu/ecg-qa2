# ecg_vqa_system/scripts/train.py
import torch
from src.data.clinical_loader import ClinicalECGDataset
from src.model import MedicalImageEncoder, ClinicalTextEncoder, MedicalCrossAttention, ClinicalClassifier
from src.engine.clinical_trainer import MedicalTrainer
from torch.utils.data import DataLoader
from src.model.clinical_vqa import ClinicalVQAModel
from src.configs.clinical_config import ClinicalConfig


def main():
    config = ClinicalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Reduce batch size for Colab
    config.batch_size = 4  

     # Use lighter EfficientNet version
    config.image_encoder = 'efficientnet-b0'
    
    # Enable mixed precision
    config.use_amp = True
    
    
    # Initialize components
    train_dataset = ClinicalECGDataset(
        'clinical_data/train.json',
        'clinical_data/',
        config.tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model with proper architecture
    model = ClinicalVQAModel().to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-5
    )
    
    trainer = MedicalTrainer(model, train_loader, None, optimizer, device)
    
    for epoch in range(config.epochs):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f}")
    
    torch.save(model.state_dict(), config.model_save_path)

     # Add memory optimization
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
