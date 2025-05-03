# ecg_vqa_system/scripts/train.py
import torch
from torch.utils.data import DataLoader
from src.data.clinical_loader import ClinicalECGDataset
from src.model.clinical_vqa import ClinicalVQAModel
from src.engine.clinical_trainer import MedicalTrainer
from src.engine.clinical_evaluator import ClinicalEvaluator
from src.configs.clinical_config import ClinicalConfig
from torch.optim.lr_scheduler import OneCycleLR


def main():
    config = ClinicalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Adjust for Colab or low-resource setup
    config.batch_size = 4
    config.image_encoder = 'efficientnet-b0'
    config.use_amp = True

    # Load training and validation datasets
    train_dataset = ClinicalECGDataset('clinical_data/train.json', 'clinical_data/', config.tokenizer)
    val_dataset = ClinicalECGDataset('clinical_data/valid.json', 'clinical_data/', config.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ClinicalVQAModel(image_encoder=config.image_encoder).to(device)

    # optimizer = torch.optim.AdamW([
    #     {'params': model.image_encoder.parameters(), 'lr': 1e-5},
    #     {'params': model.text_encoder.parameters(), 'lr': 2e-5},
    #     {'params': model.cross_attn.parameters(), 'lr': 3e-5},
    #     {'params': model.classifier.parameters(), 'lr': 1e-4}
    # ], weight_decay=1e-5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = OneCycleLR(optimizer, max_lr=3e-4,steps_per_epoch=len(train_loader),epochs=config.epochs)

    trainer = MedicalTrainer(model, train_loader, val_loader, optimizer, device)
    evaluator = ClinicalEvaluator(model, val_loader, device)

    best_f1 = 0

    for epoch in range(config.epochs):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} - Training Loss: {loss:.4f}")

        metrics = evaluator.evaluate()
        print(f"Validation Accuracy: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), config.model_save_path)
            print("✅ New best model saved.")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
