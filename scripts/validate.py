# ecg_vqa_system/scripts/validation.py

import torch
from torch.utils.data import DataLoader
from src.data.clinical_loader import ClinicalECGDataset
from src.engine.clinical_evaluator import ClinicalEvaluator
from src.model.clinical_vqa import ClinicalVQAModel
from src.configs.clinical_config import ClinicalConfig


def main():
    config = ClinicalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = ClinicalECGDataset(
        'clinical_data/valid.json',
        'clinical_data/',
        config.tokenizer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = ClinicalVQAModel().to(device)
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))

    evaluator = ClinicalEvaluator(model, val_loader, device)
    metrics = evaluator.evaluate()

    print("Validation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
