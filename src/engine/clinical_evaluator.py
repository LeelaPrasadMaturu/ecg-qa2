# ecg_vqa_system/src/engine/clinical_evaluator.py
import torch
from sklearn.metrics import accuracy_score, f1_score
from src.utils.clinical_metrics import medical_sensitivity, medical_specificity

class ClinicalEvaluator:
    def __init__(self, model, loader, device):
        self.model = model.to(device)
        self.loader = loader
        self.device = device
        

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.loader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                answers = batch['answer'].cpu().numpy()

                outputs = self.model(images, input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(answers)

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'sensitivity': medical_sensitivity(all_labels, all_preds),
            'specificity': medical_specificity(all_labels, all_preds)
        }
