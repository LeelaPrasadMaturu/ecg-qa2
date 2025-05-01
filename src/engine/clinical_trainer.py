# ecg_vqa_system/src/engine/clinical_trainer.py
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class MedicalTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            self.optimizer.zero_grad()

            images = batch['image'].to(self.device)
            texts = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answers = batch['answer'].to(self.device)

            with autocast():
                outputs = self.model(images, texts, attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, answers)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            
            # outputs = self.model(images, texts)
            # loss = torch.nn.functional.cross_entropy(outputs, answers)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
