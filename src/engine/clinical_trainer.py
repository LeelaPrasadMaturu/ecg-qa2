# ecg_vqa_system/src/engine/clinical_trainer.py
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class MedicalTrainer:
    def __init__(self, config ,model, train_loader, val_loader, optimizer, device):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=config.use_amp)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            self.optimizer.zero_grad()

            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answers = batch['answer'].to(self.device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(images, input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, answers)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)    
            torch.nn.utils.clip_grad_norm_( self.model.parameters(),max_norm=0.5, error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

