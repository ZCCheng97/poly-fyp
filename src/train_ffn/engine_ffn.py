import torch
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import spearmanr

class Tester:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.funcs = [mean_absolute_error,mean_squared_error, spearmanr,r2_score]
    
    def __call__(self, test_dataloader):
        self.model.eval()
        all_outputs = list()
        all_labels = list()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Test batch", total=len(test_dataloader)):
                text_input = batch['input_ids'].squeeze(1) 
                attention_mask = batch['attention_mask'].squeeze(1)
                salt_input = batch['salt_embedding']
                continuous_vars = batch['continuous_vars']
                labels = batch['label_var']

                # Move tensors to device
                text_input = text_input.to(self.device) 
                attention_mask = attention_mask.to(self.device)
                salt_input = salt_input.to(self.device)
                continuous_vars = continuous_vars.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(text_input, attention_mask, salt_input, continuous_vars)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
            
        all_outputs = torch.cat(all_outputs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        scores = np.array([func(all_labels,all_outputs) if func != spearmanr else func(all_labels,all_outputs).statistic for func in self.funcs])

        return scores


class Engine:
    def __init__(self, model, criterion, optimizer, device, accumulation_steps):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps

    def __call__(self, train_dataloader, val_dataloader):
        train_loss, val_loss = 0.0,0.0

        self.model.train()
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training batch", total=len(train_dataloader))):
            text_input = batch['input_ids'].squeeze(1) 
            attention_mask = batch['attention_mask'].squeeze(1)
            salt_input = batch['salt_embedding']
            continuous_vars = batch['continuous_vars']
            labels = batch['label_var']

            # Move tensors to device
            text_input = text_input.to(self.device) 
            attention_mask = attention_mask.to(self.device)
            salt_input = salt_input.to(self.device)
            continuous_vars = continuous_vars.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(text_input, attention_mask, salt_input, continuous_vars)
            loss = self.criterion(outputs, labels) / self.accumulation_steps

            # Backward pass and optimization
            loss.backward()
            if ((i + 1) % self.accumulation_steps == 0) or (i == len(train_dataloader) - 1):
                self.optimizer.step()  # Update the model's parameters
                self.optimizer.zero_grad()

            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation batch", total=len(val_dataloader)):
                text_input = batch['input_ids'].squeeze(1) 
                attention_mask = batch['attention_mask'].squeeze(1)
                salt_input = batch['salt_embedding']
                continuous_vars = batch['continuous_vars']
                labels = batch['label_var']


                # Move tensors to device
                text_input = text_input.to(self.device) 
                attention_mask = attention_mask.to(self.device)
                salt_input = salt_input.to(self.device)
                continuous_vars = continuous_vars.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(text_input, attention_mask, salt_input, continuous_vars)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
            
        val_loss = val_loss / len(val_dataloader)

        return train_loss, val_loss