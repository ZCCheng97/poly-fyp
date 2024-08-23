import torch
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import spearmanr

from .utils import arrhenius_score

class Tester:
    def __init__(self, model, device,arrhenius=False):
        self.model = model
        self.device = device
        self.arrhenius = arrhenius
        self.funcs = [mean_absolute_error,mean_squared_error, spearmanr,r2_score]
    
    def __call__(self, test_dataloader):
        self.model.eval()
        all_outputs = list()
        all_labels = list()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Test batch", total=len(test_dataloader)):
                text_input = batch['poly_inputs']
                salt_input = batch['salt_inputs']
                continuous_vars = batch['continuous_vars']
                labels = batch['label_var']
                if self.arrhenius:
                    temperatures = batch['temperature']
                    temperatures = temperatures.to(self.device)

                # Move tensors to device
                text_input = text_input.to(self.device) 
                salt_input = salt_input.to(self.device)
                continuous_vars = continuous_vars.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(text_input, salt_input, continuous_vars)
                if self.arrhenius:
                    outputs = arrhenius_score(outputs, temperatures)

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
            
        all_outputs = torch.cat(all_outputs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        scores = np.array([func(all_labels,all_outputs) if func != spearmanr else func(all_labels,all_outputs).statistic for func in self.funcs])

        return scores, all_labels, all_outputs

class Engine:
    def __init__(self, model, criterion, optimizer, device, accumulation_steps, arrhenius=False, regularisation = 0.0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.arrhenius = arrhenius
        self.regularisation = regularisation

    def __call__(self, train_dataloader, val_dataloader):
        train_loss, val_loss = 0.0,0.0

        self.model.train()
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training batch", total=len(train_dataloader))):
            text_input = batch['poly_inputs']
            salt_input = batch['salt_inputs']
            continuous_vars = batch['continuous_vars']
            labels = batch['label_var']
            if self.arrhenius:
                temperatures = batch['temperature']
                temperatures = temperatures.to(self.device)

            # Move tensors to device
            text_input = text_input.to(self.device) 
            salt_input = salt_input.to(self.device)
            continuous_vars = continuous_vars.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(text_input, salt_input, continuous_vars)
            if self.arrhenius:
                lnA,Ea = outputs[:,0],outputs[:,1]
                outputs = arrhenius_score(outputs,temperatures)

            if self.arrhenius:
                loss = self.criterion(outputs, labels) + self.arrhenius_reg(lnA,Ea,self.regularisation)
            else:
                loss = self.criterion(outputs, labels)

            train_loss += loss.item()
            loss = loss/self.accumulation_steps
            
            # Backward pass and optimization
            loss.backward()
            if ((i + 1) % self.accumulation_steps == 0) or (i == len(train_dataloader) - 1):
                self.optimizer.step()  # Update the model's parameters
                self.optimizer.zero_grad()

        train_loss = train_loss / len(train_dataloader)
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation batch", total=len(val_dataloader)):
                text_input = batch['poly_inputs']
                salt_input = batch['salt_inputs']
                continuous_vars = batch['continuous_vars']
                labels = batch['label_var']

                if self.arrhenius:
                    temperatures = batch['temperature']
                    temperatures = temperatures.to(self.device)

                # Move tensors to device
                text_input = text_input.to(self.device) 
                salt_input = salt_input.to(self.device)
                continuous_vars = continuous_vars.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(text_input, salt_input, continuous_vars)
                if self.arrhenius:
                    outputs = arrhenius_score(outputs,temperatures)
                
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
            
        val_loss = val_loss / len(val_dataloader)

        return train_loss, val_loss

    def arrhenius_reg(self,lnA, Ea, reg_term=0.0):
        #fit terms and int range taken from experimental fit data and correlation between Ea and lnA
        slope = 31.999564332937958
        intercept = -10.412690081157656
        int_range = 3.06
        R = 8.63e-5

        #get expected value of lnA
        exp_lnA = Ea*R * slope + intercept
        
        #get absolute value of all residuals, subtract 15 from each
        residuals = torch.abs(exp_lnA - lnA) - int_range
        
        #zero out all residuals within certain distance of fit line
        #eg don't punish for values within 3 units of fit line
        residuals = torch.max(residuals,torch.zeros(residuals.shape).to(residuals.device))
    
        return reg_term*torch.sum(residuals)