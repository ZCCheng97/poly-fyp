import torch
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import spearmanr
from captum.attr import LayerIntegratedGradients, visualization
import transformers

from .utils import arrhenius_score

class Visualiser:
    def __init__(self, model, device, tokeniser,arrhenius=False):
        self.model = model
        self.device = device
        self.arrhenius = arrhenius
        self.tokeniser = tokeniser
    
    def __call__(self, test_dataloader):
        self.model.eval()
        vis_record = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Visualising", total= test_dataloader.batch_sampler.sampler_len):
                text_input = batch['poly_inputs']
                salt_input = batch['salt_inputs']
                continuous_vars = batch['continuous_vars']
                labels = batch['label_var']

                # Move tensors to device
                text_input = text_input.to(self.device) 
                if isinstance(text_input,(dict, transformers.tokenization_utils_base.BatchEncoding)) and "attention_mask" in text_input:
                    attention_mask = text_input["attention_mask"].squeeze(1)
                    text_input = text_input["input_ids"].squeeze(1)
                else:
                    attention_mask = None 
                salt_input = salt_input.to(self.device)
                continuous_vars = continuous_vars.to(self.device)
                labels = labels.to(self.device)    
                
                lig = LayerIntegratedGradients(self.model, self.model.polymerencoder.model.embeddings)
                # Compute attributions based on the use_pretrained_embeddings flag
                attributions, delta = lig.attribute( # attributions.shape = [batch_size,512,600] for polyBERT
                    inputs=text_input,  # This assumes use_pretrained_embeddings=True
                    additional_forward_args=(attention_mask, salt_input, continuous_vars),
                    return_convergence_delta=True
                )
                outputs = self.model(text_input, attention_mask, salt_input, continuous_vars)

                # Sum the attributions over embedding dimensions
                attributions_sum = attributions.sum(dim=-1).squeeze(0) # shape = [batch_size, 512]
                tokens = self.tokeniser.convert_ids_to_tokens(text_input.squeeze(0).cpu())

                # Filter out [PAD] tokens
                filtered_tokens = [token for token in tokens if token != "[PAD]"]
                filtered_attributions = [attributions_sum[i] for i in range(len(tokens)) if tokens[i] != "[PAD]"]
                
                 # Create the visualization data record
                vis_record.append(visualization.VisualizationDataRecord(
                    filtered_attributions, 
                    outputs.item(),
                    0, labels.item(), None, sum(filtered_attributions), 
                    filtered_tokens, 
                    convergence_score=delta.cpu()
                ))

        return vis_record, (filtered_tokens, [t.item() for t in filtered_attributions])

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
                if isinstance(text_input,(dict, transformers.tokenization_utils_base.BatchEncoding)) and "attention_mask" in text_input:
                    attention_mask = text_input["attention_mask"].squeeze(1)
                    text_input = text_input["input_ids"].squeeze(1)
                else:
                    attention_mask = None    
                salt_input = salt_input.to(self.device)
                continuous_vars = continuous_vars.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(text_input, attention_mask, salt_input, continuous_vars)
                if self.arrhenius:
                    outputs = arrhenius_score(outputs, temperatures)

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
            
        all_outputs = torch.cat(all_outputs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        scores = np.array([func(all_labels,all_outputs) if func != spearmanr else func(all_labels,all_outputs).statistic for func in self.funcs])

        return scores, all_labels, all_outputs

class Engine:
    def __init__(self, model, criterion, optimizer, grad_clip, device, accumulation_steps, arrhenius=False, regularisation = 0.0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.arrhenius = arrhenius
        self.regularisation = regularisation
    
    def gradual_unfreeze(self, curr_epoch_idx, freq):
        # Check epoch ranges to determine how many layers to unfreeze
        unfreeze_embeds = False
        if curr_epoch_idx < freq:
            # Epochs 0 to freq-1: Freeze all layers
            layers_to_unfreeze = set()
        elif curr_epoch_idx < 2 * freq:
            # Epochs freq to 2*freq-1: Unfreeze last 4 layers (indices 8 to 11)
            layers_to_unfreeze = range(8, 12)
        elif curr_epoch_idx < 3 * freq:
            # Epochs 2*freq to 3*freq-1: Unfreeze last 8 layers (indices 4 to 11)
            layers_to_unfreeze = range(4, 12)
        else:
            # Epochs 3*freq and beyond: Unfreeze all layers (indices 0 to 11)
            layers_to_unfreeze = range(0, 12)
            unfreeze_embeds = True

        for name, param in self.model.named_parameters():
            if "embeddings" in name and unfreeze_embeds:
                param.requires_grad = True
            if "layer" in name:
                layer_idx = int(name.split('.')[4])  #named as polymerencoder.model.encoder.layer.0... etc."
                if layer_idx in layers_to_unfreeze:
                    param.requires_grad = True

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
            if isinstance(text_input,(dict, transformers.tokenization_utils_base.BatchEncoding)) and "attention_mask" in text_input:
                    attention_mask = text_input["attention_mask"].squeeze(1)
                    text_input = text_input["input_ids"].squeeze(1)
            else:
                attention_mask = None    
            salt_input = salt_input.to(self.device)
            continuous_vars = continuous_vars.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(text_input, attention_mask, salt_input, continuous_vars)
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
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
                if isinstance(text_input,(dict, transformers.tokenization_utils_base.BatchEncoding)) and "attention_mask" in text_input:
                    attention_mask = text_input["attention_mask"].squeeze(1)
                    text_input = text_input["input_ids"].squeeze(1)
                else:
                    attention_mask = None    
                salt_input = salt_input.to(self.device)
                continuous_vars = continuous_vars.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(text_input, attention_mask, salt_input, continuous_vars)
                if self.arrhenius:
                    outputs = arrhenius_score(outputs,temperatures)
                
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
            
        val_loss = val_loss / len(val_dataloader)

        return train_loss, val_loss

    def arrhenius_reg(self,lnA, Ea, reg_term=0.0):
        #fit terms and int range taken from experimental fit data and correlation between Ea and lnA
        slope = 31.999279182429
        intercept = -10.41255022673735
        int_range = 3.050483449135184
        R = 8.63e-5

        #get expected value of lnA
        exp_lnA = Ea*R * slope + intercept
        
        #get absolute value of all residuals, subtract 15 from each
        residuals = torch.abs(exp_lnA - lnA) - int_range
        
        #zero out all residuals within certain distance of fit line
        #eg don't punish for values within 3 units of fit line
        residuals = torch.max(residuals,torch.zeros(residuals.shape).to(residuals.device))
    
        return reg_term*torch.sum(residuals)