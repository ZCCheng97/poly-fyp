import torch
import pandas as pd

from torch.utils.data import Dataset

class FFNDataset(Dataset):
    def __init__(self, x,y,args):

        self.df, self.y = x,y
        self.args = args
        self.indices = self._get_random_indices()

    def _get_random_indices(self):
        subset_size = int(self.args.data_fraction * len(self.df))
        return torch.randperm(len(self.df))[:subset_size]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.args.data_fraction < 1.0: idx = self.indices[idx].item()
        labels = self.y.iloc[idx]
        if self.args.poly_model_name:
            poly_inputs = self.df[f"{self.args.poly_col}_tokens"].iloc[idx]  
        else: 
            poly_fp = self.df['morgan_fp_'+self.args.poly_col].iloc[idx]
            poly_inputs = torch.tensor(poly_fp, dtype=torch.float32)

        if self.args.salt_model_name:
            salt_inputs = self.df[f"{self.args.salt_col}_tokens"].iloc[idx]
        else: 
            salt_fp = self.df['morgan_fp_'+self.args.salt_col].iloc[idx]
            salt_inputs = torch.tensor(salt_fp, dtype=torch.float32)
        
        continuous_vars = self.df[self.args.conts].iloc[idx]
        temperatures = self.df[self.args.temperature_name].iloc[idx]
        return {
            'poly_inputs': poly_inputs,
            'salt_inputs': salt_inputs,
            'continuous_vars': torch.tensor(continuous_vars.values, dtype=torch.float32),
            'temperature': torch.tensor(temperatures, dtype=torch.float32),
            'label_var': torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
        }