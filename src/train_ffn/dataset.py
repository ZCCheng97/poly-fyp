import torch
from torch.utils.data import Dataset

class FFNDataset(Dataset):
    def __init__(self, x,y,args):
        self.df = x
        self.y = y
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
        tokens = self.df['tokens'].iloc[idx]
        salt_input = self.df[f"{self.args.salt_encoding}_{self.args.salt_col}"].iloc[idx]
        continuous_vars = self.df[self.args.conts].iloc[idx]
        temperatures = self.df[self.args.temperature_name].iloc[idx]
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'salt_embedding': torch.tensor(salt_input, dtype=torch.float32),
            'continuous_vars': torch.tensor(continuous_vars.values, dtype=torch.float32),
            'temperature': torch.tensor(temperatures, dtype=torch.float32),
            'label_var': torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
        }