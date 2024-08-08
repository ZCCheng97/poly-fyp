import torch
from torch.utils.data import Dataset

class FFNDataset(Dataset):
    def __init__(self, x,y,args):
        self.df = x
        self.y = y
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        labels = self.y.iloc[idx]
        tokens = self.df['tokens'].iloc[idx]
        salt_input = self.df[f"{self.args.salt_encoding}_{self.args.salt_col}"].iloc[idx]
        continuous_vars = self.df[self.args.conts].iloc[idx]
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'salt_embedding': torch.tensor(salt_input, dtype=torch.float32),
            'continuous_vars': torch.tensor(continuous_vars.values, dtype=torch.float32),
            'label_var': torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
        }