import numpy as np
import pandas as pd
import torch

from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

def smiles_to_tokens(df,col_name, tokenizer):
    df = df.copy()
    df["tokens"] = df[col_name].apply(lambda x: tokenizer(x, max_length = 512,return_tensors='pt', padding="max_length", truncation=True))
    return df