import numpy as np
import pandas as pd
import torch

from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

def smiles_to_fpbinary(df,col_name, fpSize = 128, salt_encoding ="morgan"):
    # Based on https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    df = df.copy()

    if salt_encoding == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=fpSize)

    df[f"{salt_encoding}_{col_name}"] = df[col_name].apply(lambda x: gen.GetFingerprint(Chem.MolFromSmiles(x)))
    return df

def smiles_to_tokens(df,col_name, tokenizer):
    df = df.copy()
    df["tokens"] = df[col_name].apply(lambda x: tokenizer(x, max_length = 512,return_tensors='pt', padding="max_length", truncation=True))
    return df