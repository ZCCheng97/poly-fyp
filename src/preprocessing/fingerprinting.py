import numpy as np
import pandas as pd
import torch

from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

def array_to_cols(df, fp_type, col_name, fpSize):
    dataframe_list = []
    df = df.copy()
    for i in range(df.shape[0]):
      array = np.array(df[f'{fp_type}_{col_name}'][i])
      dataframe_i = pd.DataFrame(array)
      dataframe_i = dataframe_i.T
      dataframe_list.append(dataframe_i)

    # To concatenate the list of dataFrames into a single DataFrame, and renaming the columns with a 'polymer_mp_' prefix.
    concatenated_fp_only = pd.concat(dataframe_list, ignore_index=True)
    concatenated_fp_only.columns = [f'{fp_type}_{col_name}' + str(i) for i in range(fpSize)]
    # To join the fingerprint dataFrame with the original chem_dataframe
    df = concatenated_fp_only.join(df, how = "outer")
    df.drop(columns = [col_name,f'{fp_type}_{col_name}'], inplace = True)
    return df

def add_fingerprint_cols(df,fpSize, polymer_use_fp, salt_use_fp, poly_tokeniser,poly_model, salt_tokeniser,salt_model):
  if polymer_use_fp == "morgan":
     df = smiles_to_fingerprint(df, "long_smiles",fpSize=fpSize)
  if polymer_use_fp == "morgan_monomer":
     df = smiles_to_fingerprint(df, "monomer_smiles",fpSize=fpSize)
  if polymer_use_fp == "polybert":
     df = smiles_to_encoding(df, "psmiles", tokeniser=poly_tokeniser, model=poly_model, model_name = polymer_use_fp)
  if salt_use_fp == "morgan":
     df = smiles_to_fingerprint(df,col_name="salt smiles",fpSize=fpSize)
  if salt_use_fp == "chemberta":
     df = smiles_to_encoding(df, col_name="salt smiles", tokeniser=salt_tokeniser, model=salt_model, model_name = salt_use_fp)
  return df

def smiles_to_encoding(df,col_name, tokeniser, model, model_name, to_cols = True):
  df = df.copy()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  def get_embeddings(texts):
    encoded_input = tokeniser(texts,max_length = 512,return_tensors='pt', padding=True, truncation=True)
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    fingerprints = mean_pooling(outputs, encoded_input['attention_mask'])
    return fingerprints

  unique_psmiles = list(df[col_name].unique())
  unique_embeddings = get_embeddings(unique_psmiles)
  embedding_dict = {desc: embedding.cpu().numpy() for desc, embedding in zip(unique_psmiles, unique_embeddings)}
  df[f'{model_name}_fp_'+col_name] = df[col_name].map(embedding_dict)
  if to_cols: df = array_to_cols(df, f"{model_name}_fp", col_name, fpSize = unique_embeddings.size(1))
  return df

def smiles_to_fingerprint(df,col_name, fpSize = 128, to_cols = True):
    # Based on https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    df = df.copy()
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=fpSize)

    df['morgan_fp_'+col_name] = df[col_name].apply(lambda x: mfpgen.GetFingerprint(Chem.MolFromSmiles(x)))

    if to_cols: df = array_to_cols(df, "morgan_fp", col_name, fpSize = fpSize)
    return df