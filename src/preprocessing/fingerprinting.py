import numpy as np
import pandas as pd

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

def smiles_to_fingerprint(df,col_name, fpSize = 128):
    # Based on https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    df = df.copy()
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=fpSize)

    df['Morgan_fp_'+col_name] = df[col_name].apply(lambda x: mfpgen.GetFingerprint(Chem.MolFromSmiles(x)))

    df = array_to_cols(df, "Morgan_fp", col_name, fpSize = fpSize)
    return df