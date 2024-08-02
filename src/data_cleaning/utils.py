from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, AllChem, rdFingerprintGenerator
from rdkit.Chem.Lipinski import HeavyAtomCount
import rdkit
import math

def create_long_smiles(smile, req_length):
    # check if smile is a polymer
    if "Cu" in smile:
        # calculate required repeats so smiles > 30 atoms long
        num_heavy = HeavyAtomCount(MolFromSmiles(smile)) - 2
        repeats = math.ceil(req_length / num_heavy) - 1

        # if polymer is less than 30 long, repeat until 30 long
        if repeats > 0:
            try:
                # code to increase length of monomer
                mol = MolFromSmiles(smile)
                new_mol = mol

                # join repeats number of monomers into polymer
                for i in range(repeats):
                    # join two polymers together at Cu and Au sites
                    rxn = AllChem.ReactionFromSmarts("[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]")
                    results = rxn.RunReactants((mol, new_mol))
                    assert len(results) == 1 and len(results[0]) == 1, smile
                    new_mol = results[0][0]

                new_smile = MolToSmiles(new_mol)

            except:
                # make smile none if reaction fails
                return "None"

        # if monomer already long enough use 1 monomer unit
        else:
            new_smile = smile

        # caps ends of polymers with carbons
        new_smile = (
            new_smile.replace("[Cu]", "C").replace("[Au]", "C").replace("[Ca]", "C")
        )

    else:
        new_smile = smile

    # make sure new smile in cannonical
    long_smile = MolToSmiles(MolFromSmiles(new_smile))
    return long_smile

def add_long_smiles(df, req_length = 30):
  df = df.copy()
  df["monomer_smiles"] = df["smiles"]
  smiles = df["smiles"].value_counts().index

  for smile in smiles:
      if isinstance(smile, str):
          idx = df.index[df["smiles"] == smile].tolist()
          long_smile = create_long_smiles(smile, req_length)
          df.loc[idx, "smiles"] = long_smile
  return df

def remove_all_na(df):
  df = df.copy()
  df = df.dropna()
  return df
def fill_salt_with_Li(df):
  df = df.copy()
  df["salt smiles"].fillna("[Li+]", inplace=True)
  return df
def fill_molality(df):
  df = df.copy()
  df.loc[:, "molality"].fillna(0, inplace=True)
  return df
def fill_mw(df):
  df = df.copy()
  df.loc[:, "mw"].fillna(65000, inplace=True)
  df.loc[:, "mw"] = np.log10(df["mw"])
  return df
def add_temperature_K(df):
  df = df.copy()
  df["temperature_K"] = df.temperature.apply(lambda x: x+273)
  return df