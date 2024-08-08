from pandas import DataFrame
from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

from .fingerprinting import add_fingerprint_cols
from .tensorify import smiles_to_tokens, smiles_to_fpbinary
from .utils import stratified_split, standardise, xy_split

class TorchDataset:
  def __init__(self,df:DataFrame):
      self.df = df
      self.folds = list()

  def process(self,args) -> List[dict]:
    tokeniser = AutoTokenizer.from_pretrained(args.transformer_name)
    self.df = smiles_to_tokens(self.df, args.text_col,tokeniser)
    self.df = smiles_to_fpbinary(self.df, args.salt_col, args.fpSize)

    df_list, label_counts_list = stratified_split(self.df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, nfolds = args.nfolds,verbose = args.verbose)
    for fold in tqdm(range(args.nfolds), desc = "Curr Fold"):
      train_df, val_df, test_df = df_list[fold]
      train_df, val_df, test_df = standardise(train_df, val_df, test_df, conts = args.conts)

      curr_fold = TabularSplit(train_df,val_df,test_df, label_counts = label_counts_list[fold])
      self.folds.append(curr_fold)
    return self.folds

class TabularSplit:
  def __init__(self,tr,val,te, label_counts):
      self.label_counts = label_counts
      
      self.x_train, self.y_train = xy_split(tr)
      self.x_val, self.y_val = xy_split(val)
      self.x_test, self.y_test = xy_split(te)

class TabularDataset:
  def __init__(self,df:DataFrame):
      self.df = df
      self.folds = list()

  def process(self,args) -> List[TabularSplit]:
    if args.polymer_use_fp == "polybert":
      tokeniser = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
      model = AutoModel.from_pretrained('kuelumbus/polyBERT')
    else:
      tokeniser = None
      model = None
      
    df_list, label_counts_list = stratified_split(self.df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, nfolds = args.nfolds,verbose = args.verbose)
    for fold in tqdm(range(args.nfolds), desc = "Curr Fold"):
      train_df, val_df, test_df = df_list[fold]

      train_df, val_df, test_df = (add_fingerprint_cols(train_df,args.fpSize,args.polymer_use_fp, args.salt_use_fp, tokeniser,model), 
                                  add_fingerprint_cols(val_df, args.fpSize, args.polymer_use_fp, args.salt_use_fp, tokeniser,model), 
                                  add_fingerprint_cols(test_df,args.fpSize, args.polymer_use_fp, args.salt_use_fp, tokeniser,model))

      train_dfs, val_dfs, test_dfs = standardise(train_df, val_df, test_df, conts = args.conts)

      curr_fold = TabularSplit(train_dfs,val_dfs,test_dfs, label_counts = label_counts_list[fold])

      self.folds.append(curr_fold)
    return self.folds