from pandas import DataFrame
from .utils import stratified_split, add_fingerprint_cols, standardise, xy_split
from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

class TorchSplit(Dataset): # must already take in standardised dataframe split
    def __init__(self, df: DataFrame,text_col, salt_col, conts, transformer_name:str = 'kuelumbus/polyBERT', salt_encoding:str="morgan", fpSize: int = 128):
        self.tokeniser = AutoTokenizer.from_pretrained(transformer_name)
        self.fpSize = fpSize
        self.salt_encoding = salt_encoding
        
        self.text_sequences = df[text_col]
        self.salt_sequences = df[salt_col]
        self.cont_sequences = df[conts]
        self.labels = df["conductivity"]
        self.label_counts = df[text_col].nunique()
        

    def __len__(self):
        return len(self.text_sequences)
    
    def __getitem__(self, idx):
        text = self.text_sequences[idx]
        salt_text = self.salt_sequences[idx]
        continuous_vars = self.cont_sequences[idx]
        label_var = self.labels[idx]
        
        # Tokenize the text sequence
        encoded_text = self.tokeniser(text,max_length = 512,return_tensors='pt', padding=True, truncation=True)
        
        # Process the second sequence using the autoencoder
        if self.salt_encoding == "morgan":
           self.salt_encoder = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=self.fpSize) \
                             .GetFingerprint(Chem.MolFromSmiles(salt_text))
        salt_embedding = torch.tensor(self.salt_encoder,dtype=torch.float32)
        
        # Convert continuous variables to tensor
        continuous_vars_tensor = torch.tensor(continuous_vars, dtype=torch.float32)

        label_vars_tensor = torch.tensor(label_var, dtype=torch.float32)
        
        # Return a dictionary of the encoded inputs
        return {
            'encoded_text': encoded_text,
            'second_seq_embedding': salt_embedding,
            'continuous_vars': continuous_vars_tensor,
            "label_var": label_vars_tensor
        }

class TorchDataset:
  def __init__(self,df:DataFrame):
      self.df = df
      self.folds = list()

  def process(self,args) -> List[TorchSplit]:
      
    df_list, _ = stratified_split(self.df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, nfolds = args.nfolds,verbose = args.verbose)
    for fold in tqdm(range(args.nfolds), desc = "Curr Fold"):
      train_df, val_df, test_df = df_list[fold]
      self.folds.append({"train": TorchSplit(train_df, args.text_col,args.salt_col, args.conts, args.transformer_name, args.salt_encoding, args.fpSize),
                         "val": TorchSplit(val_df, args.text_col,args.salt_col, args.conts, args.transformer_name, args.salt_encoding, args.fpSize),
                         "test":TorchSplit(test_df, args.text_col,args.salt_col, args.conts, args.transformer_name, args.salt_encoding, args.fpSize),})
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

