from pandas import DataFrame
from .utils import categorify,stratified_split, add_fingerprint_cols, standardise, xy_split
from typing import List
from tqdm import tqdm

class DataSplit:
  def __init__(self,tr,val,te, label_col):
      self.label_counts = (tr[label_col].nunique(),val[label_col].nunique(),te[label_col].nunique())
      
      self.x_train, self.y_train = xy_split(tr)
      self.x_val, self.y_val = xy_split(val)
      self.x_test, self.y_test = xy_split(te)

class TabularDataset:
  def __init__(self,df:DataFrame):
      
      self.df = df
      self.folds = list()

  def process(self,args) -> List[DataSplit]:
    df_categorify = self.df.pipe(categorify,cats=args.cats)
    df_list = stratified_split(df_categorify, label_col = args.label_col, train_ratio=args.train_ratio, val_ratio=args.val_ratio, nfolds = args.nfolds,verbose = args.verbose)
    for fold in tqdm(range(args.nfolds, desc = "Curr Fold")):
      train_df, val_df, test_df = df_list[fold]

      train_df, val_df, test_df = (add_fingerprint_cols(train_df,args.fpSize,args.polymer_use_fp, args.salt_use_fp), 
                                  add_fingerprint_cols(val_df, args.fpSize, args.polymer_use_fp, args.salt_use_fp), 
                                  add_fingerprint_cols(test_df,args.fpSize, args.polymer_use_fp, args.salt_use_fp))

      train_dfs, val_dfs, test_dfs = standardise(train_df, val_df, test_df, conts = args.conts)

      curr_fold = DataSplit(train_dfs,val_dfs,test_dfs, label_col = args.label_col)

      self.folds.append(curr_fold)
    return self.folds

