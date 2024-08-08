import pandas as pd
import numpy as np
from typing import List,Tuple

from sklearn.preprocessing import StandardScaler

def distribute_labels(labels, counts, target_per_group=820, n_groups=10):
    # Combine labels and counts into a list of tuples and sort by count descending
    label_counts = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)

    # Initialize empty groups
    groups = [[] for _ in range(n_groups)]
    group_counts = [0] * n_groups

    for label, count in label_counts:
        # Find the group with the smallest total count that can accommodate this label
        min_group_index = np.argmin(group_counts)
        groups[min_group_index].append(label)
        group_counts[min_group_index] += count

    return groups, group_counts

# Creates a copy of the original dataframe with specified category columns dtype converted to 'category'
def categorify(df:pd.DataFrame, cats:List[str]) -> pd.DataFrame:
    model_df = df.copy()
    if cats:
      model_df[cats] = model_df[cats].apply(lambda x: x.astype('category'))

    return model_df

def stratified_split(df, train_ratio=0.8, val_ratio=0.1, nfolds = 10,verbose = True) -> List[Tuple[pd.DataFrame]]:
  label_col = "smiles"

  df_c_len = len(df)
  train_ratio = int(train_ratio*10)
  val_ratio = int(val_ratio*10)

  train_len = df_c_len//10*train_ratio
  val_len = df_c_len//10*val_ratio
  test_len = df_c_len - train_len - val_len

  poly_smiles_counts = df[label_col].value_counts()
  poly_smiles = poly_smiles_counts.index.tolist()
  label_counts = poly_smiles_counts.values.tolist()

  # Create a copy of the original dataframe
  OCC_df = df[df[label_col] == poly_smiles[0]] # currently 3235
  leftover_df = df[df[label_col] != poly_smiles[0]]

  # Extract labels and counts, omitting the two most frequent polymers
  groups, group_counts = distribute_labels(poly_smiles[2:], label_counts[2:], target_per_group = test_len, n_groups = nfolds)

  output_list = list()
  label_counts_list = list()
  for i in range(nfolds):
    test_c = leftover_df[leftover_df[label_col].isin(groups[i])]
    test_c = test_c.reset_index(drop=True)
    curr_leftover = leftover_df[~leftover_df[label_col].isin(groups[i])]

    unique_labels = curr_leftover[label_col].unique().tolist()

    label = unique_labels.pop(0)
    val_c = curr_leftover[curr_leftover[label_col] == label]
    curr_val_len = val_c.shape[0]

    while curr_val_len < val_len:
      label = unique_labels.pop(0)
      curr_val_c = curr_leftover[curr_leftover[label_col] == label]
      val_c = pd.concat([val_c,curr_val_c], ignore_index = True)
      curr_val_len = val_c.shape[0]

    curr_train_c = curr_leftover[curr_leftover[label_col].isin(unique_labels)]
    train_c = pd.concat([OCC_df,curr_train_c], ignore_index = True)

    label_counts = train_c[label_col].nunique(),val_c[label_col].nunique(),test_c[label_col].nunique()
    train_c,val_c, test_c = train_c.drop(columns = label_col),val_c.drop(columns = label_col),test_c.drop(columns = label_col)

    output_list.append((train_c,val_c,test_c))
    label_counts_list.append(label_counts)
    
    if verbose:
      print("target vals:", train_len,val_len,test_len)
      print("final vals:",len(train_c),len(val_c),len(test_c))
      print("Unique labels:",label_counts)
  return output_list, label_counts_list

def standardise(x_train:pd.DataFrame,x_val:pd.DataFrame,x_test:pd.DataFrame, conts:List[str]) -> Tuple[pd.DataFrame]:
    scaler = StandardScaler()

    x_train[conts] = scaler.fit_transform(x_train[conts])
    x_val[conts] = scaler.transform(x_val[conts])
    x_test[conts] = scaler.transform(x_test[conts])
    return x_train, x_val, x_test

# Splits into the features (x) and labels (y)
def xy_split(x: pd.DataFrame,
             y_label:str="conductivity") -> Tuple[pd.DataFrame | pd.Series]:

    x,y = x.drop(columns = [y_label]), x[y_label]

    return x,y