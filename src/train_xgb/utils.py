from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import spearmanr
import csv
from tqdm import tqdm
import wandb

def xgb(fold, seed_list = [42], verbose = True, params: dict = dict()):
  funcs = [mean_absolute_error,mean_squared_error, spearmanr,r2_score]
  outputs = []
  for seed in tqdm(seed_list, desc= "Seed"):
    model = XGBRegressor(seed = seed, subsample = 0.8) if not params else XGBRegressor(seed = seed, subsample = 0.8,**params)
    model.fit(fold.x_train,fold.y_train)
    y_pred = model.predict(fold.x_test)
    res = [func(fold.y_test,y_pred) if func != spearmanr else func(fold.y_test,y_pred).statistic for func in funcs]
    outputs.append(res)
  outputs = np.array(outputs)
  res_mean = np.mean(outputs, axis = 0)
  res_std = np.std(outputs, axis = 0)
  if verbose:
    for func in funcs:
      print(f"{func.__name__}: {res_mean[funcs.index(func)]}")
  return res_mean, res_std, y_pred

def save_results(path, df, labels, preds):
    df = df.copy()

    # Add these as new columns to the dataframe
    df['actual_conductivity'] = labels
    df['predicted_conductivity'] = preds

    # Save the dataframe as a .csv file
    df.to_csv(path, index=False)

def logger(data_dict:dict, csv_file_path:str, use_wandb: bool= False):
  if use_wandb: wandb.log(data_dict)
  file_exists = False
  try:
    with open(csv_file_path, 'r') as csvfile:
        file_exists = True
  except FileNotFoundError:
    pass

  with open(csv_file_path, 'a', newline='') as csvfile:
    fieldnames = data_dict.keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()

    writer.writerow(data_dict)

def load_best_params(best_params:str="", verbose: bool=False) -> dict:
  params = dict()
  if best_params: # loads best params from wandb API if provided path.
    api = wandb.Api()
    sweep = api.sweep(best_params)
    best_run = sweep.best_run()
    params = best_run.config
    if verbose: print(params)
  return params
    