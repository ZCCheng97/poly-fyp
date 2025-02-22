import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np
import wandb

from .utils import xgb

def xgb_sweep(args):
  script_dir = Path(__file__).resolve().parent

  data_dir = script_dir.parent.parent / args.data_dir_name
  input_data_path = data_dir / args.input_data_name

  project = args.output_name.split(".")[0]
  api = wandb.Api()
  entity  = api.viewer.entity
  sweep_id = f"{entity}/{project}/{args.sweep_id}" if args.sweep_id else wandb.sweep(args.sweep_config,project=project)
  
  def tune():
      with open(input_data_path, 'rb') as f:
            data = pickle.load(f) # object of list of Datasplit Classes
  
      nfolds = len(data)
      fold_means = list()
      
      wandb.init(config=args.as_dictionary)
      wandb_config = wandb.config

      for fold in tqdm(range(nfolds)):
            print(f"Currently running fold: {fold}")

            datasplit = data[fold] # object of DataSplit class.

            res_mean, _, _= xgb(datasplit, seed_list = args.seed_list, verbose = args.verbose, params = wandb_config, sweep = True)
            fold_means.append(res_mean)

      mean_of_fold_means = np.mean(np.array(fold_means),axis = 0)
      stds_of_fold_means = np.std(np.array(fold_means),axis = 0)
      mean_dict = {
        "mae_mean": mean_of_fold_means[0],
        "mse_mean": mean_of_fold_means[1],
        "spearman_mean": mean_of_fold_means[2],
        "r2_mean": mean_of_fold_means[3]}
      std_dict = {
            "mae_std": stds_of_fold_means[0],
            "mse_std": stds_of_fold_means[1],
            "spearman_std": stds_of_fold_means[2],
            "r2_std": stds_of_fold_means[3]}
      wandb.log({**mean_dict,**std_dict})

  wandb.agent(sweep_id, tune, count=args.epochs)

  
