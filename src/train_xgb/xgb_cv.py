import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np
import wandb

from .utils import xgb, logger, load_best_params, save_results

def xgb_cv(args):
  script_dir = Path(__file__).resolve().parent

  data_dir = script_dir.parent.parent / args.data_dir_name
  results_dir = script_dir.parent.parent / args.results_dir_name
  input_data_path = data_dir / args.input_data_name
  output_path = results_dir / f"{args.output_name}.csv"
  

  with open(input_data_path, 'rb') as f:
     data = pickle.load(f) # object of list of Datasplit Classes
  
  nfolds = len(data)
  fold_means = list()

  for fold in tqdm(range(nfolds)):
    print(f"Currently running fold: {fold}")
    output_data_path = results_dir / f"{args.output_name}_fold{fold}_data.csv"  
    if args.use_wandb: 
      project = args.output_name.split(".")[0]

      wandb.init(
            project=project, 
            name=f"Fold {fold}", 
            config=args.as_dictionary)  

    datasplit = data[fold] # object of DataSplit class.
    params = load_best_params(args.best_params, verbose = args.verbose)

    res_mean,res_std, y_pred = xgb(datasplit, seed_list = args.seed_list, verbose = args.verbose, params=params)
    data_dict = {
        "fold": fold,
        "mae_mean": res_mean[0],
        "mse_mean": res_mean[1],
        "spearman_mean": res_mean[2],
        "r2_mean": res_mean[3],
        "mae_std": res_std[0],
        "mse_std": res_std[1],
        "spearman_std": res_std[2],
        "r2_std": res_std[3],
        "train_len": len(datasplit.x_train),
        "val_len": len(datasplit.x_val),
        "test_len": len(datasplit.x_test),
        "train_labels": datasplit.label_counts[0],
        "val_labels": datasplit.label_counts[1],
        "test_labels": datasplit.label_counts[2]
    }
    logger(data_dict, output_path, args.use_wandb)
    save_results(output_data_path, datasplit.x_test, datasplit.y_test, y_pred)
    fold_means.append(res_mean)
    if args.use_wandb: wandb.finish()

  mean_of_fold_means = np.mean(np.array(fold_means),axis = 0)
  stds_of_fold_means = np.std(np.array(fold_means),axis = 0)

  mean_dict = {
        "fold": "Overall Mean",
        "mae_mean": mean_of_fold_means[0],
        "mse_mean": mean_of_fold_means[1],
        "spearman_mean": mean_of_fold_means[2],
        "r2_mean": mean_of_fold_means[3]}
  std_dict = {
        "fold": "Overall Std",
        "mae_mean": stds_of_fold_means[0],
        "mse_mean": stds_of_fold_means[1],
        "spearman_mean": stds_of_fold_means[2],
        "r2_mean": stds_of_fold_means[3]}
  
  logger(mean_dict, output_path, use_wandb=False)
  logger(std_dict, output_path, use_wandb=False)

  
