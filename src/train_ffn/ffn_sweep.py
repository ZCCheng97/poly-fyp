import pickle
from pathlib import Path
import numpy as np
import wandb
import torch

from .utils import seed_everything
from .train_ffn import train_ffn
from .test_ffn import test_ffn

def ffn_sweep(args):
  script_dir = Path(__file__).resolve().parent

  data_dir = script_dir.parent.parent / args.data_dir_name
  results_dir = script_dir.parent.parent / args.results_dir_name
  models_dir = script_dir.parent.parent / args.models_dir_name
  input_data_path = data_dir / args.input_data_name

  project = args.output_name.split(".")[0]
  sweep_id = wandb.sweep(args.sweep_config, project=project)
  fold = args.fold

  def tune():
      with open(input_data_path, 'rb') as f:
            data = pickle.load(f) # object of list of Datasplit Classes
      torch.cuda.empty_cache()
      seed_everything(args.seed)
      output_log_path = results_dir / f"{args.output_name}_fold{fold}.csv"
      output_model_path = models_dir / f"{args.output_name}_fold{fold}.pt"
      
      wandb.init(config=args.as_dictionary)
      wandb_config = wandb.config
      print(f"Currently running fold: {fold}")

      datasplit = data[fold] # object of DataSplit class.

      train_res = train_ffn(datasplit, args, output_model_path,output_log_path)
      test_scores = test_ffn(datasplit,args, output_model_path)

      mean_dict = {
        "mae_mean": test_scores[0],
        "mse_mean": test_scores[1],
        "spearman_mean": test_scores[2],
        "r2_mean": test_scores[3]}
      wandb.log({**mean_dict})

  wandb.agent(sweep_id, tune, count=args.epochs)

  
