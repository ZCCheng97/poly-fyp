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
  api = wandb.Api()
  entity  = api.viewer.entity
  sweep_id = f"{entity}/{project}/{args.sweep_id}" if args.sweep_id else wandb.sweep(args.sweep_config,project=project)
  fold = args.fold
  
  def tune():
      with open(input_data_path, 'rb') as f:
            data = pickle.load(f) # object of list of Datasplit Classes
      torch.cuda.empty_cache()
      
      
      wandb.init(config=args.as_dictionary)
      config = wandb.config
      seed_everything(args.seed)
      output_log_path = results_dir / f"{args.output_name}_fold{fold}.csv"
      output_model_path = models_dir / f"{args.output_name}_fold{fold}.pt"
      
      print(f"Currently running fold: {fold}")

      datasplit = data[fold] # object of DataSplit class.

      train_res = train_ffn(datasplit, config, output_model_path,output_log_path,save = False)

      mean_dict = {
        "mae_mean": train_res}
      wandb.log(mean_dict)

  wandb.agent(sweep_id, tune, count=args.rounds)

  
