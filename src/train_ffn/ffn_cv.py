# for now, implement just one run using one initialisation seed and one fold

import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np
import wandb
import torch

from .train_ffn import train_ffn
from .utils import seed_everything

def ffn_cv(args):
    script_dir = Path(__file__).resolve().parent

    data_dir = script_dir.parent.parent / args.data_dir_name
    results_dir = script_dir.parent.parent / args.results_dir_name
    models_dir = script_dir.parent.parent / args.models_dir_name
    input_data_path = data_dir / args.input_data_name

    with open(input_data_path, 'rb') as f:
        data = pickle.load(f) # object of list of Datasplit Classes

    torch.cuda.empty_cache()
    seed_everything(args.seed)
    for fold in tqdm(args.fold_list):
        output_log_path = results_dir / f"{args.output_name}_fold{fold}.csv"
        output_model_path = models_dir / f"{args.output_name}_fold{fold}.pt"
        print(f"Currently running fold: {fold}")

        if args.use_wandb: 
            project = args.output_name.split(".")[0]

            wandb.init(
                    project=project, 
                    name=f"Fold {fold}", 
                    config=args.as_dictionary)  

        datasplit = data[fold] # object of DataSplit class.
        
        res = train_ffn(datasplit, args, output_model_path,output_log_path)
    