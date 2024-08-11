# for now, implement just one run using one initialisation seed and one fold

import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np
import wandb
import torch

from .train_ffn import train_ffn
from .test_ffn import test_ffn
from .utils import seed_everything, logger, load_best_params

def ffn_cv(args):
    script_dir = Path(__file__).resolve().parent

    data_dir = script_dir.parent.parent / args.data_dir_name
    results_dir = script_dir.parent.parent / args.results_dir_name
    models_dir = script_dir.parent.parent / args.models_dir_name
    input_data_path = data_dir / args.input_data_name

    with open(input_data_path, 'rb') as f:
        data = pickle.load(f) # object of list of TabularSplit Classes

    torch.cuda.empty_cache()
    seed_everything(args.seed)
    for fold in tqdm(args.fold_list):
        output_log_path = results_dir / f"{args.output_name}_fold{fold}.csv"
        output_log_test_path = results_dir / f"{args.output_name}_testscores.csv"
        output_model_path = models_dir / f"{args.output_name}_fold{fold}.pt"
        print(f"Currently running fold: {fold}")

        datasplit = data[fold]
        params = load_best_params(args.best_params) if args.best_params else args
        if "train" in args.modes:
            if args.use_wandb: 
                project = args.output_name.split(".")[0]

                wandb.init(
                        project=project, 
                        name=f"Fold {fold} Train", 
                        config=params.as_dictionary)  

             # object of TabularSplit class.
            train_res = train_ffn(datasplit, params, output_model_path,output_log_path)

            if args.use_wandb: wandb.finish()

        if "test" in args.modes:
            if args.use_wandb: 
                project = args.output_name.split(".")[0]

                wandb.init(
                        project=project, 
                        name=f"Fold {fold} Test", 
                        config=params.as_dictionary) 

            test_scores = test_ffn(datasplit,params, output_model_path)
            data_dict = {
            "fold": fold,
            "mae_mean": test_scores[0],
            "mse_mean": test_scores[1],
            "spearman_mean": test_scores[2],
            "r2_mean": test_scores[3],
            "train_len": len(datasplit.x_train),
            "val_len": len(datasplit.x_val),
            "test_len": len(datasplit.x_test),
            "train_labels": datasplit.label_counts[0],
            "val_labels": datasplit.label_counts[1],
            "test_labels": datasplit.label_counts[2]}

            logger(data_dict, output_log_test_path, args.use_wandb)
            if args.use_wandb: wandb.finish()
    