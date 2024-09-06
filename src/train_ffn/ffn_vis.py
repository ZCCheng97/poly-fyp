# for now, implement just one run using one initialisation seed and one fold

import pickle
from pathlib import Path
import torch
import os
from captum.attr import LayerIntegratedGradients, visualization
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from .model import FFNModel
from .dataset import FFNDataset
from .test_ffn import visualise 

def ffn_vis(args):
    script_dir = Path(__file__).resolve().parent

    data_dir = script_dir.parent.parent / args.data_dir_name
    results_dir = script_dir.parent.parent / args.results_dir_name
    models_dir = script_dir.parent.parent / args.models_dir_name
    input_data_path = data_dir / args.input_data_name

    with open(input_data_path, 'rb') as f:
        data = pickle.load(f) # object of list of TabularSplit Classes

    torch.cuda.empty_cache()
    output_log_image_path = results_dir / f"{args.output_name}_fold{args.fold}_sample{args.start_idx}_to_{args.end_idx}.html"
    output_model_path = models_dir / f"{args.output_name}_fold{args.fold}.pt"
    print(f"Currently running fold: {args.fold}")

    datasplit = data[args.fold]
    # Convert token ids to tokens
    tokeniser = AutoTokenizer.from_pretrained(args.poly_model_name)
    vis_data_record = visualise(datasplit, args, output_model_path, tokeniser)

    # Visualize and save as html
    data = visualization.visualize_text(vis_data_record)
    with open(output_log_image_path, "w", encoding="utf-8") as file:
        file.write(data._repr_html_())