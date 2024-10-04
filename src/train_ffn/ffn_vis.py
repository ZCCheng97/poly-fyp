# for now, implement just one run using one initialisation seed and one fold

import pickle
from pathlib import Path
import torch
import os
from captum.attr import visualization
from transformers import AutoTokenizer

from .test_ffn import visualise 
from .utils import plot_attributions

def ffn_vis(args):
    script_dir = Path(__file__).resolve().parent

    data_dir = script_dir.parent.parent / args.data_dir_name
    results_dir = script_dir.parent.parent / args.results_dir_name
    models_dir = script_dir.parent.parent / args.models_dir_name
    input_data_path = data_dir / args.input_data_name

    with open(input_data_path, 'rb') as f:
        data = pickle.load(f) # object of list of TabularSplit Classes

    datasplit = data[args.fold]
    # Convert token ids to tokens
    tokeniser = AutoTokenizer.from_pretrained(args.poly_model_name)
    vis_data = list()
    hidden_size = args.hidden_size
    num_hidden_layers = args.num_hidden_layers

    for idx, output_name in enumerate(args.output_names):
        torch.cuda.empty_cache()
        args.hidden_size = hidden_size[idx]
        args.num_hidden_layers = num_hidden_layers[idx]

        output_log_image_path = results_dir / f"{output_name}_fold{args.fold}_sample{args.start_idx}_to_{args.end_idx}.html"
        output_model_path = models_dir / f"{output_name}_fold{args.fold}.pt"
        print(f"Currently running fold: {args.fold}")

        vis_data_record,vis_tuple = visualise(datasplit, args, output_model_path, tokeniser)

        # Visualize and save as html
        data = visualization.visualize_text(vis_data_record)
        with open(output_log_image_path, "w", encoding="utf-8") as file:
            file.write(data._repr_html_())
        vis_data.append(vis_tuple)
    plot_attributions(vis_data, results_dir / f"{output_name}_fold{args.fold}_sample{args.start_idx}_to_{args.end_idx}.png")