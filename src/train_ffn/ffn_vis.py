# for now, implement just one run using one initialisation seed and one fold

import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import os
from captum.attr import LayerIntegratedGradients, visualization
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from .model import FFNModel
from .dataset import FFNDataset
from .utils import seed_everything, logger, save_results

def ffn_vis(args):
    script_dir = Path(__file__).resolve().parent

    data_dir = script_dir.parent.parent / args.data_dir_name
    results_dir = script_dir.parent.parent / args.results_dir_name
    models_dir = script_dir.parent.parent / args.models_dir_name
    input_data_path = data_dir / args.input_data_name

    with open(input_data_path, 'rb') as f:
        data = pickle.load(f) # object of list of TabularSplit Classes

    torch.cuda.empty_cache()
    output_log_image_path = results_dir / f"{args.output_name}_fold{args.fold}_sample{args.sample_idx}.html"
    output_model_path = models_dir / f"{args.output_name}_fold{args.fold}.pt"
    print(f"Currently running fold: {args.fold}")

    datasplit = data[args.fold]

    test_dataset = FFNDataset(datasplit.x_test,datasplit.y_test,args)
    test_sample = test_dataset[args.sample_idx]
    
    model = FFNModel(poly_model_name= args.poly_model_name, 
                     salt_model_name=args.salt_model_name, 
                     num_polymer_features=args.num_polymer_features, 
                     num_salt_features=args.num_salt_features, 
                     num_continuous_vars=args.num_continuous_vars,
                     hidden_size=args.hidden_size, 
                     num_hidden_layers=args.num_hidden_layers, 
                     activation_fn=args.activation_fn, 
                     init_method=args.init_method, 
                     output_size=args.output_size)
    model.to(args.device)

    if not os.path.exists(output_model_path):
        print("Model not found!")
        return None
    print(f"Model found at location: {output_model_path}")
    print("Loading from checkpoint...")
    cp = torch.load(output_model_path, map_location=args.device)
    model.load_state_dict(cp["model_state_dict"])
    
    text_input = test_sample['poly_inputs']
    salt_input = test_sample['salt_inputs'].unsqueeze(0)
    continuous_vars = test_sample['continuous_vars'].unsqueeze(0)

    # Move tensors to device
    text_input = text_input.to(args.device) 
    salt_input = salt_input.to(args.device)
    continuous_vars = continuous_vars.to(args.device)    
    lig = LayerIntegratedGradients(model, model.polymerencoder.model.embeddings)
    # Compute attributions based on the use_pretrained_embeddings flag
    attributions, delta = lig.attribute(
        inputs=text_input["input_ids"],  # This assumes use_pretrained_embeddings=True
        additional_forward_args=(text_input["attention_mask"], salt_input, continuous_vars),
        return_convergence_delta=True
    )

    # Sum the attributions over embedding dimensions
    attributions_sum = attributions.sum(dim=-1).squeeze(0)

    # Convert token ids to tokens
    tokeniser = AutoTokenizer.from_pretrained(args.poly_model_name)
    tokens = tokeniser.convert_ids_to_tokens(text_input['input_ids'].squeeze(0).cpu())

    # Create the visualization data record
    vis_data_record = visualization.VisualizationDataRecord(
        attributions_sum, 
        model(text_input["input_ids"],text_input["attention_mask"], salt_input, continuous_vars).item(),
        0, 0, 0, 0, 
        tokens, 
        convergence_score=delta.cpu()
    )

    # Visualize and save as html
    
    data = visualization.visualize_text([vis_data_record])
    with open(output_log_image_path, "w", encoding="utf-8") as file:
        file.write(data._repr_html_())