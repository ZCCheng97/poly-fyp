import random
import numpy as np
import torch
import csv
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def save_results(path, df, labels, preds):
    df = df.copy()

    # Add these as new columns to the dataframe
    df['actual_conductivity'] = labels
    df['predicted_conductivity'] = preds

    # Save the dataframe as a .csv file
    df.to_csv(path, index=False)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def arrhenius_score(outputs, temperatures):
    lnA, Ea = outputs[:,0], outputs[:,1]
    R = 8.63e-5
    e = np.exp(1)
    
    lnC = lnA - Ea/(R*temperatures)
    conductivity = lnC * np.log10(e)
    return conductivity.unsqueeze(1) 

def initialize_optimizer(args, model):
    optimizer_params = list()
    if args.optimizer == "AdamW":
       optimizer_params.append({'params': model.ffn.parameters(), 'lr': args.lr})

    if args.poly_model_name:
        optimizer_params.extend([
        {'params': model.polymerencoder.model.embeddings.parameters(), 'lr': args.encoder_init_lr}, 
        {'params': model.polymerencoder.model.encoder.layer[:4].parameters(), 'lr': args.encoder_init_lr}, 
        {'params': model.polymerencoder.model.encoder.layer[4:8].parameters(), 'lr': args.encoder_init_lr*1.75},  # Lower layers of polymerencoder
        {'params': model.polymerencoder.model.encoder.layer[8:].parameters(), 'lr': args.encoder_init_lr*3.5} # Upper layers of polymerencoder
        ])  
    if args.salt_model_name:
        optimizer_params.extend([
        {'params': model.saltencoder.model.embeddings.parameters(), 'lr': args.encoder_init_lr}, 
        {'params': model.saltencoder.model.encoder.layer[:4].parameters(), 'lr': args.encoder_init_lr}, 
        {'params': model.saltencoder.model.encoder.layer[4:8].parameters(), 'lr': args.encoder_init_lr*1.75},  # Lower layers of saltencoder
        {'params': model.saltencoder.model.encoder.layer[8:].parameters(), 'lr': args.encoder_init_lr*3.5} # Upper layers of saltencoder
        ])  

    return torch.optim.AdamW(optimizer_params)

def initialize_scheduler(args,optimizer,num_training_steps):
    if args.scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    elif args.scheduler == "LinearLR":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    elif args.scheduler == "CosineLR":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

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

def load_best_params(best_params:str="") -> dict:
  params = dict()
  if best_params: # loads best params from wandb API if provided path.
    api = wandb.Api()
    sweep = api.sweep(best_params)
    best_run = sweep.best_run()
    params = best_run.config
  return get_args(params)


def plot_attributions(vis_data, save_path=None):
    characters = vis_data[0][0]
    attributions1 = np.array(vis_data[0][1])
    attributions2 = np.array(vis_data[1][1])
    
    # Create the figure and subplots for two heatmaps (one for each attribution array)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2), gridspec_kw={'height_ratios': [1, 1]})  # Two rows, one column
    
    # Create a colormap that ranges from red to green with a midpoint at 0
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'red_white_green', 
        [(0, 'red'), (0.5, 'white'), (1, 'green')], 
        N=256
    )
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # Plot the first heatmap (for attributions1)
    heatmap1 = ax1.imshow(attributions1[np.newaxis, :], cmap=cmap, norm=norm, aspect='auto')
    ax1.set_xticks([])  # Remove x-tick labels from the first heatmap
    ax1.get_yaxis().set_visible(False)

    # Plot the second heatmap (for attributions2)
    heatmap2 = ax2.imshow(attributions2[np.newaxis, :], cmap=cmap, norm=norm, aspect='auto')
    ax2.set_xticks(np.arange(len(characters)))
    ax2.set_xticklabels(characters)  # Set the character labels only for the bottom plot
    ax2.get_yaxis().set_visible(False)

    # Add a single colorbar to the side of the figure (shared by both heatmaps)
    cbar = fig.colorbar(heatmap2, ax=[ax1, ax2], location = "bottom", pad = 0.2, shrink = 0.5)
    cbar.set_label('Attribution Score')

    ax1.annotate(
        "Fine-Tuned",
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    ax2.annotate(
        "Pre-Trained",
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    fig.suptitle("Atom Importance", y = 1,fontsize=14)
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Return the figure object to the calling script
    return fig

class Arguments:
    def __init__(self, d):
        self.as_dictionary = d
        for key, value in self.as_dictionary.items():
            setattr(self, key, value)
        
def get_args(d):
    return Arguments(d)