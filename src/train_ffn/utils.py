import random
import numpy as np
import torch
import csv
import wandb
import pandas as pd
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

class Arguments:
    def __init__(self, d):
        self.as_dictionary = d
        for key, value in self.as_dictionary.items():
            setattr(self, key, value)
        
def get_args(d):
    return Arguments(d)