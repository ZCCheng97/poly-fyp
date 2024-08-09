import random
import numpy as np
import torch
import csv
import wandb
from transformers import get_linear_schedule_with_warmup

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def initialize_optimizer(args, model):
    if args.optimizer == "AdamW":
        return torch.optim.AdamW([
    {'params': model.polymerencoder.model.encoder.layer[:6].parameters(), 'lr': args.encoder_init_lr},  # Lower layers of polyencoder
    {'params': model.polymerencoder.model.encoder.layer[6:].parameters(), 'lr': args.encoder_init_lr*1.75},  # Upper layers of polyencoder
    {'params': model.ffn.parameters(), 'lr': args.lr} ])

def initialize_scheduler(args,optimizer,num_training_steps):
    if args.scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    elif args.scheduler == "LinearLR":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

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