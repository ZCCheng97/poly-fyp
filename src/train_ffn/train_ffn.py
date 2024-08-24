from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import os

from .utils import logger, initialize_scheduler, initialize_optimizer
from .engine_ffn import Engine
from .model import FFNModel
from .dataset import FFNDataset

def train_ffn(tabularsplit,args, trained_model_path, log_csv_path, save = True) -> float:
    train_dataset = FFNDataset(tabularsplit.x_train,tabularsplit.y_train,args)
    val_dataset = FFNDataset(tabularsplit.x_val,tabularsplit.y_val,args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    num_training_steps = (len(train_loader)//args.batch_size)*(args.epochs//args.accumulation_steps)
    
    model = FFNModel(poly_model_name= args.poly_model_name, 
                     salt_model_name=args.salt_model_name, 
                     num_polymer_features=args.num_polymer_features, 
                     num_salt_features=args.num_salt_features, 
                     num_continuous_vars=args.num_continuous_vars,
                     hidden_size=args.hidden_size, 
                     num_hidden_layers=args.num_hidden_layers, 
                     dropout=args.dropout, 
                     activation_fn=args.activation_fn, 
                     init_method=args.init_method, 
                     output_size=args.output_size,
                     freeze_layers=args.freeze_layers,
                     salt_freeze_layers=args.salt_freeze_layers)
    model.to(args.device)
  
    criterion = nn.L1Loss()
    optimizer = initialize_optimizer(args, model) # Fully connected layers (FFN)
    scheduler = initialize_scheduler(args,optimizer,num_training_steps=num_training_steps)

    elapsed_epochs = 0
    current_best_loss = np.inf
    if save and os.path.exists(trained_model_path):
        print(f"Model found at location: {trained_model_path}")
        print("Loading from checkpoint...")
        cp = torch.load(trained_model_path, map_location=args.device)
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        scheduler.load_state_dict(cp["scheduler_state_dict"])
        elapsed_epochs = cp["epoch"]
        current_best_loss = cp["current_best_loss"]
        best_state = cp["model_state_dict"]
    engine = Engine(model, criterion, optimizer, args.device, args.accumulation_steps, args.arrhenius,args.regularisation)

    for epoch in tqdm(range(args.epochs-elapsed_epochs), desc="Epoch", total=args.epochs-elapsed_epochs):
        train_loss, val_loss = engine(train_loader, val_loader)
        if val_loss < current_best_loss:
            current_best_loss = val_loss
            best_state = model.state_dict()

        if save: torch.save({"epoch":epoch+1+elapsed_epochs,
                    "model_state_dict":best_state,
                    "optimizer_state_dict":optimizer.state_dict(),
                    "scheduler_state_dict":scheduler.state_dict(),
                    "current_best_loss":current_best_loss}, trained_model_path)
            
        scheduler.step(val_loss)
        data_dict = {"Epoch": epoch+1, "Train_loss": train_loss, "Valid_loss": val_loss} 
        
        logger(data_dict, log_csv_path, args.use_wandb)
    return current_best_loss