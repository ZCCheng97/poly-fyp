from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import os

from .engine_ffn import Tester
from .model import FFNModel
from .dataset import FFNDataset

def test_ffn(tabularsplit, args, trained_model_path) -> float:

    test_dataset = FFNDataset(tabularsplit.x_test,tabularsplit.y_test,args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False)
    
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

    if not os.path.exists(trained_model_path):
        print("Model not found!")
        return None
    print(f"Model found at location: {trained_model_path}")
    print("Loading from checkpoint...")
    cp = torch.load(trained_model_path, map_location=args.device)
    model.load_state_dict(cp["model_state_dict"])

    engine = Tester(model, args.device, args.arrhenius)
    scores, labels, preds = engine(test_loader)
    

    return scores,test_dataset.indices,labels,preds