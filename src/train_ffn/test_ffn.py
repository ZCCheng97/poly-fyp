from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import torch
import os

from .engine_ffn import Tester, Visualiser
from .model import FFNModel
from .dataset import FFNDataset

def visualise(tabularsplit, args, trained_model_path, tokeniser) -> list:

    test_dataset = FFNDataset(tabularsplit.x_test,tabularsplit.y_test,args)
    batch_sampler = RangeBatchSampler(test_dataset, args.start_idx, args.end_idx, 1)
    test_loader = DataLoader(test_dataset, batch_sampler=batch_sampler)

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

    if not os.path.exists(trained_model_path):
        print("Model not found!")
        return None
    print(f"Model found at location: {trained_model_path}")
    print("Loading from checkpoint...")
    cp = torch.load(trained_model_path, map_location=args.device)
    model.load_state_dict(cp["model_state_dict"])

    engine = Visualiser(model, args.device, tokeniser, args.arrhenius)
    vis_record,vis_tuple = engine(test_loader)
    return vis_record,vis_tuple

def test_ffn(tabularsplit, args, trained_model_path) -> tuple:

    test_dataset = FFNDataset(tabularsplit.x_test,tabularsplit.y_test,args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False)
    
    model = FFNModel(poly_model_name= args.poly_model_name, 
                     salt_model_name=args.salt_model_name, 
                     num_polymer_features=args.num_polymer_features, 
                     num_salt_features=args.num_salt_features, 
                     num_continuous_vars=args.num_continuous_vars,
                     hidden_size=args.hidden_size, 
                     num_hidden_layers=args.num_hidden_layers, 
                     batchnorm=args.batchnorm,
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

class RangeBatchSampler(BatchSampler):
    def __init__(self, data_source, start_idx, end_idx, batch_size):
        super().__init__(SequentialSampler(data_source), batch_size, drop_last=False)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.batch_size = batch_size
        self.sampler_len = self.end_idx - self.start_idx + 1

    def __iter__(self):
        # Generate indices in the desired range
        indices = list(range(self.start_idx, self.end_idx + 1))
        return iter([indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)])
