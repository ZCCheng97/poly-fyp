import torch
import torch.nn as nn
from transformers import AutoModel

class FFNModel(nn.Module):
    def __init__(self, poly_model_name, salt_model_name, num_polymer_features,num_salt_features, num_continuous_vars, 
                 hidden_size = 2048, num_hidden_layers=2, dropout = 0.1, activation_fn="relu", init_method="glorot", 
                 output_size = 1,freeze_layers=12, salt_freeze_layers = 12):
        super(FFNModel, self).__init__()
        if poly_model_name:
            self.polymerencoder = TransformerEncoder(model_name=poly_model_name, freeze_layers=freeze_layers)
        else:
            self.polymerencoder = None

        if salt_model_name:
            self.saltencoder = TransformerEncoder(model_name=salt_model_name, freeze_layers=salt_freeze_layers)
        else:
            self.saltencoder = None

        self.num_continuous_vars = num_continuous_vars
        self.input_dim  = num_polymer_features + num_salt_features + num_continuous_vars
        self.activation_fn = pick_activation(activation_fn)
        self.init_method = init_method

        self.create_ffn(self.input_dim, num_hidden_layers, hidden_size, self.activation_fn, dropout, output_size)

    def create_ffn(self, input_dim, num_hidden_layers, hidden_size, activation_fn, dropout, output_size):
        dropout = nn.Dropout(dropout)
        if num_hidden_layers == 0:
            ffn = [
                dropout,
                nn.Linear(input_dim, output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(input_dim, hidden_size)
            ]
            for _ in range(num_hidden_layers-1):
                ffn.extend([
                    activation_fn,
                    dropout,
                    nn.Linear(hidden_size, hidden_size),
                ])
            ffn.extend([
                activation_fn,
                dropout,
                nn.Linear(hidden_size,output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        # Initialize weights if an initialization method is provided
        initialize_weights(self.ffn, init_method=self.init_method)
     
    def forward(self, poly_input, poly_attention_mask, salt_input, continuous_vars):
        if self.polymerencoder:
            polymer_embedding = self.polymerencoder(input_ids=poly_input, attention_mask = poly_attention_mask)
        else:
            polymer_embedding = poly_input
        
        if self.saltencoder:
            salt_input_ids, salt_attention_mask = salt_input["input_ids"].squeeze(1), salt_input["attention_mask"].squeeze(1)
            salt_embedding = self.saltencoder(input_ids=salt_input_ids, attention_mask = salt_attention_mask)
        else:
            salt_embedding = salt_input

        continuous_vars = continuous_vars[:,:self.num_continuous_vars]
        combined_embedding = torch.cat((polymer_embedding, salt_embedding, continuous_vars), dim=1)
        ffn_output = self.ffn(combined_embedding)

        return ffn_output

class TransformerEncoder(nn.Module):
    def __init__(self, model_name, freeze_layers=12):
        super(TransformerEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        if freeze_layers:
            self.freeze_model_layers(self.model, freeze_layers)
        
    def freeze_model_layers(self, model, layers_to_freeze):
        for name, param in model.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
            if "layer" in name:
                layer_idx = int(name.split('.')[2])  #named as polymerencoder.model.encoder.layer.0... etc."
                if layer_idx < layers_to_freeze:
                    param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        return pooled_output

def initialize_weights(model: nn.Module, init_method:str = "glorot") -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    intialisations_d = {"glorot": nn.init.xavier_normal_}
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            intialisations_d[init_method](param)

def pick_activation(activation_fn:str = "relu"):
    activations_d = {"relu": nn.ReLU(),
                     'leaky_relu': nn.LeakyReLU(0.1),
                     'prelu': nn.PReLU(),
                     'selu': nn.SELU(),
                     'elu': nn.ELU()}
    return activations_d[activation_fn]