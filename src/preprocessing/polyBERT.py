import torch

from .fingerprinting import array_to_cols

def smiles_to_polyBERT(df,col_name, tokeniser, model):
  df = df.copy()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  model.to(device)

  def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  def get_embeddings(texts):
    encoded_input = tokeniser(texts,max_length = 512,return_tensors='pt', padding=True, truncation=True)
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    fingerprints = mean_pooling(outputs, encoded_input['attention_mask'])
    return fingerprints

  unique_psmiles = list(df[col_name].unique())
  unique_embeddings = get_embeddings(unique_psmiles)
  embedding_dict = {desc: embedding.cpu().numpy() for desc, embedding in zip(unique_psmiles, unique_embeddings)}
  df['polyBERT_fp_'+col_name] = df[col_name].map(embedding_dict)
  df = array_to_cols(df, "polyBERT_fp", col_name, fpSize = 600)
  return df