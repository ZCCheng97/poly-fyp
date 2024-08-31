import pickle
from pathlib import Path
import pandas as pd

from .dataset import TorchDataset

def preprocess_ffn(args):
  script_dir = Path(__file__).resolve().parent

  data_dir = script_dir.parent.parent / args.data_dir_name
  input_data_path = data_dir / args.input_data_name
  output_data_path = data_dir / args.output_data_name

  df = pd.read_excel(input_data_path)
  df["psmiles_split"] = df.psmiles
  torch_dataset = TorchDataset(df).process(args)

  with open(output_data_path, 'wb') as f:
    pickle.dump(torch_dataset, f)

  return torch_dataset