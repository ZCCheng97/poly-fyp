from .utils import * 
from pathlib import Path
import pandas as pd

def data_cleaning(args):
  script_dir = Path(__file__).resolve().parent
  # Construct the path to the data directory
  data_dir = script_dir.parent.parent / args.data_dir_name 
  csv_dir = data_dir / args.data_name
  output_file_name = data_dir / args.cleaned_data_name

  df = pd.read_csv(csv_dir)
  new_df = df \
  .pipe(add_long_smiles)\
  .pipe(add_raw_psmiles)\
  .pipe(add_psmiles)\
  .pipe(add_monomer_smiles)\
  .pipe(fill_salt_with_Li)\
  .pipe(fill_molality)\
  .pipe(fill_mw)\
  .pipe(add_temperature_K)
  
  new_df = new_df.drop_duplicates()
  new_df.to_excel(output_file_name,index=False)
  
  print(f"Cleaning completed. Sample size of {len(new_df)}")
  return new_df