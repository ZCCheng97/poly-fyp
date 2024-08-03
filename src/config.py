data_cleaning = {
    "data_dir_name": "data",
    "data_name": "clean_train_data.csv",
    "cleaned_data_name": "cleaned_data.xlsx" # should be saved as a xlsx file
}

preprocessing = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "morgan_xgb_128.pickle",
  "label_col": "long_smiles",
  "cats": ["long_smiles","salt smiles"],
  "conts": ["mw","molality", "temperature_K"],
  "drop_columns": ["smiles","raw_psmiles","psmiles","temperature"],
  "train_ratio":0.8,
  "val_ratio":0.1,
  "nfolds": 10,
  "polymer_use_fp": "morgan", # {"polybert", "morgan", "none"}
  "salt_use_fp": "morgan", # {"morgan", chemprop?}
  "fpSize": 128,
  "verbose":True
}

xgb_cv = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "polyBERT_xgb_128.pickle",
  "output_name": "xgb_polyBERT.csv",
  "seed_list":[42,3,34,43,83], 
  "verbose": True
}






step_args = {
       'data_cleaning': data_cleaning,
       'preprocessing': preprocessing,
       "xgb_cv": xgb_cv}