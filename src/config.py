data_cleaning = {
    "data_dir_name": "data",
    "data_name": "clean_train_data.csv",
    "cleaned_data_name": "cleaned_data.xlsx" # should be saved as a xlsx file
}

preprocess_xgb = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "morgan_xgb_morgan_90_10_new.pickle",
  "cats": ["long_smiles","salt smiles"], # psmiles for polyBERT, long_smiles for morgan
  "conts": ["mw","molality", "temperature_K"],
  "drop_columns": ["smiles","raw_psmiles","psmiles","temperature","monomer_smiles"], # always drop "smiles"
  "train_ratio":0.9,
  "val_ratio":0.05,
  "nfolds": 10,
  "polymer_use_fp": "morgan", # {"polybert", "morgan", "morgan_monomer"}
  "salt_use_fp": "morgan", # {"morgan", "chemberta"}
  "fpSize": 128,
  "verbose":True
}

preprocess_ffn = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "polybert_ffn_morgan_90_10_new.pickle", # {poly_encoding}_{ffn/xgb}_{salt_encoding}_{arr/None}.pickle
  "train_ratio":0.9,
  "val_ratio":0.05,
  "nfolds": 10,
  "poly_encoding": "polybert_tokenizer", # {"polybert_tokenizer", "morgan"}
  "poly_model_name": 'kuelumbus/polyBERT', # {'kuelumbus/polyBERT', ''}
  "poly_col": "psmiles",
  "salt_encoding": "morgan", # {"morgan", "chemberta_tokenizer"}
  "salt_model_name": '', # {'seyonec/ChemBERTa-zinc-base-v1',''}
  "salt_col": "salt smiles",
  "conts": ["mw","molality"],
  "fpSize": 128,
  "verbose":True
}

xgb_cv = {
  "use_wandb" : False,
  "best_params": "zccheng97-nanyang-technological-university-singapore/morgan_monomer_xgb_morgan_10fold_90_10_hpsweep/fw2m7ga4", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>"
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "morgan_monomer_xgb_morgan_90_10_new.pickle",
  "output_name": "morgan_monomer_xgb_morgan_90_10_new_seed42_best", # no suffix
  # "seed_list":[42,3,34,43,83], 
  "seed_list":[42], 
  "verbose": False,
}

xgb_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "morgan_monomer_xgb_morgan_10fold_90_10.pickle",
  "output_name": "morgan_monomer_xgb_morgan_10fold_90_10_hpsweep.csv",
  "seed_list":[42], 
  'sweep_id': '', # to resume a sweep after stopping
  "params":{"n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "reg_lambda":0.01,
        'reg_alpha':0.01,
        "gamma": 0},
  "epochs": 200,
  "sweep_config":{
    "method": "bayes", # try grid or random or bayes
    "metric": {
      "name": "mae_mean",
      "goal": "minimize"   
    },
    "parameters": {
        "n_estimators": {
            "distribution":"int_uniform",
            "min": 100,
            "max":2000
        },
        "max_depth": {
            "distribution":"int_uniform",
            "min": 5,
            "max": 50
        },
        "learning_rate": {
            "distribution":"uniform",
            "min": 0.01,
            "max": 1.0
        },
        'reg_lambda':{
          "distribution":"log_uniform_values",
          "min": 1e-9,
          "max": 10.0
        },
        'reg_alpha':{
          "distribution":"log_uniform_values",
          "min": 1e-9,
          "max": 10.0
        },
        'gamma':{
          "distribution":"uniform",
          "min": 1e-9,
          "max": 10.0
        }
    }
},
  "verbose": False,

}

ffn_cv = {
  "use_wandb" : True,
  "best_params": "", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>."
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "polybert_ffn_morgan_90_10_new.pickle",
  "output_name": "polybert_ffn_morgan_90_10_new_seed42", # remember to not include .csv for this particular variable, used to name the model file also
  "modes": ["train","test"], # can be either "train", "test" or both
  "arrhenius": False,
  "regularisation": 0,

  # defines model architecture
  "salt_col": "salt smiles", # matches column name in df
  "salt_encoding": "morgan", # matches column name in df, use "chemberta_tokenizer" for encoding, "morgan" for fp, "" to omit using salt as a predictor
  "salt_model_name": '', # 'seyonec/ChemBERTa-zinc-base-v1' for Chemberta, blank if not using trained embeddings
  'poly_col': "psmiles",# matches column name in df
  "poly_encoding": "polybert_tokenizer", # matches column name in df, use "polybert_tokenizer" for encoding, "morgan" for fp
  "poly_model_name": 'kuelumbus/polyBERT', # 'kuelumbus/polyBERT' if using polyBERT, blank if not using trained embeddings
  "conts": ["mw","molality","temperature_K"], # conts that are selected for modeling, include temp_K column even if using Arrhenius
  "temperature_name": "temperature_K",
  "fold_list":[0,1,2,3,4,5,6,7,8,9], 
  "seed": 42,
  "device": "cuda",
  "num_polymer_features": 600, # 600 for polybert, 128 for morgan
  "num_salt_features": 128, # 768 for chemberta, 128 for morgan
  "num_continuous_vars": 3, # change to 2 if using Arrhenius mode, otherwise 3 cont variables
  "data_fraction": 1, # use something small like 0.01 if you want to do quick run for error checking

  # tunable hyperparameters
  "batch_size": 16, # cannot exceed 32 atm due to memory limits
  "accumulation_steps": 8,
  "hidden_size": 2048,
  "num_hidden_layers": 1,
  "dropout": 0.1,
  "activation_fn": "relu",
  "init_method": "glorot",
  "output_size": 1, # change to 2 if using Arrhenius mode, otherwise 1
  "freeze_layers": 12, # by default 12 layers in polyBERT
  "encoder_init_lr" : 1e-6, # only passed to initialise_optimiser
  "salt_freeze_layers": 12,
  "salt_encoder_init_lr": 1e-6, # only passed to initialise_optimiser
  "lr": 5e-5,
  'optimizer': "AdamW", # Use "AdamW_ratedecay_4_4_4" only if using encoders for either salt or polymer. 
  "scheduler": "ReduceLROnPlateau", # {"ReduceLROnPlateau", "LinearLR", "CosineLR"}
  'warmup_steps': 10, # Usually 6% for LinearLR, 3% of total training steps for CosineLR.
  "epochs": 25,
  }

ffn_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "polybert_ffn_morgan_arr_10fold_90_10.pickle",
  "output_name": "polybert_ffn_morgan_arr_10fold_90_10_unfrozen_sweep_accsteps_warmup",
  "fold": 0, # the fold index
  "rounds": 6,
  "seed": 42, 
  'sweep_id': '', # to resume a sweep after stopping
  "sweep_config":{
    "method": "grid", # try grid or random or bayes
    "metric": {
      "name": "mae_mean",
      "goal": "minimize"   
    },
    "parameters": {
        'use_wandb' : {
            'value': True # do not change this value. Passed to train_ffn so it does not use wandb
        },
        "arrhenius": {
            'value': True
        },
        "regularisation": {
            'value':0
        },
        "salt_col": {
            "value": "salt smiles"
        },
        "salt_encoding": {
            "value": "morgan"
        },
        "salt_model_name": {
            "value": ""
        },
        "conts": {
            "value": ["mw","molality", "temperature_K"]
        },
        "temperature_name": {
            "value": "temperature_K"
        },
        'device' : {
            'value': 'cuda'
        },
        'poly_model_name': {
            'value': 'kuelumbus/polyBERT'
        },
        'poly_encoding': {
            'value': 'polybert_tokenizer'
        },
        'poly_col': {
            'value': 'psmiles'
        },
        'num_polymer_features': {
            'value': 600
        },
        'num_salt_features': {
            'value': 128
        },
        'num_continuous_vars': {
            'value': 2
        },
        'data_fraction': {
            'value': 1
        },
        'batch_size': {
            'value': 16
        },
        'accumulation_steps': {
            'values': [4,8]
        },
        'hidden_size': {
            'value': 2048
        },
        'num_hidden_layers': {
            'value':2
        },
        'dropout': {
            'value': .1
        },
        'activation_fn': {
            'value': "relu"
        },
        'init_method': {
            'value': 'glorot'
        },
        'freeze_layers': {
            'value': 0
        },
        'encoder_init_lr': {
            'value': 1e-6
        },
        'salt_freeze_layers': {
            'value': 12
        },
        'salt_encoder_init_lr': {
            'value': 1e-6
        },
        'output_size': {
            'value': 2
        },
        'lr': {
            'value': 1e-4
        },
        'optimizer': {
            'value': "AdamW"
        },
        'scheduler': {
            'value': "CosineLR"
        },
        'warmup_steps': {
            'values': [50,100,150]
        },
        'epochs': {
            'value': 50
        },
    }
},
  "verbose": False,
}

check = {
  "data_dir_name": "data",
  "input_data_name": "polybert_ffn_morgan_90_10_new.pickle",
  "fold_list": [0,1,2,3,4,5,6,7,8,9],
  "feature_columns":["psmiles","salt smiles"]
}

step_args = {
       'data_cleaning': data_cleaning,
       'preprocess_xgb': preprocess_xgb,
       'preprocess_ffn':preprocess_ffn,
       "xgb_cv": xgb_cv,
       "xgb_sweep": xgb_sweep,
       "ffn_cv": ffn_cv,
       "ffn_sweep": ffn_sweep,
       'check': check}