data_cleaning = {
    "data_dir_name": "data",
    "data_name": "clean_train_data.csv",
    "cleaned_data_name": "cleaned_data.xlsx" # should be saved as a xlsx file
}

preprocess_xgb = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "morgan_xgb_monomerSMILES.pickle",
  "cats": ["monomer_smiles","salt smiles"], # psmiles for polyBERT, long_smiles for morgan
  "conts": ["mw","molality", "temperature_K"],
  "drop_columns": ["raw_psmiles","long_smiles","temperature","psmiles"],
  "train_ratio":0.8,
  "val_ratio":0.1,
  "nfolds": 10,
  "polymer_use_fp": "morgan_monomer", # {"polybert", "morgan", "morgan_monomer","none"}
  "salt_use_fp": "morgan", # {"morgan", chemprop?}
  "fpSize": 128,
  "verbose":False
}

preprocess_ffn = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "morgan_ffn_128.pickle",
  "train_ratio":0.8,
  "val_ratio":0.1,
  "nfolds": 10,
  "text_col": "psmiles",
  "salt_col": "salt smiles",
  "conts": ["mw","molality", "temperature_K"],
  "transformer_name": 'kuelumbus/polyBERT',
  "salt_encoding": "morgan", # {"morgan", chemprop?}
  "fpSize": 128,
  "verbose":False
}

xgb_cv = {
  "use_wandb" : True,
  "best_params": "zccheng97-nanyang-technological-university-singapore/xgb_morgan_monomerSMILES_hpsweep/1e52oiaj", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>"
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "morgan_xgb_monomerSMILES.pickle",
  "output_name": "xgb_morgan_monomerSMILES_seed42_best.csv",
  # "seed_list":[42,3,34,43,83], 
  "seed_list":[42], 
  "verbose": True,
}

xgb_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "morgan_xgb_monomerSMILES.pickle",
  "output_name": "xgb_morgan_monomerSMILES_hpsweep.csv",
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
  "use_wandb" : False,
  "best_params": "", 
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "morgan_ffn_128.pickle",
  "output_name": "dummy", # remember to not include .csv for this particular variable, used to name the model file also
  "modes": ["train"], # can be either "train", "test" or both

  # defines model architecture
  "salt_col": "salt smiles", # matches column name in df
  "salt_encoding": "morgan", # matches column name in df
  "conts": ["mw","molality", "temperature_K"],
  "fold_list":[0], 
  "seed": 42,
  "device": "cuda",
  "chemberta_model_name": 'kuelumbus/polyBERT',
  "use_salt_encoder": False, # always False for now.
  "num_polymer_features": 600,
  "num_salt_features": 128,
  "num_continuous_vars": 3,
  "data_fraction": 0.01, # use something small like 0.01 if you want to do quick run for error checking

  # tunable hyperparameters
  "batch_size": 16, # cannot exceed 32 atm due to memory limits
  "accumulation_steps":4,
  "hidden_size": 1024,
  "num_hidden_layers": 2,
  "dropout": 0.1,
  "activation_fn": "relu",
  "init_method": "glorot",
  "output_size": 1,
  "freeze_layers": 12, # by default 12 layers in polyBERT
  "encoder_init_lr" : 1e-6,
  "lr": 1e-4,
  'optimizer': "AdamW",
  "scheduler": "ReduceLROnPlateau", # {"ReduceLROnPlateau", "LinearLR"}
  "epochs": 20,
  
  }

ffn_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "morgan_ffn_128.pickle",
  "output_name": "ffn_morgan_hpsweep_data_size",
  "fold": 0, # the fold index
  "rounds": 5,
  "seed":42, 
  'sweep_id': '', # to resume a sweep after stopping
  "sweep_config":{
    "method": "grid", # try grid or random or bayes
    "metric": {
      "name": "mae_mean",
      "goal": "minimize"   
    },
    "parameters": {
        'use_wandb' : {
            'value': False # do not change this value. Passed to train_ffn so it does not use wandb
        },
        "salt_col": {
            "value": "salt smiles"
        },
        "salt_encoding": {
            "value": "morgan"
        },
        "conts": {
            "value": ["mw","molality", "temperature_K"]
        },
        'device' : {
            'value': 'cuda'
        },
        'chemberta_model_name': {
            'value': 'kuelumbus/polyBERT'
        },
        'use_salt_encoder': {
            'value': False
        },
        'num_polymer_features': {
            'value': 600
        },
        'num_salt_features': {
            'value': 128
        },
        'num_continuous_vars': {
            'value': 3
        },
        'data_fraction': {
            'values': [0.01,0.05,0.1,0.5,1.0]
        },
        'batch_size': {
            'value': 16
        },
        'accumulation_steps': {
            'values': 2
        },
        'hidden_size': {
            'value': 1024
        },
        'num_hidden_layers': {
            'value': 1
        },
        'dropout': {
            'value': 0.1
        },
        'activation_fn': {
            'value': 'relu'
        },
        'init_method': {
            'value': 'glorot'
        },
        'freeze_layers': {
            'value': 12
        },
        'output_size': {
            'value': 1
        },
        'encoder_init_lr': {
            'value': 1e-6
        },
        'lr': {
            'value': 5e-4
        },
        'optimizer': {
            'value': 'AdamW'
        },
        'scheduler': {
            'value': 'ReduceLROnPlateau'
        },
        'epochs': {
            'value': 20
        },
    }
},
  "verbose": False,
}

step_args = {
       'data_cleaning': data_cleaning,
       'preprocess_xgb': preprocess_xgb,
       'preprocess_ffn':preprocess_ffn,
       "xgb_cv": xgb_cv,
       "xgb_sweep": xgb_sweep,
       "ffn_cv": ffn_cv,
       "ffn_sweep": ffn_sweep}