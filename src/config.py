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
  "use_wandb" : True,
  "best_params": "", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>."
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "morgan_ffn_128.pickle",
  "output_name": "ffn_morgan_colSMILES_seed42", # remember to not include .csv for this particular variable, used to name the model file also
  "modes": ["test"], # can be either "train", "test" or both

  # defines model architecture
  "salt_col": "salt smiles", # matches column name in df
  "salt_encoding": "morgan", # matches column name in df
  "conts": ["mw","molality", "temperature_K"],
  "fold_list":[0,1,2,3,4,5,6,7,8,9], 
  "seed": 42,
  "device": "cuda",
  "chemberta_model_name": 'kuelumbus/polyBERT',
  "use_salt_encoder": False, # always False for now.
  "num_polymer_features": 600,
  "num_salt_features": 128,
  "num_continuous_vars": 3,
  "data_fraction": 1, # use something small like 0.01 if you want to do quick run for error checking

  # tunable hyperparameters
  "batch_size": 16, # cannot exceed 32 atm due to memory limits
  "accumulation_steps":8,
  "hidden_size": 1024,
  "num_hidden_layers": 1,
  "dropout": 0.1,
  "activation_fn": "relu",
  "init_method": "glorot",
  "output_size": 1,
  "freeze_layers": 12, # by default 12 layers in polyBERT
  "encoder_init_lr" : 1e-6,
  "lr": 1e-4,
  'optimizer': "AdamW",
  "scheduler": "ReduceLROnPlateau", # {"ReduceLROnPlateau", "LinearLR"}
  'warmup_steps': 0,
  "epochs": 20,
  }

ffn_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "morgan_ffn_128.pickle",
  "output_name": "ffn_morgan_hpsweep_other_params",
  "fold": 0, # the fold index
  "rounds": 40,
  "seed":42, 
  'sweep_id': '', # to resume a sweep after stopping
  "sweep_config":{
    "method": "bayes", # try grid or random or bayes
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
            'value': 1.0
        },
        'batch_size': {
            'value': 16
        },
        'accumulation_steps': {
            'value': 2
        },
        'hidden_size': {
            'values': [512,1024,2048]
        },
        'num_hidden_layers': {
            'value': 1
        },
        'dropout': {
            'values': [0.1,0.2,0.3]
        },
        'activation_fn': {
            'values': ['relu','leaky_relu','prelu','selu','elu']
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
            'values': [5e-4, 1e-3]
        },
        'optimizer': {
            'value': 'AdamW'
        },
        'scheduler': {
            'value': 'ReduceLROnPlateau'
        },
        'warmup_steps': {
            'values': 0
        },
        'epochs': {
            'value': 15
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