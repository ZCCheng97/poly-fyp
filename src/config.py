data_cleaning = {
    "data_dir_name": "data",
    "data_name": "clean_train_data.csv",
    "cleaned_data_name": "cleaned_data.xlsx" # should be saved as a xlsx file
}

preprocess_xgb = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "chemberta_xgb_colSMILES.pickle",
  "cats": ["psmiles","salt smiles"], # psmiles for polyBERT, long_smiles for morgan
  "conts": ["mw","molality", "temperature_K"],
  "drop_columns": ["raw_psmiles","long_smiles","temperature","monomer_smiles"],
  "train_ratio":0.8,
  "val_ratio":0.1,
  "nfolds": 10,
  "polymer_use_fp": "polybert", # {"polybert", "morgan", "morgan_monomer","none"}
  "salt_use_fp": "chemberta", # {"morgan", "chemberta"}
  "fpSize": 128,
  "verbose":False
}

preprocess_ffn = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "morgan_ffn_128_arrTemp.pickle",
  "train_ratio":0.8,
  "val_ratio":0.1,
  "nfolds": 10,
  "text_col": "psmiles",
  "salt_col": "salt smiles",
  "conts": ["mw","molality"],
  "transformer_name": 'kuelumbus/polyBERT',
  "salt_encoding": "morgan", # {"morgan", "chemberta"}
  "fpSize": 128,
  "verbose":False
}

xgb_cv = {
  "use_wandb" : True,
  "best_params": "", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>"
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "chemberta_xgb_colSMILES.pickle",
  "output_name": "xgb_chemberta_colSMILES_seed42_best.csv",
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
  "best_params": "", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>."
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "chemberta_ffn_128.pickle",
  "output_name": "chemberta_frozen_dummy", # remember to not include .csv for this particular variable, used to name the model file also
  "modes": ["train"], # can be either "train", "test" or both
  "arrhenius": False,
  "regularisation": 0,

  # defines model architecture
  "salt_col": "salt smiles", # matches column name in df
  "salt_encoding": "chemberta", # matches column name in df
  "conts": ["mw","molality","temperature_K"], # include temp_K column even if using Arrhenius
  "temperature_name": "temperature_K",
  "fold_list":[0], 
  "seed": 42,
  "device": "cuda",
  "chemberta_model_name": 'kuelumbus/polyBERT',
  "use_salt_encoder": False, # always False for now.
  "num_polymer_features": 600,
  "num_salt_features": 768, # 768 for chemberta, 128 for morgan
  "num_continuous_vars": 3, # change to 2 if using Arrhenius mode, otherwise 3 cont variables
  "data_fraction": .01, # use something small like 0.01 if you want to do quick run for error checking

  # tunable hyperparameters
  "batch_size": 16, # cannot exceed 32 atm due to memory limits
  "accumulation_steps": 2,
  "hidden_size": 2048,
  "num_hidden_layers": 1,
  "dropout": 0.1,
  "activation_fn": "relu",
  "init_method": "glorot",
  "output_size": 1, # change to 2 if using Arrhenius mode, otherwise 1
  "freeze_layers": 12, # by default 12 layers in polyBERT
  "encoder_init_lr" : 1e-6,
  "lr": 1e-4,
  'optimizer': "AdamW_ratedecay_4_4_4",
  "scheduler": "ReduceLROnPlateau", # {"ReduceLROnPlateau", "LinearLR"}
  'warmup_steps': 100,
  "epochs": 2,
  }

ffn_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "models_dir_name": "models",
  "input_data_name": "morgan_ffn_128_arrTemp.pickle",
  "output_name": "dummy",
  "fold": 0, # the fold index
  "rounds": 12,
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
            'value': False # do not change this value. Passed to train_ffn so it does not use wandb
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
        "conts": {
            "value": ["mw","molality", "temperature_K"]
        },
        "temperature_name": {
            "value": "temperature_K"
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
            'value': 2
        },
        'data_fraction': {
            'value': 1
        },
        'batch_size': {
            'value': 16
        },
        'accumulation_steps': {
            'value': 2
        },
        'hidden_size': {
            'values': [1024,4096]
        },
        'num_hidden_layers': {
            'values': [2,3]
        },
        'dropout': {
            'value': 0.1
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
        'output_size': {
            'value': 2
        },
        'encoder_init_lr': {
            'value': 1e-6
        },
        'lr': {
            'values': [1e-4, 5e-5,1e-5]
        },
        'optimizer': {
            'value': 'AdamW_ratedecay_4_4_4'
        },
        'scheduler': {
            'value': "ReduceLROnPlateau"
        },
        'warmup_steps': {
            'value': 200
        },
        'epochs': {
            'value': 2
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