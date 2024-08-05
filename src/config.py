data_cleaning = {
    "data_dir_name": "data",
    "data_name": "clean_train_data.csv",
    "cleaned_data_name": "cleaned_data.xlsx" # should be saved as a xlsx file
}

preprocess_xgb = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "morgan_xgb_128.pickle",
  "cats": ["long_smiles","salt smiles"],
  "conts": ["mw","molality", "temperature_K"],
  "drop_columns": ["raw_psmiles","psmiles","temperature"],
  "train_ratio":0.8,
  "val_ratio":0.1,
  "nfolds": 10,
  "polymer_use_fp": "morgan", # {"polybert", "morgan", "none"}
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
  "best_params": "", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>"
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "morgan_xgb_128.pickle",
  "output_name": "xgb_morgan_colSMILES.csv",
  "seed_list":[42,3,34,43,83], 
  "verbose": True,
}

xgb_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "morgan_xgb_128.pickle",
  "output_name": "xgb_morgan_hpsweep.csv",
  "seed_list":[42], 
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

step_args = {
       'data_cleaning': data_cleaning,
       'preprocess_xgb': preprocess_xgb,
       'preprocess_ffn':preprocess_ffn,
       "xgb_cv": xgb_cv,
       "xgb_sweep": xgb_sweep}