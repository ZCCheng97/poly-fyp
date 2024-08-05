data_cleaning = {
    "data_dir_name": "data",
    "data_name": "clean_train_data.csv",
    "cleaned_data_name": "cleaned_data.xlsx" # should be saved as a xlsx file
}

preprocessing = {
  "data_dir_name": "data",
  "input_data_name": "cleaned_data.xlsx",
  "output_data_name": "polyBERT_xgb.pickle",
  "cats": ["psmiles","salt smiles"],
  "conts": ["mw","molality", "temperature_K"],
  "drop_columns": ["raw_psmiles","long_smiles","temperature"],
  "train_ratio":0.8,
  "val_ratio":0.1,
  "nfolds": 10,
  "polymer_use_fp": "polybert", # {"polybert", "morgan", "none"}
  "salt_use_fp": "morgan", # {"morgan", chemprop?}
  "fpSize": 128,
  "verbose":True
}

xgb_cv = {
  "use_wandb" : True,
  "best_params": "zccheng97-nanyang-technological-university-singapore/xgb_poly_hpsweep/2p05zvwu", # leave blank to not use best wandb sweep, otherwise use "<entity>/<project>/<run_id>"
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "polyBERT_xgb.pickle",
  "output_name": "xgb_polyBERT_colSMILES_best.csv",
  "seed_list":[42,3,34,43,83], 
  "verbose": True,
}

xgb_sweep = {
  "data_dir_name": "data",
  "results_dir_name": "results",
  "input_data_name": "polyBERT_xgb.pickle",
  "output_name": "xgb_poly_hpsweep.csv",
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
  "verbose": True,

}

step_args = {
       'data_cleaning': data_cleaning,
       'preprocessing': preprocessing,
       "xgb_cv": xgb_cv,
       "xgb_sweep": xgb_sweep}