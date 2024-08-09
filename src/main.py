import argparse
import config as config_module
from args import get_args

from data_cleaning.data_cleaning import data_cleaning
from preprocessing.preprocess_xgb import preprocess_xgb
from preprocessing.preprocess_ffn import preprocess_ffn
from train_xgb.xgb_cv import xgb_cv
from train_xgb.xgb_sweep import xgb_sweep
from train_ffn.ffn_cv import ffn_cv
from train_ffn.ffn_sweep import ffn_sweep

import wandb
import os

step_funcs = {"data_cleaning": data_cleaning,
              "preprocess_xgb": preprocess_xgb,
              "preprocess_ffn": preprocess_ffn,
              "xgb_cv": xgb_cv,
              "xgb_sweep": xgb_sweep,
              "ffn_cv": ffn_cv,
              "ffn_sweep": ffn_sweep
              }

def main():
    # wandb.login(key=os.environ["WANDB_API_KEY"])

    parser = argparse.ArgumentParser(description='Run pipeline steps.')
    parser.add_argument('steps', nargs='+', help='List of steps to run')
    args = parser.parse_args()

    for step in args.steps:
        step_funcs[step](get_args(config_module.step_args[step]))

if __name__ == "__main__":
    main()