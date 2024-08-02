import argparse
import config as config_module
from args import get_args

from data_cleaning.data_cleaning import data_cleaning

def main():
    parser = argparse.ArgumentParser(description='Run pipeline steps.')
    parser.add_argument('steps', nargs='+', help='List of steps to run')
    
    args = parser.parse_args()

    for step in args.steps:
        if step == "data_cleaning":
            data_cleaning(get_args(config_module.step_args[step]))

if __name__ == "__main__":
    main()