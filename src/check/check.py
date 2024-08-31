from pathlib import Path
import pickle
from tqdm import tqdm

def check(args):
  script_dir = Path(__file__).resolve().parent

  data_dir = script_dir.parent.parent / args.data_dir_name
  input_data_path = data_dir / args.input_data_name

  with open(input_data_path, 'rb') as f:
     data = pickle.load(f) # object of list of Datasplit Classes
  for fold in tqdm(args.fold_list):
        
        print(f"Currently running fold: {fold}")
        datasplit = data[fold]
        tr,val,te = datasplit.x_train, datasplit.x_val, datasplit.x_test
        # Get unique types in each dataframe
        train_types = set(tr[args.feature_columns].apply(tuple, axis=1).unique())
        val_types = set(val[args.feature_columns].apply(tuple, axis=1).unique())
        test_types = set(te[args.feature_columns].apply(tuple, axis=1).unique())

        # Check if any type in test set is present in train or validation set
        leakage_in_train = test_types.intersection(train_types)
        leakage_in_val = test_types.intersection(val_types)

        if not leakage_in_train and not leakage_in_val:
            print("No leakage detected in test set")
        else:
            print(f"Data leakage detected in train set: {leakage_in_train}")
            print(f"Data leakage detected in validation set: {leakage_in_val}")