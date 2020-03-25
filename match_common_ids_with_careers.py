from utilities.pandas_utils import *
import pickle
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onet_skills', type=str, required=True)
    parser.add_argument('--col_to_use', type=str, choices=['Career_Cluster', 'Career_Pathway'], required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    onet_df = pickle.load(open(args.onet_data, 'rb'))
    col_to_use = args.col_to_use.replace('_', ' ')
    onet_df = onet_df['common_id', col_to_use].drop_duplicates()
    result_dict = make_two_cols_map(onet_df, 'common_id', col_to_use)
    with open(os.path.join(args.output_dir, args.col_to_use+'_id_map.json'), 'w') as f:
        json.dump(result_dict, f)

if __name__ == '__main__':
    main()