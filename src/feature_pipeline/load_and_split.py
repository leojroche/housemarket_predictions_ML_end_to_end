# Loads the original raw data and splits it in three different datasets : 
# - 'train.csv' : from beginning to DATE_CUTOFF_EVAL
# - 'eval.csv' : from DATE_CUTOFF_EVAL to DATE_CUTOFF_HOLDOUT
# - 'holdout.csv' : from 'date_cutoff_holdout' to end

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path('data/raw/')
RAW_DATA_PATH = RAW_DATA_DIR / 'untouched_raw_original.csv'

DATE_CUTOFF_EVAL = '2020-01-01'
DATE_CUTOFF_HOLDOUT = '2022-01-01'

def load_and_split_data(
        raw_path : Path | str = RAW_DATA_PATH, 
        output_dir : Path | str = RAW_DATA_DIR
):
    # Loads the raw data located at raw_path. The data is split and then save in output_dir as .csv files
    # Returns the splitted datasets as DataFrames

    print('Load and split data : ')

    # Loading the raw original dataset
    df = pd.read_csv(raw_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date'])
    print(f'Full raw dataset size: {df.shape}')

    # Dates determining where the dataset is splitted
    date_cutoff_eval = pd.Timestamp(DATE_CUTOFF_EVAL)
    date_cutoff_holdout = pd.Timestamp(DATE_CUTOFF_HOLDOUT)

    # Splitting the dataset
    df_train = df[df['date'] <= date_cutoff_eval]
    df_eval = df[(df['date'] > date_cutoff_eval) & (df['date'] <= date_cutoff_holdout)]
    df_holdout = df[df['date'] > date_cutoff_holdout]

    # Saving the datasets and printing their size
    print(f'Train dataset : {df_train.shape}, {float(100*df_train.shape[0]/df.shape[0]):.2f} % ', end='')
    df_train.to_csv(output_dir / 'train.csv', index=False)
    print('✅ Saved (train.csv)')

    print(f'Eval dataset : {df_eval.shape}, {float(100*df_eval.shape[0]/df.shape[0]):.2f} % ', end='')
    df_eval.to_csv(output_dir / 'eval.csv', index=False)
    print('✅ Saved (eval.csv)')

    print(f'Hold-out dataset : {df_holdout.shape}, {float(100*df_holdout.shape[0]/df.shape[0]):.2f} % ', end='')
    df_holdout.to_csv(output_dir / 'holdout.csv', index=False)
    print('✅ Saved (holdout.csv)\n')

    return df_train, df_eval, df_holdout


if __name__== '__main__':
    load_and_split_data()
