from pathlib import Path
import pandas as pd
import os 
from category_encoders import TargetEncoder

from src.feature_pipeline.load_and_split import (load_and_split_data, DATE_CUTOFF_EVAL, DATE_CUTOFF_HOLDOUT)
from src.feature_pipeline.preprocess import (add_city_coordinates, 
                                             remove_outliers, 
                                             PRICE_OUTLIER_CUTOFF, 
                                             save_cleaned_data,
                                             preprocess_split)
from src.feature_pipeline.feature_engineering import (add_temporal_features, add_zipcode_freq_encoding, add_city_price_target_encoding, feature_engineering)
                                             

# To run the test, run the following command in the terminal: 
# $ PYTHONPATH=. pytest tests/test_features_pipeline.py

ROOT_PATH = Path(__file__).resolve().parents[1]
TMP_DIR = Path(ROOT_PATH / 'tmp')

#------------------------------------------------
# 'load_and_split.py'
#------------------------------------------------
def test_load_and_split():
    TMP_DIR.mkdir(exist_ok=True)
    dummy_path = TMP_DIR / 'dummy.csv'
    df_dummy = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=6, freq="365D"),
        "price": [100, 200, 300, 400, 500, 600],
        })
    df_dummy.to_csv(dummy_path)

    df_train, df_eval, df_holdout = load_and_split_data(raw_path=dummy_path, output_dir=TMP_DIR)

    assert not df_train.empty and not df_eval.empty  and not df_holdout.empty
    assert df_train['date'].max() <= pd.Timestamp(DATE_CUTOFF_EVAL)
    assert df_eval['date'].min() > pd.Timestamp(DATE_CUTOFF_EVAL)
    assert df_eval['date'].max() <= pd.Timestamp(DATE_CUTOFF_HOLDOUT)
    assert df_holdout['date'].min() > pd.Timestamp(DATE_CUTOFF_HOLDOUT)
    assert (TMP_DIR / 'train.csv').exists()
    assert (TMP_DIR / 'eval.csv').exists()
    assert (TMP_DIR / 'holdout.csv').exists()
    os.system('rm ' + str(TMP_DIR / 'train.csv'))
    os.system('rm ' + str(TMP_DIR / 'eval.csv'))
    os.system('rm ' + str(TMP_DIR / 'holdout.csv'))


#------------------------------------------------
# 'preprocess.py'
#------------------------------------------------
def test_add_city_coordinates():

    # Load sample of data
    df = pd.read_csv('data/raw/train.csv')
    df = df.sample(100)

    # Check that if usmetro dataset not present, we just skip
    df = add_city_coordinates(df, usmetro_dataset=None)
    assert 'price' in  df

    # If usmetro is present
    df = add_city_coordinates(df)
    assert 'lat' in df.columns and 'lng' in df.columns
    missing = df[df['lat'].isnull()]
    assert missing.shape[0] == 0
    missing = df[df['lng'].isnull()]
    assert missing.shape[0] == 0

def test_drop_duplicates():
    df = pd.read_csv('data/raw/train.csv')
    df = df.sample(100)

    number_duplicate_row_after = df[df.duplicated(subset=df.columns.difference(['date', 'year']))].shape[0]
    assert number_duplicate_row_after == 0

def test_remove_outliers():
    df = pd.DataFrame({'median_list_price' : [10, 20, 1000, 1_000, 10_000, 100_000, 1_000_000]})
    df = remove_outliers(df)
    assert df['median_list_price'].max() < PRICE_OUTLIER_CUTOFF

def test_save_cleaned_data():
    df = pd.DataFrame({'dummy' : [1]})
    save_cleaned_data(df, 'dummy.csv', TMP_DIR)
    filepath = TMP_DIR / 'dummy.csv'
    assert Path(filepath).exists()
    os.system('rm ' + str(filepath))


def test_preprocess_split():
    preprocess_split(split='eval.csv', output_dir=TMP_DIR)
    filepath = TMP_DIR / 'preprocessed_eval.csv'
    assert Path(filepath).exists()
    os.system('rm ' + str(filepath))


#------------------------------------------------
# 'feature_engineering.py'
#------------------------------------------------
def test_add_temporal_feature():
    df_input = pd.read_csv('data/raw/eval.csv').sample(10)
    df_output = add_temporal_features(df_input)

    assert 'year' in df_output.columns
    assert 'quart' in df_output.columns
    assert 'month' in df_output.columns


def test_add_zipcode_freq_encoding():
    df_train = pd.read_csv('data/raw/train.csv').sample(10)
    df_eval = pd.read_csv('data/raw/eval.csv').sample(10)
    df_train, df_eval, zipcode_mapping = add_zipcode_freq_encoding(df_train, df_eval)

    assert 'zipcode_freq' in df_train.columns
    assert 'zipcode_freq' in df_eval.columns
    assert isinstance(zipcode_mapping, pd.Series)

def test_add_city_price_target_encoding():
    df_train = pd.read_csv('data/raw/train.csv').sample(10)
    df_eval = pd.read_csv('data/raw/eval.csv').sample(10)
    df_train, df_eval, target_encoder = add_city_price_target_encoding(df_train, df_eval)

    assert 'city_encoded' in df_train.columns
    assert 'city_encoded' in df_eval.columns
    assert isinstance(target_encoder, TargetEncoder)

def test_feature_engineering():
    df_train = pd.read_csv('data/processed/preprocessed_train.csv').sample(10)
    df_eval = pd.read_csv('data/processed/preprocessed_eval.csv').sample(10)
    feature_engineering(df_A=df_train, df_B=df_eval, output_dir=TMP_DIR, models_dir=TMP_DIR, filename_A='file_A.csv', filename_B='file_B.csv')

    filenames = ['freq_encoder.pkl', 'target_encoder.pkl', 'file_A.csv', 'file_B.csv']
    for filename in filenames:
        filepath = Path(TMP_DIR / filename)
        print(filepath.exists())
        os.system('rm ' + str(filepath))


# if __name__ == '__main__':
#     test_feature_engineering()
