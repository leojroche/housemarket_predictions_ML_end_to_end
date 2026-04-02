# Feature engineering of the two different datasets :
# - Datasets used for the model hyperparameter tuning (train, eval) :
#       1. Training      : train
#       2. Evaluating    : eval
# - Datasets used to train the final model used to perform the final predictions
#       1. Training      : train + eval
#       2. Prediction    : holdout

# The feature engineering includes the following steps: 
# - Adding temporal features : 'year', 'quart' and 'month' from the 'date' feature
# - Adding zipcode frequency encoding 'zipcode_freq'. For the two datasets, the encoder is trained on dataset 1. to prevent data leakage on dataset 2.
# - Adding city price target encoding 'city_encoded'. For the two datasets, the encoder is trained on dataset 1. to prevent data leakage on dataset 2.
# - Removing the unnecessary columns to keep numerical features

# We assume that the datasets (train, eval and holdout) were already preprocessed and saved in 'data/processed/'

import pandas as pd
from category_encoders import TargetEncoder
from pathlib import Path
import pickle 

# Dataset paths
PROCESSED_DIR = Path('data/processed/')
PROCESSED_TRAIN_PATH = PROCESSED_DIR / 'preprocessed_train.csv'
PROCESSED_EVAL_PATH = PROCESSED_DIR / 'preprocessed_eval.csv'
PROCESSED_HOLDOUT_PATH = PROCESSED_DIR / 'preprocessed_holdout.csv'
FE_DIR = Path('data/feature_engineered/') 
# Model paths
MODELS_DIR = Path('models/')
MODELS_TUNING_DIR = Path('models/tuning/')
# Columns to drop 
COLUMNS_TO_DROP = ['date', 'zipcode', 'city_full', 'city', 'median_sale_price', 'metro_full']

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame :
    # Add the year (int), quart (int) and month (int) features from the known date to the DataFrame and returns the DataFrame

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['quart'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month

    return df


def add_zipcode_freq_encoding(
                df_train: pd.DataFrame,
                df_eval: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    # zipcode frequency encoding. The Encoder is trained only on df_train to avoid data leakage
    # Returns a tuple containing the train and eval DataFrames with the 'zipcode_freq' feature and the zipcode_mapping (pd.Serie)

    zipcode_mapping = df_train['zipcode'].value_counts()
    df_train['zipcode_freq'] = df_train['zipcode'].map(zipcode_mapping)
    df_eval['zipcode_freq'] = df_eval['zipcode'].map(zipcode_mapping).fillna(0)

    print('✅ zipcode_freq features added (frequency encoding)')

    return (df_train, df_eval, zipcode_mapping)
    
def add_city_price_target_encoding(
                df_train: pd.DataFrame,
                df_eval: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    # Target encoding from 'city_full' to 'price'. The encoder is only fitted on the training dataset to avoid data leakage
    # Returns a tuple containing the train and eval DataFrames with the 'city_encoded' feature and the target encoder

    target_encoder = TargetEncoder(cols=['city_full'])
    df_train['city_encoded'] = target_encoder.fit_transform(df_train['city_full'], df_train['price']) 
    df_eval['city_encoded'] = target_encoder.transform(df_eval['city_full'])

    print('✅ city_encoded feature added (target encoding)')

    return (df_train, df_eval, target_encoder)

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame :
    # Drop the unused columns in order to keep numeral values and returns the DataFrame

    df.drop( columns = COLUMNS_TO_DROP, inplace=True)

    return df

def save_feature_engineered_data(
                df: pd.DataFrame | str,
                filename : str,
                output_dir: Path | str = PROCESSED_DIR
) -> pd.DataFrame :
    # Save the DataFrame in 'output_dir' under 'filename'

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / filename, index=False)

    return df



def feature_engineering(
                df_A: pd.DataFrame,
                df_B: pd.DataFrame,
                output_dir: Path | str,
                models_dir: Path | str,
                filename_A : str,
                filename_B : str
):
    # Feature engineering of two datasets A and B using the above function, where the encoders are only trained on dataset A to avoid data leakage

    models_dir.mkdir(parents=True, exist_ok=True)

    # Add temporal features (year, quart, month)
    df_A = add_temporal_features(df_A)
    df_B = add_temporal_features(df_B)
    print('✅ year, quart and month features added')

    # Add zipcode_freq feature
    (df_A, df_B, freq_encoding) = add_zipcode_freq_encoding(df_A, df_B)
    pickle.dump(freq_encoding, open(models_dir / 'freq_encoder.pkl', 'wb')) # saving frequency encoding model 

    # Add city_encoded feature
    (df_A, df_B, target_encoder) = add_city_price_target_encoding(df_A, df_B)
    pickle.dump(target_encoder, open(models_dir / 'target_encoder.pkl', 'wb')) # saving frequency encoding model 

    # Drop unused columns
    df_A = drop_unused_columns(df_A)
    df_B = drop_unused_columns(df_B)
    print('✅ Unused columns were dropped')

    # Save feature engineered data
    save_feature_engineered_data(df_A, filename_A, output_dir)
    save_feature_engineered_data(df_B, filename_B, output_dir)
    print(f'✅ {filename_A} and {filename_B} saved to {output_dir}')

if __name__=='__main__':

    # Loading preprocessed data
    df_train = pd.read_csv(PROCESSED_TRAIN_PATH)
    df_eval = pd.read_csv(PROCESSED_EVAL_PATH)
    df_holdout = pd.read_csv(PROCESSED_HOLDOUT_PATH)
    print('✅ Preprocessed data loaded')

    # Feature engineering of the datasets used for the hyperparameter tuning (train, eval)
    print('Feature engineering data for hyperparameter tuning (train, eval): ')
    feature_engineering(
                df_A=df_train.copy(), 
                df_B=df_eval.copy(), 
                output_dir=FE_DIR, 
                models_dir=MODELS_TUNING_DIR, 
                filename_A='tuning_fe_train.csv', 
                filename_B='tuning_fe_eval.csv')

    
    # Feature engineering of the datasets used for the training of the final model and the final prediction (train : train + eval, prediction : holdout)
    print('Feature engineering data for final model (train + eval, holdout): ')
    df_train_eval = pd.concat([df_train, df_eval])
    feature_engineering(
                df_A=df_train_eval, 
                df_B=df_holdout, 
                output_dir=FE_DIR, 
                models_dir=MODELS_DIR, 
                filename_A='fe_train_eval.csv', 
                filename_B='fe_holdout.csv')
    print('')

