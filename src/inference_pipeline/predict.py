# Predict the price of a given dataset using a given model supporting CLI entrypoints : 
# usage: predict.py [-h] [--output OUTPUT] [--xgb_model XGB_MODEL] input
# positional arguments:
#   input                 Path to the input .csv to be predicted
# options:
#   -h, --help            show this help message and exit
#   --output OUTPUT       Path where save the predictions
#   --xgb_model XGB_MODEL
#                         Path of the XGBoost model to use

import sys, argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
ROOT_PATH = Path(__file__).resolve().parents[2]
if not str(ROOT_PATH) in sys.path:
    sys.path.append(str(ROOT_PATH))
MODELS_DIR = ROOT_PATH / 'models/'
DEFAULT_XGB_MODEL_PATH = MODELS_DIR / 'xgb_model.pkl' # XGBoost model trained on (train + eval) using the hyperparameters selected during the tuning
DEFAULT_PREDICTION_PATH = ROOT_PATH / 'predictions.csv'


def predict(
        df_input: pd.DataFrame,
        xgb_model_path: Path | str
) -> pd.DataFrame :
    # Predict the price using the given model (dataset should be feature engineered)
    # Return predictions

    # Load XGBoost model
    if not Path(xgb_model_path).exists():
        print('❌ XGBoost model pickle file not found')
        return None
    xgb_model = pickle.load(open(xgb_model_path, 'rb'))

    # Predict
    target = 'price'
    y_true = df_input[target]
    df_input = df_input.drop(columns=[target])
    y_pred = xgb_model.predict(df_input)
    df_input['true_price'] = y_true
    df_input['predicted_price'] = y_pred

    # Calculate metrics
    mae = mean_absolute_error(y_true.values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true.values, y_pred))
    r2 = r2_score(y_true.values, y_pred)
    print(f'📊 MAE: {mae:,.2f}')
    print(f'📊 RMSE: {rmse:,.2f}')
    print(f'📊 R2: {r2:,.4f}')


    return df_input 


if __name__ == '__main__':
    # Allow CLI entrypoints to inform the input data, the prediction output path and the model
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to the input .csv to be predicted', type=str)
    parser.add_argument('--output', default=DEFAULT_PREDICTION_PATH, help='Path where save the predictions', type=str)
    parser.add_argument('--xgb_model', default=DEFAULT_XGB_MODEL_PATH, help='Path of the XGBoost model to use', type=str)
    args = parser.parse_args()

    # Load dataset to predict and predict
    df_input = pd.read_csv(args.input)
    predictions = predict(df_input=df_input,
                        xgb_model_path=args.xgb_model
                    )

    # Save predictions to .csv
    predictions.to_csv(args.output)
    print(f'✅ Predictions saved to {args.output}')
    print('')
    
