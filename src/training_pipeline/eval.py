# Evaluate the a model found on the given evaluation data.
# In the pipeline logic, this script is executed just after the tuning of the model hyperparameters 

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path
import pickle

# Paths
EVAL_PATH = Path('data/feature_engineered/tuning_fe_eval.csv')
MODELS_DIR = Path('models/tuning/')
MODEL_PATH = MODELS_DIR / 'best_xgb_model.pkl'

def eval(
        eval_path: Path | str = EVAL_PATH,
        model : Path | str | XGBRegressor = MODEL_PATH
) -> float :
    # Evaluate the performance XGBoost model on the evaluation data 
    # Returns the RMSE 

    print('Evaluating : ')

    # Load evaluation data
    df_eval = pd.read_csv(eval_path)
    target = 'price'
    X_eval = df_eval.drop(columns=[target])
    y_eval = df_eval[target]

    # Load model if needed
    if type(model)!=XGBRegressor:
        xgb_model = pickle.load(open(Path(model), 'rb'))
    
    # Predict
    y_pred = xgb_model.predict(X_eval)

    # Metrics calculation
    mae = mean_absolute_error(y_eval.values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_eval.values, y_pred))
    r2 = r2_score(y_eval.values, y_pred)

    print(f'📊 MAE: {mae:,.2f}')
    print(f'📊 RMSE: {rmse:,.2f}')
    print(f'📊 R2: {r2:,.4f}')
    print('')

    return rmse


if __name__ == '__main__':
    eval()

