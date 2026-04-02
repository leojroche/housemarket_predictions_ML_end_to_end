from pathlib import Path
import pandas as pd
import os 

from src.inference_pipeline.predict import (predict, DEFAULT_XGB_MODEL_PATH)

# To run the test, run the following command in the terminal: 
# $ PYTHONPATH=. pytest tests/test_features_pipeline.py

ROOT_PATH = Path(__file__).resolve().parents[1]
TMP_DIR = Path(ROOT_PATH / 'tmp')

#------------------------------------------------
# 'predict.py'
#------------------------------------------------
def test_predict():
    df_input = pd.read_csv('data/feature_engineered/fe_holdout.csv')
    df_output = predict(df_input = df_input,
            xgb_model_path=DEFAULT_XGB_MODEL_PATH)
    
    assert df_output.empty == False
    assert 'true_price' in df_output and 'predicted_price' in df_output


# if __name__=='__main__':
#     test_predict()

