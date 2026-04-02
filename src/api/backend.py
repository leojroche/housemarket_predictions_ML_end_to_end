
# House Market Prediction Backend API powered by FastAPI in oder to ask for prediction using the inference pipeline and the already trained models.

# Run the API locally for debugging purposes (from root directory) using : 
# $ fastapi dev src/api/backend.py

# The API can be tested and debugged using '/tests/test_backend.py'.

from fastapi import FastAPI
from pathlib import Path
import pandas as pd 
import sys

app = FastAPI()

# Paths
ROOT_PATH = Path(__file__).resolve().parents[2]
MODELS_PATH = ROOT_PATH / 'models'
MODEL_PATH = MODELS_PATH / 'xgb_model.pkl'

if not str(ROOT_PATH) in sys.path:
    sys.path.append(str(ROOT_PATH))
from src.inference_pipeline.predict import predict

# Welcome message at the start of the API
@app.get("/")
def read_root():
    return {"message": "House Market Prediction API is running 🚀."}

@app.get('/health') 
def read_health(): # Checks the health of the API : if the necessary model pickle files exist
    status = MODEL_PATH.exists() 
    if status:
        return {'status' : 'healthy'}
    else:
        return {'status' : 'unhealthy'}
    
@app.post('/predict') # Return the predicted price of the data given as inputs (should be already preprocessed and feature engineered)
def get_predictions(inputs: list[dict]) -> list[float]:

    if not MODEL_PATH.exists():
        print('❌ XGBoost model not found')
        return None

    df_inputs = pd.DataFrame(inputs)
    if df_inputs.empty:
        print('❌ No input provided')
        return None

    df_predictions = predict(df_inputs, MODEL_PATH) # make prediction using the inference pipeline
    predictions = df_predictions['predicted_price']

    return predictions

