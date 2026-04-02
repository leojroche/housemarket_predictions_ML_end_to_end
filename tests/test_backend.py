import requests
from pathlib import Path
import pandas as pd


BASE_URL = 'http://localhost:8000'

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'data'
HOLDOUT_DATA_PATH = DATA_PATH / 'feature_engineered' / 'fe_holdout.csv'

def test_status():
    response = requests.get(f'{BASE_URL}/health')
    if response.json()['status']=='healthy':
        print('✅ status : healthy')
    else :
        print('❌ status : unhealthy')

    assert response.json()['status']=='healthy'


def test_predict():

    df_holdout = pd.read_csv(HOLDOUT_DATA_PATH).sample(10)
    input = df_holdout.to_dict('records')

    # Submit request to predict 
    response = requests.post(f'{BASE_URL}/predict', json=input)

    # Check response status code
    if response.status_code == 200:
        print('✅ valid status code')
    else :
        print('❌ invalid status code')

    # Check if length from input and response math
    predictions = response.json()
    if len(predictions) == len(input):
        print('✅ valid prediction length')
    else :
        print('❌ invalid invalid prevision length')

    print(predictions)


if __name__ == '__main__':
    test_status()
    test_predict()