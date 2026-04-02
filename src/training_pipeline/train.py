# Train a XGBoost model.
# In the pipeline logic, the hyperparameter of the model are already tuned using 'tune.py' and this training uses the hyperparameters of the best performing model.
# The datasets for the training of the final prediction model are (train + eval).


import pandas as pd
from xgboost import XGBRegressor
from pathlib import Path
import pickle

# Paths
TRAIN_PATH = Path('data/feature_engineered/fe_train_eval.csv')
MODELS_DIR = Path('models/')
MODEL_HYPERPARAMS_PATH = MODELS_DIR / 'tuning' / 'best_hyperparameters.pkl'

def train(
        train_path: Path | str = TRAIN_PATH,
        model_hyperparams_path: Path | str = MODEL_HYPERPARAMS_PATH,
        models_dir : Path | str = MODELS_DIR,
        save_model : bool = True,
) -> XGBRegressor :
    # Train a XGBoost model on the train data and save it as a pickle file if needed
    # Return the trained model

    print('Training model:')
    if not Path(model_hyperparams_path).exists():
        print('❌ Model hyperparameters pickle file not found.')
        return None

    # Loading hyperparameters
    model_hyperparams = pickle.load(open(model_hyperparams_path, 'rb'))
    print('✅ Hyperparameters loaded')

    # Train XGBoost model
    df_train = pd.read_csv(train_path)
    target = 'price'
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    xgb_model = XGBRegressor(**model_hyperparams)
    xgb_model.fit(X_train, y_train)
    print('⚙️ XGBoost model trained')

    # Save the model as pickle file if needed
    if save_model:
        filepath = models_dir / 'xgb_model.pkl'
        pickle.dump(xgb_model, open(filepath, 'wb'))
        print(f'✅ model saved as {filepath}')

    print('')

    return xgb_model


if __name__ == '__main__':
    train()

