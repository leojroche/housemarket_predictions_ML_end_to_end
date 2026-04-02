from pathlib import Path
import pandas as pd
import os 

from src.training_pipeline.train import train
from src.training_pipeline.eval import eval
from src.training_pipeline.tune import tune

# To run the test, run the following command in the terminal: 
# $ PYTHONPATH=. pytest tests/test_features_pipeline.py

ROOT_PATH = Path(__file__).resolve().parents[1]
TMP_DIR = Path(ROOT_PATH / 'tmp')

#------------------------------------------------
# 'tune.py'
#------------------------------------------------
def test_tune():
    best_params, best_rmse = tune(n_trials = 1,
            experiment_name = 'test',
            models_dir = TMP_DIR,
            model_name = 'test_model',
            save_model  = True,
            random_state = 42
        )
    assert isinstance(best_params, dict)
    assert isinstance(best_rmse, float)
    model_path = TMP_DIR / 'test_model.pkl'
    assert Path(model_path).exists()
    os.system('rm ' + str(model_path))
    hyperparams_path = TMP_DIR / 'best_hyperparameters.pkl'
    assert Path(hyperparams_path).exists()
    os.system('rm ' + str(hyperparams_path))
    
#------------------------------------------------
# 'eval.py'
#------------------------------------------------
def test_eval():
    rmse = eval()
    assert isinstance(rmse, float) 

#------------------------------------------------
# 'train.py'
#------------------------------------------------
def test_train():
    train(models_dir=TMP_DIR,
        save_model=True)

    filepath = TMP_DIR / 'xgb_model.pkl'
    assert Path(filepath).exists()
    os.system('rm ' + str(filepath))

    
# if __name__=='__main__':
#     eval()

