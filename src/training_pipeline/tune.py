# Tune the hyperparameter of a XGBoost model on the preprocessed and feature engineered training and eval data using Optuna. 
# The performance of each model and the models themselves for each optimization iteration are saved with MLflow.
# The best performing model and its hyperparameters are saved as pickle files for the next final training. 

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path
import pickle
import optuna, mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
FE_DIR = Path('data/feature_engineered/')
TRAIN_PATH = FE_DIR / 'tuning_fe_train.csv'
EVAL_PATH = FE_DIR / 'tuning_fe_eval.csv'
MODELS_DIR = Path('models/tuning/')

def tune(
        train_path: Path | str = TRAIN_PATH,
        eval_path: Path | str = EVAL_PATH,
        n_trials: int = 15,
        experiment_name: str = 'XGBoost Hyperparameter Tuning Experiment',
        tracking_uri : str = None,
        models_dir : Path | str = MODELS_DIR,
        model_name : str = 'best_xgb_model',
        save_model : bool = True,
        random_state : int = 42,
) -> tuple[dict, dict]:
    # Tunes the hyperparameters of a XGBoost model on the preprocessed and feature engineered training and evaluating data using Optuna and track the performance of the different models with MLflow.
    # The best performing model and its hyperparameters are saved as pickle files for the next final training. 
    # Returns best_model_parameters (dict), best_rmse (float).

    # Optuna's objective function to minimize
    def objective(trial: optuna.Trial) -> float : 
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run : 
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": random_state,
                "n_jobs": -1,
                "tree_method": "hist",
            }

            # train and evaluate model
            xgb_model = XGBRegressor(**model_params)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_eval)
            mae = mean_absolute_error(y_eval.values, y_pred)
            rmse = np.sqrt(mean_squared_error(y_eval.values, y_pred))
            r2 = r2_score(y_eval.values, y_pred)
            print('XGBoost:')
            print(f'MAE: {mae:,.2f}')
            print(f'RMSE: {rmse:,.2f}')
            print(f'R2: {r2:,.4f}')

            mlflow.log_params(model_params) # save model parameters
            mlflow.log_metrics({'mae':mae, 'rmse':rmse, 'r2':r2}) # save metrics
            # log model (how?)
            trial.set_user_attr("run_id", child_run.info.run_id)

        return rmse

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load train and eval data
    df_train = pd.read_csv(train_path)
    df_eval = pd.read_csv(eval_path)
    target = 'price'
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_eval = df_eval.drop(columns=[target])
    y_eval = df_eval[target]

    # Start MLflow model performance tracking 
    with mlflow.start_run(run_name='study') as run:

        mlflow.log_param('n_trials', n_trials)

        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        print(f'🎛  XGBoost hyperparameters tuned (n_trials = {n_trials})')

        # Train and evaluate best model
        best_xbg_model = XGBRegressor(**study.best_trial.params)
        best_xbg_model.fit(X_train, y_train)
        y_pred = best_xbg_model.predict(X_eval)
        mae = mean_absolute_error(y_eval.values, y_pred)
        rmse = np.sqrt(mean_squared_error(y_eval.values, y_pred))
        r2 = r2_score(y_eval.values, y_pred)
        print(f'Best metrics :')
        print(f'📊 MAE: {mae:,.2f}')
        print(f'📊 RMSE: {rmse:,.2f}')
        print(f'📊 R2: {r2:,.4f}')

        # Save best model in MLflow
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_error": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)

        # Save best model as pkl
        if save_model:
            model_path = models_dir / str(model_name + '.pkl')
            pickle.dump(best_xbg_model, open(model_path, 'wb'))
            print(f'✅ Best model saved in {model_path}')

        # Save best hyperparameters as pkl
        best_params = study.best_trial.params
        print('✅ Best hyperparameters : ', best_params)
        best_hyperparams_path = models_dir / 'best_hyperparameters.pkl'
        pickle.dump(best_params, open(best_hyperparams_path, 'wb'))
        print(f'✅ Best hyperparameters saved in {best_hyperparams_path}')
        print('')


    return best_params, study.best_value
        

if __name__ == '__main__':
    tune()


