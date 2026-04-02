clean_data : # Delete all the data file except the original raw data needed to start the pipeline
	rm -f data/raw/train.csv
	rm -f data/raw/eval.csv
	rm -f data/raw/holdout.csv
	rm -rf data/processed/
	rm -rf data/feature_engineered/

clean_models: # Delete all the saved models
	rm -rf models/

clean_all:
	make clean_data
	make clean_models
	rm -f predictions.csv
	rm -rf tmp/
	rm -f mlflow.db

feature_pipeline: # Run the whole feature pipeline : splits the raw data, preprocesses, feature engineers to generate tuning and training ready datasets
	python src/feature_pipeline/load_and_split.py
	python src/feature_pipeline/preprocess.py
	python src/feature_pipeline/feature_engineering.py

training_pipeline: # Run the whole training pipeline : tune the hyperparameters, evaluate the resulting model, use selected hyperparameters to train model on (train + eval)
	python src/training_pipeline/tune.py
	python src/training_pipeline/eval.py 
	python src/training_pipeline/train.py 

inference_pipeline: # Runs the inference pipeline : predict the price in holdout
	python src/inference_pipeline/predict.py  "data/feature_engineered/fe_holdout.csv"

build : # Runs the whole pipeline
	make feature_pipeline
	make training_pipeline
	make inference_pipeline

sb : # Start Backend
	fastapi dev src/api/backend.py

sf : # Start Frontend (UI)
	streamlit run src/api/frontend.py

run_pipeline_tests : # Perform all pipeline tests
	PYTHONPATH=. pytest tests/test_features_pipeline.py	
	PYTHONPATH=. pytest tests/test_training_pipeline.py	
	PYTHONPATH=. pytest tests/test_inference_pipeline.py	

