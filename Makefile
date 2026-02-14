.PHONY: setup download-data preprocess features train predict evaluate all clean

PYTHON=python

setup:
	$(PYTHON) -m pip install -r requirements.txt

download-data:
	$(PYTHON) src/download_data.py

preprocess: download-data
	$(PYTHON) src/preprocess.py

features: preprocess
	$(PYTHON) src/feature_engineering.py

train: features
	$(PYTHON) src/train.py

predict: train
	$(PYTHON) src/predict.py

evaluate: predict
	$(PYTHON) src/evaluate.py

all: setup evaluate

clean:
	rm -rf data/processed/* features/* models/* results/*
