install:
	pip install -r requirements.txt

run:
	python linear_regression_predictor.py

test:
	python -m pytest tests/

all: install run
