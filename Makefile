.PHONY: install data train evaluate serve dashboard test lint clean

install:
	pip install -e ".[dev]"

data:
	python -m agri_credit_score.synthetic.generator --n 10000 --out data/synthetic/borrowers.parquet --seed 42 --spread-months 24

train:
	python -m agri_credit_score.models.train --data data/synthetic/borrowers.parquet --out models/

evaluate:
	python -m agri_credit_score.models.evaluate --model models/xgb.pkl --data data/synthetic/borrowers.parquet

serve:
	uvicorn agri_credit_score.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest --cov=src/agri_credit_score --cov-report=term-missing

lint:
	ruff check src/ tests/ dashboard/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
