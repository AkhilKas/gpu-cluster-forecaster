.PHONY: format lint test train clean all

# ── Formatting ──
format:
	black .
	ruff check --fix .

# ── Linting (check only, no fix) ──
lint:
	black --check --diff .
	ruff check .

# ── Tests ──
test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

# ── Training ──
train:
	python scripts/train.py --model lstm --data synthetic --epochs 50

train-all:
	python scripts/train.py --model lstm --data synthetic --epochs 50
	python scripts/train.py --model seq2seq --data synthetic --epochs 50
	python scripts/train.py --model linear --data synthetic --epochs 30

# ── Cleanup ──
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

# ── Full check (run before pushing) ──
all: format lint test