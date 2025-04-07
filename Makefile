VENV := venv

ifeq ($(OS),Windows_NT)
   BIN=$(VENV)/Scripts
else
   BIN=$(VENV)/bin
endif

export PATH := $(BIN):$(PATH)

PROJECT := fractal
TESTS := tests


# Clean
clean:
	rm -rf $(VENV) || true
	find . -name __pycache__ -exec rm -rf {} \; || true
	find . -name .pytest_cache -exec rm -rf {} \; || true
	find . -name .mypy_cache -exec rm -rf {} \; || true
	find . -name .coverage -exec rm -rf {} \; || true
	find . -name '*runs' -exec rm -rf {} \; || true
	find . -name '*mlruns' -exec rm -rf {} \; || true
	find . -name '*mlartifacts' -exec rm -rf {} \; || true
	find . -name .ipynb_checkpoints -exec rm -rf {} \; || true
	find . -name '*loader' -exec rm -rf {} \; || true
	find . -name '*_cache' -exec rm -rf {} \; || true
	find . -name '*_output' -exec rm -rf {} \; || true
	find . -name '*_logs' -exec rm -rf {} \; || true
	find . -name '*_data' -exec rm -rf {} \; || true
	find . -name '*_results' -exec rm -rf {} \; || true
	find . -name '*build' -exec rm -rf {} \; || true
	find . -name 'dist' -exec rm -rf {} \; || true
	find . -name '*.egg-info' -exec rm -rf {} \; || true


# Setup
.venv:
	python3 -m venv $(VENV)
	pip3 install -r requirements.txt

setup: .venv


# Format
isort_fix: .venv
	isort $(PROJECT) $(TESTS)

format: isort_fix


# Lint
isort: .venv
	isort --check $(PROJECT) $(TESTS)

flake: .venv
	flake8 $(PROJECT)

mypy: .venv
	mypy $(PROJECT)

pylint: .venv
	pylint $(PROJECT) --disable=C0116,C0115,C0114,C0301,C3001,W0622

lint: isort flake pylint


# Test
.pytest:
	pytest -vvs

test: .venv .pytest


build:
	python3 setup.py sdist bdist_wheel


# Requires export TWINE_PASSWORD=...
twine_upload:
	twine upload dist/*


twine_upload_test:
	twine upload --repository testpypi dist/*

# All
dev: setup format lint test
all: setup format lint test run

.DEFAULT_GOAL = run
