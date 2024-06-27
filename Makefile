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
	rm -rf $(VENV)
	find . -name __pycache__ -exec rm -rf {} \;
	find . -name .pytest_cache -exec rm -rf {} \;
	find . -name .mypy_cache -exec rm -rf {} \;
	find . -name .coverage -exec rm -rf {} \;
	find . -name '*runs' -exec rm -rf {} \;
	find . -name '*mlruns' -exec rm -rf {} \;
	find . -name '*mlartifacts' -exec rm -rf {} \;
	find . -name .ipynb_checkpoints -exec rm -rf {} \;
	find . -name '*loader' -exec rm -rf {} \;
	find . -name '*_cache' -exec rm -rf {} \;
	find . -name '*_output' -exec rm -rf {} \;
	find . -name '*_logs' -exec rm -rf {} \;
	find . -name '*_data' -exec rm -rf {} \;
	find . -name '*_results' -exec rm -rf {} \;
	find . -name '*build' -exec rm -rf {} \;
	find . -name '*dist' -exec rm -rf {} \;
	find . -name '*egg-info' -exec rm -rf {} \;


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

lint: isort flake mypy pylint


# Test
.pytest:
	pytest -s -v

test: .venv .pytest


# Docker
build:
	docker-compose build

run: build
	docker-compose up -d

# All
dev: setup format lint test
all: setup format lint test run

.DEFAULT_GOAL = run
