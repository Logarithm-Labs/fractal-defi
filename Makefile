# Fractal — Makefile convenience targets.
#
# Run ``make <target>`` from the repo root. ``make`` (no arg) prints
# the help. Targets are kept in lock-step with the pre-commit hooks
# (``.pre-commit-config.yaml``) and the GitHub Actions workflow
# (``.github/workflows/ci.yml``) — anything you can run locally is
# what CI will run on your PR.

VENV     := venv
PYTHON   := python3

ifeq ($(OS),Windows_NT)
    BIN := $(VENV)/Scripts
else
    BIN := $(VENV)/bin
endif

PROJECT  := fractal
TESTS    := tests
EXAMPLES := examples

.PHONY: help setup install pre-commit format lint \
        test test-slow test-integration test-all test-e2e \
        docs docs-strict docs-serve docs-clean \
        clean clean-runs clean-all \
        build release-test release

help:
	@echo "Fractal Makefile targets"
	@echo ""
	@echo "  Setup:"
	@echo "    setup            create venv, install editable + dev extras, install pre-commit hooks"
	@echo "    install          install editable + dev extras into the active environment"
	@echo ""
	@echo "  Lint / format:"
	@echo "    pre-commit       run every pre-commit hook (matches CI lint job)"
	@echo "    format           auto-fix import order (isort)"
	@echo "    lint             flake8 + pylint(fractal) + pylint(tests) + isort --check"
	@echo ""
	@echo "  Tests (layered):"
	@echo "    test             offline core suite (~1100 tests, ~10s) — default for CI on every PR"
	@echo "    test-slow        real-data CSV-replay smoke tests"
	@echo "    test-integration live-API tests (Binance / Hyperliquid / Aave / TheGraph)"
	@echo "    test-all         every layer combined (core + slow + integration)"
	@echo "    test-e2e         Docker MLflow end-to-end harness"
	@echo ""
	@echo "  Docs:"
	@echo "    docs             build Sphinx html (open docs/build/html/index.html)"
	@echo "    docs-strict      build Sphinx with warnings-as-errors (CI mode)"
	@echo "    docs-serve       serve built docs on http://localhost:8000"
	@echo "    docs-clean       remove the built docs"
	@echo ""
	@echo "  Clean (artifacts pile up under examples/, runs/, fractal_data/):"
	@echo "    clean            python bytecode, test caches, packaging artifacts, sphinx build"
	@echo "    clean-runs       loader cache (fractal_data/), strategy run logs (runs/), MLflow data"
	@echo "    clean-all        clean + clean-runs"
	@echo ""
	@echo "  Release:"
	@echo "    build            sdist + wheel via ``python -m build``"
	@echo "    release-test     upload dist/* to test PyPI"
	@echo "    release          upload dist/* to PyPI"

# ─── setup ─────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/pre-commit install
	@echo ""
	@echo "Done. Activate with: source $(VENV)/bin/activate"

install:
	pip install -e ".[dev]"

# ─── lint / format ────────────────────────────────────────────────
pre-commit:
	pre-commit run --all-files

format:
	isort $(PROJECT) $(TESTS) $(EXAMPLES)

lint:
	isort --check-only --diff $(PROJECT) $(TESTS) $(EXAMPLES)
	flake8 $(PROJECT) $(TESTS)
	pylint $(PROJECT) --disable=R --score=no
	pylint --rcfile=$(TESTS)/.pylintrc $(TESTS) --disable=R --score=no

# ─── tests ────────────────────────────────────────────────────────
test:
	pytest -m core -q --no-header

test-slow:
	pytest -m slow -q --no-header

test-integration:
	pytest -m integration -q --no-header

test-all:
	pytest -m "" -q --no-header

test-e2e:
	bash tests/mlflow_tests/scripts/e2e.sh

# ─── docs ─────────────────────────────────────────────────────────
docs:
	$(MAKE) -C docs html

docs-strict:
	$(MAKE) -C docs html SPHINXOPTS="-W"

docs-serve:
	@echo "Serving docs on http://localhost:8000 (Ctrl+C to stop)"
	$(PYTHON) -m http.server -d docs/build/html 8000

docs-clean:
	$(MAKE) -C docs clean

# ─── clean ────────────────────────────────────────────────────────
# ``clean`` is safe to run any time — only deletes regenerable build
# artifacts + caches.
clean:
	# Python bytecode (anywhere except venv / .git)
	find . -type d -name __pycache__ \
	    ! -path './venv/*' ! -path './.git/*' \
	    -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) \
	    ! -path './venv/*' ! -path './.git/*' \
	    -delete 2>/dev/null || true
	# Test caches
	rm -rf .pytest_cache .mypy_cache .hypothesis .ipynb_checkpoints
	rm -rf .coverage .coverage.* htmlcov coverage.xml
	# Packaging
	rm -rf build dist
	find . -type d -name '*.egg-info' \
	    ! -path './venv/*' ! -path './.git/*' \
	    -exec rm -rf {} + 2>/dev/null || true
	# Sphinx build output
	rm -rf docs/build

# ``clean-runs`` wipes USER DATA: cached loader CSVs, strategy run
# logs, MLflow stores, example output files. Run only if you don't
# need the cache (everything regenerates from source).
clean-runs:
	# Loader cache (cwd-relative ``fractal_data/`` from any script)
	find . -type d -name fractal_data \
	    ! -path './venv/*' ! -path './.git/*' \
	    -exec rm -rf {} + 2>/dev/null || true
	# Strategy run logs (cwd-relative ``runs/``)
	find . -type d -name runs \
	    ! -path './venv/*' ! -path './.git/*' \
	    -exec rm -rf {} + 2>/dev/null || true
	# MLflow local stores
	rm -rf mlruns mlartifacts
	rm -rf tests/mlflow_tests/mlflow-data tests/mlflow_tests/output
	# Example output files (regenerated by quick_start.py / backtests)
	rm -f quick_start_result.csv examples/quick_start_result.csv

clean-all: clean clean-runs

# ─── release ──────────────────────────────────────────────────────
build: clean
	$(PYTHON) -m build

release-test: build
	$(PYTHON) -m twine upload --repository testpypi dist/*

release: build
	$(PYTHON) -m twine upload dist/*

.DEFAULT_GOAL := help
