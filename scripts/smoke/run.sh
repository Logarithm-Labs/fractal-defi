#!/usr/bin/env bash
#
# Smoke-test the wheel before release.
#
#   1. Build a fresh wheel + sdist via ``python -m build``.
#   2. Create a throwaway venv (so we don't touch the dev environment).
#   3. Install the wheel ALONE (no editable repo on path) into the
#      throwaway venv, plus pytest + pytest-timeout for the test run.
#   4. Run the public-API import smoke (``imports.py``).
#   5. Run the closed-form runtime smoke (``runtime.py``).
#   6. Run the offline core test suite against the installed wheel
#      (``pytest -m core --override-ini "pythonpath="`` so the repo's
#      editable copy doesn't shadow the wheel-installed package).
#   7. Tear down the throwaway venv.
#
# Exit code: 0 on success, non-zero (and noisy) on first failure.
#
# Usage:
#   bash scripts/smoke/run.sh                   # build + smoke + tests
#   SKIP_BUILD=1 bash scripts/smoke/run.sh      # reuse existing dist/*.whl
#   SKIP_TESTS=1 bash scripts/smoke/run.sh      # imports + runtime only
#   KEEP_VENV=1  bash scripts/smoke/run.sh      # don't tear down (debug)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_VENV="${SMOKE_VENV:-/tmp/fractal-smoke-$$}"

# Build needs ``build`` + ``twine`` installed. Use the dev venv if it
# exists (where ``[dev]`` extras live), else fall back to system
# ``python3`` (caller is expected to have ``build`` available).
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
echo "Using interpreter for build: $PYTHON"

cleanup() {
    if [[ -z "${KEEP_VENV:-}" ]]; then
        deactivate 2>/dev/null || true
        rm -rf "$SMOKE_VENV"
    else
        echo
        echo "KEEP_VENV set — leaving $SMOKE_VENV alive for inspection."
    fi
}
trap cleanup EXIT

cd "$REPO_ROOT"

# 1. Build.
if [[ -z "${SKIP_BUILD:-}" ]]; then
    echo "=== [1/6] building wheel + sdist ==="
    rm -rf dist build *.egg-info
    "$PYTHON" -m build >/dev/null
    echo "  built:"
    ls -lh dist/
fi

WHEEL="$(ls -1t dist/fractal_defi-*.whl | head -1)"
if [[ ! -f "$WHEEL" ]]; then
    echo "ERROR: no wheel found under dist/" >&2
    exit 1
fi
echo
echo "Using wheel: $WHEEL"

# 2-3. Fresh venv + install wheel.
echo
echo "=== [2/6] creating throwaway venv at $SMOKE_VENV ==="
"$PYTHON" -m venv "$SMOKE_VENV"
# shellcheck disable=SC1091
source "$SMOKE_VENV/bin/activate"
pip install --upgrade pip --quiet

echo
echo "=== [3/6] installing wheel + test deps into smoke venv ==="
pip install --quiet "$WHEEL"
pip install --quiet "pytest>=8.2.2" "pytest-timeout>=2.3.0"
echo "  installed packages:"
pip show fractal-defi | head -3

# Steps 4-6 run inside the SMOKE venv after ``activate`` puts its
# ``bin/`` first on PATH. Use ``python`` (no version suffix) so it
# resolves to that venv, not the dev venv we used for the build.

# 4. Imports smoke.
echo
echo "=== [4/6] public-API imports ==="
python "$REPO_ROOT/scripts/smoke/imports.py"

# 5. Runtime smoke.
echo
echo "=== [5/6] runtime smoke (closed-form lending) ==="
python "$REPO_ROOT/scripts/smoke/runtime.py"

# 6. Full core test suite — but force pytest to resolve ``import
#    fractal`` against the WHEEL-installed copy, not the editable
#    repo. We do this by temporarily disabling ``pythonpath`` from
#    pytest.ini AND moving the tests dir to a tmp location so
#    pytest's own rootdir-conftest auto-inject doesn't add the
#    repo root either.
if [[ -z "${SKIP_TESTS:-}" ]]; then
    echo
    echo "=== [6/6] pytest -m core (against the wheel) ==="
    TEST_DIR="$SMOKE_VENV/tests"
    cp -R "$REPO_ROOT/tests" "$TEST_DIR"
    cp "$REPO_ROOT/pytest.ini" "$SMOKE_VENV/"
    cd "$SMOKE_VENV"
    python -m pytest tests/ -m core -q --no-header \
        --override-ini "pythonpath=" \
        --override-ini "testpaths=tests" \
        --maxfail=10
    cd "$REPO_ROOT"
fi

echo
echo "============================================================"
echo "  ALL SMOKE CHECKS PASSED — wheel is ready for upload"
echo "============================================================"
