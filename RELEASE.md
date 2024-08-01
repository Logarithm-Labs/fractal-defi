# Release Notes
1. Run tests
```
make .venv
source venv/bin/activate
export PYTHONPATH=/path/to/Fractal
cd tests/
pytest -vvs
```
2. Update library version in `setup.py`
3. Generate new docs (more info in docs/README.md)
4. Deploy package at PyPi:
```
pip install setuptools
python setup.py sdist bdist_wheel
pip install twine
twine upload dist/*
```
