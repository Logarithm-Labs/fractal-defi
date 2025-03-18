# How to create docs?
```
cd docs
pip install sphinx
pip install sphinx-rtd-theme
sphinx-apidoc -o source ../fractal/
make html
```
Then you can deploy `/build`
