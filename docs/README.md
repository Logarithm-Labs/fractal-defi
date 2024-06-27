# How to create docs?
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cd docs
rm -rf source
mkdir source
sphinx-apidoc -o source ../fractal/
nano index.rst
make html
```
Then you can deploy `/build`
