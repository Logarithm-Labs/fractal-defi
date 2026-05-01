# Release Notes
1. Update library version in `setup.py`
2. Last smoke local run
```
git checkout main && git pull origin main
make clean
make smoke
```
3. Release to TestPyPi and test
```bash
make release-test
python3 -m venv /tmp/ck && /tmp/ck/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    fractal-defi==1.4.0 && rm -rf /tmp/ck
```
4. Release to PyPi
```bash
make release
```
5. Create tag and release in repo or run:
```bash
git tag -a v1.4.0 -m "v1.4.0"
git push origin v1.4.0

gh release create v1.4.0 --generate-notes \
    dist/fractal_defi-1.4.0-py3-none-any.whl \
    dist/fractal_defi-1.4.0.tar.gz
```
6. After squash and merge PR run:
```bash
git checkout dev
make post-release
```
