from pathlib import Path

from setuptools import find_packages, setup

# Runtime dependencies — what users of the library actually need at import
# time. Dev tooling (pytest, pytest-timeout, pylint, flake8) used to be in
# ``requirements.txt`` AND ``install_requires``, which forced every consumer
# to install the lint stack. Now they live under the ``[dev]`` extra.
RUNTIME_REQUIRES = [
    "mlflow>=2.14.1",
    "pandas>=2.2.2",
    "numpy>=1.16.0",
    "loguru>=0.7.2",
    "requests>=2.32.3",
    "scikit-learn>=1.5.0",
]

DEV_REQUIRES = [
    "pytest>=8.2.2",
    "pytest-timeout>=2.3.0",
    "pylint>=3.2.5",
    "flake8>=7.1.0",
]

setup(
    name='fractal-defi',
    version='1.3.0',
    packages=find_packages(),
    author='Logarithm Labs',
    author_email='dev@logarithm.fi',
    description=(
        'Fractal is the ultimate DeFi research library for strategies '
        'development and backtesting created by Logarithm Labs.'
    ),
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/Logarithm-Labs/Fractal',
    include_package_data=True,
    python_requires=">=3.10, <3.13",
    install_requires=RUNTIME_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "test": ["pytest>=8.2.2", "pytest-timeout>=2.3.0"],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
