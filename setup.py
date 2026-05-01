from pathlib import Path

from setuptools import find_packages, setup

# Runtime dependencies — what users of the library actually need at import
# time. Dev tooling (pytest, pytest-timeout, pylint, flake8) used to be in
# ``requirements.txt`` AND ``install_requires``, which forced every consumer
# to install the lint stack. Now they live under the ``[dev]`` extra.
RUNTIME_REQUIRES = [
    "mlflow>=2.14.1",
    "pandas>=2.2.2",
    # ``numpy>=1.26.0`` covers the Python 3.10–3.13 matrix without
    # upper-bound thrashing. The previous ``numpy<2,>=1.16.0`` pin in
    # v1.1.0 forced pip onto numpy 1.26.4 — which has no Python 3.13
    # wheel and required a C compiler at install time.
    "numpy>=1.26.0",
    "loguru>=0.7.2",
    "requests>=2.32.3",
    "scikit-learn>=1.5.0",
]

DEV_REQUIRES = [
    "pytest>=8.2.2",
    "pytest-timeout>=2.3.0",
    "pylint>=3.2.5",
    "flake8>=7.1.0",
    "isort>=5.13.0",
    "pre-commit>=3.7.0",
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
]

setup(
    name='fractal-defi',
    version='1.3.0',
    packages=find_packages(),
    author='Logarithm Labs',
    author_email='xboringwozniak@gmail.com',
    description=(
        'Open-source Python research library for DeFi strategies. '
        'Compose protocol-agnostic entities (lending, perps, DEX and LP) '
        'into typed strategies; backtest, simulate, track experiments.'
    ),
    keywords=[
        'defi', 'backtesting', 'research', 'uniswap', 'uniswap-v3', 'aave',
        'hyperliquid', 'gmx', 'binance', 'mlflow', 'algorithmic-trading',
        'quantitative-finance', 'delta-neutral', 'basis-trade',
        'liquidity-provision', 'yield-farming', 'agentic-ai',
        'monte-carlo', 'protocol-agnostic', 'composable',
    ],
    project_urls={
        'Source': 'https://github.com/Logarithm-Labs/fractal-defi',
        'Changelog': 'https://github.com/Logarithm-Labs/fractal-defi/blob/main/CHANGELOG.md',
        'Issues': 'https://github.com/Logarithm-Labs/fractal-defi/issues',
    },
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/Logarithm-Labs/fractal-defi',
    include_package_data=True,
    python_requires=">=3.10, <3.14",
    install_requires=RUNTIME_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "test": ["pytest>=8.2.2", "pytest-timeout>=2.3.0"],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
