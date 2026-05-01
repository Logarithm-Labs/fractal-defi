from pathlib import Path

from setuptools import find_packages, setup

# Runtime deps installed for end users; dev/lint/test tooling lives in the [dev] extra.
RUNTIME_REQUIRES = [
    "mlflow>=3.11.1",
    "pandas>=2.3.3",
    "numpy>=2.2.6",
    "loguru>=0.7.3",
    "requests>=2.33.1",
    "scikit-learn>=1.7.2",
]

DEV_REQUIRES = [
    "pytest>=9.0.3",
    "pytest-timeout>=2.4.0",
    "pylint>=4.0.5",
    "flake8>=7.3.0",
    "isort>=8.0.1",
    "pre-commit>=4.6.0",
    "sphinx>=8.1.3",
    "sphinx-rtd-theme>=3.1.0",
    # Release tooling — used by ``make build`` / ``make release`` / smoke harness.
    "build>=1.2.0",
    "twine>=5.0.0",
]

setup(
    name='fractal-defi',
    version='1.3.1',
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
    license="BSD-3-Clause",
    python_requires=">=3.10, <3.14",
    install_requires=RUNTIME_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "test": ["pytest>=9.0.3", "pytest-timeout>=2.4.0"],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
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
