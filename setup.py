from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line and not line.startswith('#')]


setup(
    name='fractal-defi',
    version='1.0.2',
    packages=find_packages(),
    author='Logarithm Labs',
    author_email='dev@logarithm.fi',
    description='An ultimate DeFi research library for strategy development and fractaling.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Logarithm-Labs/Fractal',
    include_package_data=True,
    python_requires=">=3.8, <4",
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
