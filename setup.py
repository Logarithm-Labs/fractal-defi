from setuptools import setup, find_packages


setup(
    name='fractal-defi',
    version='1.0.0',
    packages=find_packages(),
    author='Logarithm Labs',
    author_email='dev@logarithm.fi',
    description='An ultimate DeFi research library for strategy development and fractaling.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Logarithm-Labs/Fractal',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
