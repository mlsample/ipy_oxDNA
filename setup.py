from setuptools import setup, find_packages

setup(
    name='ipy_oxdna',
    version='0.21',
    packages=find_packages(),
    description='Enables the use of oxDNA in Jupyter notebooks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mlsample/ipy_oxDNA',
    author='Matthew Sample',
    author_email='matsample1@gmail.com',
    license='MIT',
    install_requires=[
    'numpy',
    'scipy',
    'matplotlib',
    'pymbar[jax]',
    'nvidia-ml-py3',
    'py',
    'scienceplots',
    'ipywidgets',
    'ipykernel',
    'pandas',
    'statsmodels',
    'tqdm',
    'pyarrow',
    'pytest'
    # ... other dependencies ...
    ],
    # dependencies can be listed under install_requires
)
