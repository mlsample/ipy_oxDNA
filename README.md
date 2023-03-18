# ipy_oxDNA
<center>
<img src="./src/oxDNA.png">
</center>

<!--
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
-->
This repository contains python code for running oxDNA umbrella sampling and large throughput simulations. This code is complementary to the article

"Hairygami: Analysis of DNA Nanostructure's Conformational Change Driven by Functionalizable Overhangs"

Matthew Sample, Michael Matthies, and Petr Sulc

Within the src folder exists Jupyter notebook tutorials and examples.

## Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Example Notebooks](#example-notebooks)
- [Contributing](#contributing)
- [Citation](#citation)

## Introduction
oxDNA is a molecular dynamics simulation code that can be used to study the mechanical and thermodynamic properties of DNA and RNA molecules. Umbrella sampling is a highly paralizable simulation technique that is used to calculate the free energy profiles between two particles or groups of particles. The `ipy_oxDNA` repository provides a python interface for running oxDNA umbrella sampling simulations, allowing users to easily perform these simulations and analyze their results.

## Prerequisites
Before using this code, you will need to have the following installed:
- conda/mamba environment with packages specified in `./oxdna.yml` (`conda env create -f oxdna.yml`)
- oxDNA installed with python bindings (https://github.com/lorenzo-rovigatti/oxDNA)
- Python >= 3.8

## Installation
To install the `ipy_oxDNA` code, clone the repository to your local machine:

`git clone https://github.com/mlsample/ipy_oxDNA.git`

Install the Weight Histogram Analysis Technique (http://membrane.urmc.rochester.edu/?page_id=126)

`chmod +x ./src/install_wham.sh`

`./src/install_wham.sh`


## Usage
The code can be used by importing the necessary modules into your python script or Jupyter Notebook. A tutorial on how to use the code can be found in the `src/ipy_oxdna_examples.ipynb`.

Furthermore, if you wish to run umbrella sampling as a python script an example can be found in `./src/full_umbrella.py`

## Example Notebooks
This repository includes a number of example notebooks that demonstrate how to use the code to perform oxDNA umbrella sampling simulations. These notebooks can be used as a starting point for your own simulations.

`./src/ipy_oxdna_example.ipynb`
- Running a single simulation or multiple simulations in parallel
- Examples for running single umbrella sampling simulation or multiple in parallel

`./src/double_layer.ipynb`
- Production umbrella sampling of the anti-parallel double layer strucutre
- Benchamrking the speedup from varying number of simulations in parallel for umbrella sampling

`./src/duplex_example.ipynb`
- Example of using umbrella sampling for duplex melting

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## Citation
If you use this code in your research, please cite the accompanying academic article that will be published soon.
<!--
## License
%This code is licensed under the MIT license. See [LICENSE](LICENSE) for details.
-->

