# ipy_oxDNA

<!--
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
-->
This repository contains python code for running oxDNA umbrella sampling and large throuput simulations. This code is complementary to the article

"Hairygami: Analysis of DNA Nanostructure's Conformational Change Driven by Functionalizable Overhangs"

Matthew Sample, Michael Matthies, and Petr Sulc

Within the src folder exist Jupyter notebook tutorials and examples.

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
-  3.8 <= Python <= 3.10 
- oxDNA installed with python bindings (and the pakages that come with them)
- conda/mamba envirment with pakages specified in `./ipy_oxdna.yml` (conda env create -f ipy_oxdna.yml)


## Installation
To install the `ipy_oxDNA` code, clone the repository to your local machine:
`git clone https://github.com/mlsample/ipy_oxDNA.git`

Install the Weight Histogram Analysis Technique (http://membrane.urmc.rochester.edu/?page_id=126)
`chmod +x ./src/install_wham.sh`
`./scr/install_wham.sh`


## Usage
The code can be used by importing the necessary modules into your python script or Jupyter Notebook. A tutorial on how to use the code can be found in the `scr/ipy_oxdna_examples.ipynb`.

## Example Notebooks
This repository includes a number of example notebooks that demonstrate how to use the code to perform oxDNA umbrella sampling simulations. These notebooks can be used as a starting point for your own simulations.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## Citation
If you use this code in your research, please cite the accompanying academic article that will be published soon.
<!--
## License
%This code is licensed under the MIT license. See [LICENSE](LICENSE) for details.
-->

