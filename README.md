# ipy_oxDNA
<center>
<img src="oxDNA.png">
</center>

This repository contains Python code for running oxDNA umbrella sampling and large throughput simulations. This code is complementary to the article:

Sample, Matthew, Michael Matthies, and Petr Šulc. "Hairygami: Analysis of DNA Nanostructure's Conformational Change Driven by Functionalizable Overhangs." arXiv preprint arXiv:2302.09109 (2023).

Within the `examples/tutorials` folder exist Jupyter notebook tutorials and examples. The full documentation can be found [here](https://mlsample.github.io/ipy_oxDNA/index.html).

## Contents
- [Introduction](#introduction)
- [NVIDIA Multiprocessing Service (mps)](#how-to-run-nvidia-multiprocessing-service-mps)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Example Notebooks](#example-notebooks)
- [Contributing](#contributing)
- [Citation](#citation)

## Introduction
oxDNA is a molecular dynamics simulation code that can be used to study the mechanical and thermodynamic properties of DNA and RNA molecules. Umbrella sampling is a highly parallelizable simulation technique that is used to calculate the free energy profiles between two particles or groups of particles. The `ipy_oxDNA` repository provides a Python interface for running oxDNA umbrella sampling simulations, allowing users to easily perform these simulations and analyze their results.

## How to run NVIDIA Multiprocessing Service (mps)
NVIDIA MPS is a specialized service offered by NVIDIA, designed to enhance the multiprocessing capabilities of CUDA-enabled GPUs. Utilizing MPS allows for the execution of multiple simulations on a single GPU, resulting in an approximate **2.5x performance increase** for specific simulation techniques such as *Umbrella Sampling, Metadynamics, and Multi-Replica simulations*.

In the absence of MPS, running concurrent simulations on a single GPU can lead to significant performance degradation. For comprehensive details on MPS, the official documentation is available at [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html).

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
```

## Prerequisites
Before using this code, you will need to have the following installed:

- Python >= 3.8
- oxDNA installed with Python bindings (https://github.com/lorenzo-rovigatti/oxDNA)

## Installation
To install the `ipy_oxDNA` code, clone the repository to your local machine:

`git clone https://github.com/mlsample/ipy_oxDNA.git`

Install the package using:
`pip install .`

Install the Weight Histogram Analysis Technique (http://membrane.urmc.rochester.edu/?page_id=126)

`chmod +x ./src/install_wham.sh`

`./src/install_wham.sh`

## Example Notebooks
This repository includes a number of example notebooks that demonstrate how to use the code to perform oxDNA umbrella sampling simulations. These notebooks can be used as a starting point for your own simulations.

`/examples/tutorials/origami_tile_N_nt_overhangs_gpu`
- Umbrella sampling of the no overhang, 9nt overhangs, and 20nt overhang extension twist-corrected rectangular DNA origami

`./examples/tutorials/8_nt_duplex_melting_cpu`
- Example of using umbrella sampling for duplex melting

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## Citation
If you use this code in your research, please cite the accompanying academic article:

Sample, Matthew, Michael Matthies, and Petr Šulc. “Hairygami: Analysis of DNA Nanostructure’s Conformational Change Driven by Functionalizable Overhangs.” arXiv preprint arXiv:2302.09109 (2023).