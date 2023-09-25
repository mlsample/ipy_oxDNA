.. _ipy_oxdna documentation:

===============================
Welcome to ipy_oxDNA's Documentation
===============================

 .. toctree::
    :maxdepth: 2
    :caption: Table of Contents

    Introduction <self>
    modules


Introduction
------------


`ipy_oxDNA` is a Python interface for running oxDNA umbrella sampling and large throughput simulations. This code is complementary to the article, if you use this code a citation would be appreciated:

- Sample, Matthew, Michael Matthies, and Petr Šulc. "Hairygami: Analysis of DNA Nanostructure's Conformational Change Driven by Functionalizable Overhangs." arXiv preprint arXiv:2302.09109 (2023).

For more tutorials and examples, please refer to the `src` folder within this repository.

NVIDIA Multiprocessing Service (MPS)
------------------------------------
 .. code-block:: bash

   export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
   export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
   mkdir -p $CUDA_MPS_PIPE_DIRECTORY
   mkdir -p $CUDA_MPS_LOG_DIRECTORY
   nvidia-cuda-mps-control -d

NVIDIA MPS enhances the multiprocessing capabilities of CUDA-enabled GPUs

Prerequisites
-------------

- Docker image available on `Dockerhub <https://hub.docker.com/repository/docker/mlsample/ipy_oxdna/general>`_.
 .. code-block:: bash
   
   docker pull mlsample/ipy_oxdna:v0.2 
   docker run -p 8888:8888 mlsample/ipy_oxdna:v0.2 jupyter lab --ip 0.0.0.0 --allow-root -v $(pwd):/workspace --notebook-dir=/workspace

or

- Conda/mamba environment with packages specified in `oxdna.yml`.
- oxDNA installed with Python bindings.
- Python >= 3.8

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
