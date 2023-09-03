.. _ipy_oxdna documentation:

===============================
Welcome to ipy_oxDNA's Documentation
===============================

.. image:: ./src/oxDNA.png
   :align: center

Introduction
------------

`ipy_oxDNA` is a Python interface for running oxDNA umbrella sampling and large throughput simulations. This code is complementary to the article:

- Sample, Matthew, Michael Matthies, and Petr Šulc. "Hairygami: Analysis of DNA Nanostructure's Conformational Change Driven by Functionalizable Overhangs." arXiv preprint arXiv:2302.09109 (2023).

For more tutorials and examples, please refer to the `src` folder within this repository.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   Introduction <self>
   modules

NVIDIA Multiprocessing Service (MPS)
------------------------------------

NVIDIA MPS enhances the multiprocessing capabilities of CUDA-enabled GPUs, offering approximately **2.5x performance increase** for specific simulation techniques. For more details, refer to the `How to run NVIDIA Multiprocessing Service (MPS)`_ section.

Prerequisites
-------------

- Docker image available on `Dockerhub <https://hub.docker.com/repository/docker/mlsample/ipy_oxdna/general>`_.
- Conda/mamba environment with packages specified in `oxdna.yml`.
- oxDNA installed with Python bindings.
- Python >= 3.8

Installation
------------

To install, clone the repository and follow the installation steps in the `Installation`_ section.

Usage
-----

Refer to the `Usage`_ section for a tutorial and example notebooks.

Contributing
------------

Contributions are welcome! Please fork the repository and submit a pull request.

Citation
--------

If you use this code in your research, please cite the accompanying academic article that will be published soon.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
