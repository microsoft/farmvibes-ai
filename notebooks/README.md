# FarmVibes.AI Example Notebooks

The notebooks presented here serve as examples on the how to get started, and some of the capabilities of FarmVibes.AI.
We recommend installing the required packages to run the notebooks within a conda environment.

## Installing and using conda

Please see the instalation instructions in the [Anaconda website](https://docs.conda.io/en/latest/miniconda.html).
Also check [instructions for getting started with conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) and the [conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) .

## Installing the environment

A set of common packages that should enable you to several notebooks is present in the environment described in `env.yml`. To create the environment, run

```bash
conda env create -f env.yml
```

This will create an environment named `farmvibes-ai`.
Notebooks that require other packages, such as machine learning frameworks, also have specific
environment files in their respective directories.
