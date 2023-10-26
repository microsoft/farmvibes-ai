# FarmVibes.AI Example Notebooks

The notebooks presented here serve as examples on the how to get started, and some of the capabilities of FarmVibes.AI.
We recommend installing the required packages to run the notebooks within a micromamba environment.

## Installing and using micromamba

Please see the instalation instructions in the [micromamba website](https://mamba.readthedocs.io/en/latest/).
Also check the [micromamba user guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) and the [troubleshoooting page](https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html).

## Installing the environment

A set of common packages that should enable you to several notebooks is present in the environment described in `env.yaml`. To create the environment, run

```bash
micromamba env create -f env.yaml
```

This will create an environment named `farmvibes-ai`.
Notebooks that require other packages, such as machine learning frameworks, also have specific
environment files in their respective directories.
