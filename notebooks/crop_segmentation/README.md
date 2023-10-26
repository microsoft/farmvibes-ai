# FarmVibes.AI - Crop Land Segmentation

This folder contains example notebooks and associated code for training a model to segment crop land areas leveraging the FarmVibes.AI platform.

## Conda Environment

Before running the notebooks, make sure to set up the micromamba environment with necessary dependencies. If you do not have micromamba installed, please follow the instructions from the [micromamba installation page](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

```
micromamba env create -f ./crop_env.yaml
micromamba activate crop-seg
```

## Notebooks Description

The example is divided in individual tasks:

- [Dataset generation](./01_dataset_generation.ipynb): invoke a FarmVibes.AI workflow to generate the dataset for training;
- [Dataset exploration](./02_visualize_dataset.ipynb): explore the generated data, visualizing intermediate outputs from the dataset generation workflow;
- Model training:
  - [Local training](./03_local_training.ipynb): train a segmentation model locally using PyTorch;
  - [Azure Machine Learning training](./03_aml_training.ipynb): upload dataset to AML and train the segmentation model;
- [Inference](./04_inference.ipynb): employ the trained model to predict crop land segmentation maps for new regions;
