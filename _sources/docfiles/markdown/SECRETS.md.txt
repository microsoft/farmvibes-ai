# Secrets

Operations in FarmVibes.AI can retrieve secrets to use as parameters, which can be useful to avoid
storing secrets in plain-text. Secrets are stored safely within the Kubernetes cluster and are not
transmited or visible outside the VM. For more information on how secrets within Kubernetes, refer
to [Kubernetes documentation](https://kubernetes.io/docs/concepts/configuration/secret/).

Secrets may be added to the cluster through the ```add-secret``` command of the *farmvibes-ai*
command. The secret can then be passed as parameters to the workflow yaml files.

This document details how to add or delete a secret to the cluster (both local or remote), as well as lists all workflows
that require a secret.

## Adding a secret to FarmVibes.AI cluster

To add a secret with a key `<key>` and value `<value>`, run:

```bash
farmvibes-ai <local | remote> add-secret <key> <value>
```

Note that secrets are not persisted when clusters are destroyed and must be added again after each setup.

## Using a secret within a workflow

Secrets are used in a workflow with the @SECRET notation. For example,
`@SECRET(my-keyvault-name, my-secret-key)` in which `my-secret-key` is the key and
`my-keyvault-name` is the key-vault. For local FarmVibes.AI instalation, the key-vault can be any non-empty string.

The following workflow yaml shows an example of an exposed secret parameter (`download_password`) with a default key (`my-secret-pass`):

```yaml
name: my_test_wf
sources:
  input_a:
    - download.input
sinks:
  output_b: download.output
parameters:
  download_password: "@SECRET(my-keyvault-name, my-secret-pass)"
tasks:
  download:
    op: my_exemple_op
    parameters:
      password: "@from(download_password)"
edges:
description:
  short_description:
    Example workflow.
  long_description:
    Requires secret from parameter download_password.
    Default secret key is my-secret-pass.
  sources:
    input_a: Example input.
  sinks:
    output_b: Example output.
  parameters:
    download_password: Download password secret.
```

## Deleting a secret to FarmVibes.AI cluster

The following command can be used to delete a secret from the cluster:

```bash
farmvibes-ai <local | remote> delete-secret <key>
```

## List of workflows and their associated secrets

- **Azure Data Manager for Agriculture (ADMAG) client secret** (parameter `client_secret`).
  - `data_ingestion/admag/admag_seasonal_field`
  - `data_ingestion/admag/prescriptions`

- **Ambient Weather API key** (parameter `api-key` with default secret key `ambient-api-key`) and **App key** (parameter `app-key` with default secret key `ambient-app-key`).
  - `data_ingestion/weather/get_ambient_weather`

- **EarthData API token** (parameter `earthdata_token` with default secret key `earthdata-token`).
  - `data_ingestion/gedi/download_gedi`
  - `data_ingestion/gedi/download_gedi_rh100`

- **NOAA GFS SAS token** (parameter `noaa_gfs_token` with default secret key `noaa-gfs-sas`).
  - `data_ingestion/weather/get_forecast`

- **SciHub username** and **password** (parameters `scihub_user` and `scihub_password`, and default secret keys `scihub-user` and `scihub-password`, respectively).
  - `data_ingestion/sentinel1/preprocess_s1`

- **Planetary computer API key**. By default, FarmVibes.AI workflows access the Planetary Computer catalog anonymously, when possible. However, we recommend registering for an API key [(see more information here)](https://planetarycomputer.microsoft.com/docs/overview/about/) to avoid being throttled.
  - `data_ingestion/dem/download_dem`
  - `data_ingestion/landsat/preprocess_landsat`
  - `data_ingestion/naip/download_naip`
  - `data_ingestion/sentinel1/preprocess_s1`
  - `data_ingestion/sentinel2/preprocess_s2`
  - `data_ingestion/sentinel2/preprocess_s2_improved_mask`
  - `data_ingestion/spaceeye/spaceeye`
  - `data_ingestion/spaceeye/spaceeye_interpolation`
  - `data_ingestion/spaceeye/spaceeye_preprocess`
  - `farm_ai/agriculture/canopy_cover`
  - `farm_ai/agriculture/change_detection`
  - `farm_ai/agriculture/emergence_summary`
  - `farm_ai/agriculture/methane_index`
  - `farm_ai/agriculture/ndvi_summary`
  - `farm_ai/agriculture/conservation_practices`
  - `farm_ai/agriculture/landsat_ndvi_trend`
  - `ml/dataset_generation/datagen_crop_segmentation`

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
```
