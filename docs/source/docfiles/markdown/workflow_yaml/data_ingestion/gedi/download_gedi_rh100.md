# data_ingestion/gedi/download_gedi_rh100

Downloads L2B GEDI products and extracts RH100 variables. The workflow will download the products for the input region and time range, and then extract RH100 variables for each of the beam shots. Each value is geolocated according to the lowest mode latitude and longitude values.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>rh100]
    tsk1{{download}}
    tsk2{{extract}}
    tsk1{{download}} -- product/gedi_product --> tsk2{{extract}}
    inp1>user_input] -- user_input --> tsk1{{download}}
    inp1>user_input] -- roi --> tsk2{{extract}}
    tsk2{{extract}} -- rh100 --> out1>rh100]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **rh100**: Points in EPSG:4326 with their associated RH100 values.

## Parameters

- **earthdata_token**: API token for the EarthData platform. Required to run the workflow.

- **check_quality**: Whether to filter points according to the quality flag.

## Tasks

- **download**: Downloads GEDI products for the input region and time range.

- **extract**: Extracts RH100 variables within the region of interest of a GEDIProduct.

## Workflow Yaml

```yaml

name: download_gedi_rh100
sources:
  user_input:
  - download.user_input
  - extract.roi
sinks:
  rh100: extract.rh100
parameters:
  earthdata_token: null
  check_quality: null
tasks:
  download:
    workflow: data_ingestion/gedi/download_gedi
    parameters:
      earthdata_token: '@from(earthdata_token)'
  extract:
    op: extract_gedi_rh100
    parameters:
      check_quality: '@from(check_quality)'
edges:
- origin: download.product
  destination:
  - extract.gedi_product
description:
  short_description: Downloads L2B GEDI products and extracts RH100 variables.
  long_description: The workflow will download the products for the input region and
    time range, and then extract RH100 variables for each of the beam shots. Each
    value is geolocated according to the lowest mode latitude and longitude values.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    rh100: Points in EPSG:4326 with their associated RH100 values.
  parameters:
    check_quality: Whether to filter points according to the quality flag.


```