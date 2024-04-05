# data_ingestion/weather/download_era5_monthly

Monthly estimated weather variables. Monthly weather variables obtained from combining observations and numerical model runs to estimate the state of the atmosphere.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>downloaded_product]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- era5_products/era5_product --> tsk2{{download}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    tsk2{{download}} -- downloaded_product --> out1>downloaded_product]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **downloaded_product**: 30km resolution weather variables.

## Parameters

- **cds_api_key**: api key for Copernicus CDS (https://cds.climate.copernicus.eu/user/register)

- **variable**: Options are:
  2t - 2 meter temperature (default)
  100u - 100 meter U wind component
  100v - 100 meter V wind component
  10u - 10 meter U wind component
  10v - 10 meter V wind component
  2d - 2 meter dewpoint temperature
  msl - Mean sea level pressure
  sp - Surface pressure
  ssrd - Surface solar radiation downwards
  sst - Sea surface temperature
  tp - Total precipitation

## Tasks

- **list**: Lists monthly ERA5 products for the input time range and geometry.

- **download**: Downloads requested property from ERA5 products.

## Workflow Yaml

```yaml

name: download_era5_monthly
sources:
  user_input:
  - list.input_item
sinks:
  downloaded_product: download.downloaded_product
parameters:
  cds_api_key: null
  variable: 2t
tasks:
  list:
    op: list_era5_cds
    op_dir: list_era5
    parameters:
      variable: '@from(variable)'
  download:
    op: download_era5
    parameters:
      api_key: '@from(cds_api_key)'
edges:
- origin: list.era5_products
  destination:
  - download.era5_product
description:
  short_description: Monthly estimated weather variables.
  long_description: Monthly weather variables obtained from combining observations
    and numerical model runs to estimate the state of the atmosphere.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    downloaded_product: 30km resolution weather variables.
  parameters:
    cds_api_key: api key for Copernicus CDS (https://cds.climate.copernicus.eu/user/register)
    variable: "Options are:\n  2t - 2 meter temperature (default)\n  100u - 100 meter\
      \ U wind component\n  100v - 100 meter V wind component\n  10u - 10 meter U\
      \ wind component\n  10v - 10 meter V wind component\n  2d - 2 meter dewpoint\
      \ temperature\n  msl - Mean sea level pressure\n  sp - Surface pressure\n  ssrd\
      \ - Surface solar radiation downwards\n  sst - Sea surface temperature\n  tp\
      \ - Total precipitation"


```