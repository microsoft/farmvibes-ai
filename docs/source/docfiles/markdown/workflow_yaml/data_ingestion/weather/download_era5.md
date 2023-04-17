# data_ingestion/weather/download_era5

```yaml

name: download_era5
sources:
  user_input:
  - list.input_item
sinks:
  downloaded_product: download.downloaded_product
parameters:
  pc_key: null
  variable: 2t
tasks:
  list:
    op: list_era5
    parameters:
      variable: '@from(variable)'
  download:
    op: download_era5
    parameters:
      api_key: '@from(pc_key)'
edges:
- origin: list.era5_products
  destination:
  - download.era5_product
description:
  short_description: Hourly estimated weather variables.
  long_description: Hourly weather variables obtained from combining observations
    and numerical model runs to estimate the state of the atmosphere.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    downloaded_product: 30km resolution weather variables.
  parameters:
    pc_key: Optional Planetary Computer API key.
    variable: "Options are:\n  2t - 2 meter temperature (default)\n  100u - 100 meter\
      \ U wind component\n  100v - 100 meter V wind component\n  10u - 10 meter U\
      \ wind component\n  10v - 10 meter V wind component\n  2d - 2 meter dewpoint\
      \ temperature\n  mn2t - Minimum temperature at 2 meters since previous post-processing\n\
      \  msl - Mean sea level pressure\n  mx2t - Maximum temperature at 2 meters since\
      \ previous post-processing\n  sp - Surface pressure\n  ssrd - Surface solar\
      \ radiation downwards\n  sst - Sea surface temperature\n  tp - Total precipitation"


```

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