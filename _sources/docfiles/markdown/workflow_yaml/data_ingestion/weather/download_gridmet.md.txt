# data_ingestion/weather/download_gridmet

```yaml

name: download_gridmet
sources:
  user_input:
  - list.input_item
sinks:
  downloaded_product: download.downloaded_product
parameters:
  variable: pet
tasks:
  list:
    op: list_gridmet
    op_dir: list_climatology_lab
    parameters:
      variable: '@from(variable)'
  download:
    op: download_climatology_lab
edges:
- origin: list.products
  destination:
  - download.input_product
description:
  short_description: Daily surface meteorological properties from GridMET.
  long_description: The workflow downloads weather and hydrological data for the input
    time range.  Data is available for the contiguous US and southern British Columbia
    surfaces from 1979-present, with a daily temporal resolution and a ~4-km (1/24th
    degree) spatial resolution.
  sources:
    user_input: Time range of interest.
  sinks:
    downloaded_product: Downloaded variable for each year in the input time range.
  parameters:
    variable: "Options are:\n  bi - Burning Index\n  erc - Energy Release Component\n\
      \  etr - Daily reference evapotranspiration (alfafa, units = mm)\n  fm100 -\
      \ Fuel Moisture (100-hr, units = %)\n  fm1000 - Fuel Moisture (1000-hr, units\
      \ = %)\n  pet - Potential evapotranspiration (reference grass evapotranspiration,\
      \ units = mm)\n  pr - Precipitation amount (daily total, units = mm)\n  rmax\
      \ - Maximum relative humidity (units = %)\n  rmin - Minimum relative humidity\
      \ (units = %)\n  sph - Specific humididy (units = kg/kg)\n  srad - Downward\
      \ surface shortwave radiation (units = W/m^2)\n  th - Wind direction (degrees\
      \ clockwise from North)\n  tmmn - Minimum temperature (units = K)\n  tmmx -\
      \ Maximum temperature (units = K)\n  vpd - Vapor Pressure Deficit (units = kPa)\n\
      \  vs - Wind speed at 10m (units = m/s)"


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>downloaded_product]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- products/input_product --> tsk2{{download}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    tsk2{{download}} -- downloaded_product --> out1>downloaded_product]
```