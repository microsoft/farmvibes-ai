# data_ingestion/weather/download_terraclimate

Monthly climate and hydroclimate properties from TerraClimate. The workflow downloads weather and hydrological data for the input time range.  Data is available for global terrestrial surfaces from 1958-present, with a monthly  temporal resolution and a ~4-km (1/24th degree) spatial resolution.

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

## Sources

- **user_input**: Time range of interest.

## Sinks

- **downloaded_product**: Downloaded variable for each year in the input time range.

## Parameters

- **variable**: Options are:
  aet - Actual Evapotranspiration (monthly total, units = mm)
  def - Climate Water Deficit (monthly total, units = mm)
  pet - Potential evapotranspiration (monthly total, units = mm)
  ppt - Precipitation (monthly total, units = mm)
  q - Runoff (monthly total, units = mm)
  soil - Soil Moisture (total column at end of month, units = mm)
  srad - Downward surface shortwave radiation (units = W/m2)
  swe - Snow water equivalent (at end of month, units = mm)
  tmax - Max Temperature (average for month, units = C)
  tmin - Min Temperature (average for month, units = C)
  vap - Vapor pressure (average for month, units = kPa)
  ws - Wind speed (average for month, units = m/s)
  vpd - Vapor Pressure Deficit (average for month, units = kPa)
  PDSI - Palmer Drought Severity Index (at end of month, units = unitless)

## Tasks

- **list**: Lists TerraClimate products of `variable` from years intersecting with input time range.

- **download**: Downloads Climatology Lab weather products (TerraClimate and GridMET) defined by the input product.

## Workflow Yaml

```yaml

name: download_terraclimate
sources:
  user_input:
  - list.input_item
sinks:
  downloaded_product: download.downloaded_product
parameters:
  variable: tmax
tasks:
  list:
    op: list_terraclimate
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
  short_description: Monthly climate and hydroclimate properties from TerraClimate.
  long_description: The workflow downloads weather and hydrological data for the input
    time range.  Data is available for global terrestrial surfaces from 1958-present,
    with a monthly  temporal resolution and a ~4-km (1/24th degree) spatial resolution.
  sources:
    user_input: Time range of interest.
  sinks:
    downloaded_product: Downloaded variable for each year in the input time range.
  parameters:
    variable: "Options are:\n  aet - Actual Evapotranspiration (monthly total, units\
      \ = mm)\n  def - Climate Water Deficit (monthly total, units = mm)\n  pet -\
      \ Potential evapotranspiration (monthly total, units = mm)\n  ppt - Precipitation\
      \ (monthly total, units = mm)\n  q - Runoff (monthly total, units = mm)\n  soil\
      \ - Soil Moisture (total column at end of month, units = mm)\n  srad - Downward\
      \ surface shortwave radiation (units = W/m2)\n  swe - Snow water equivalent\
      \ (at end of month, units = mm)\n  tmax - Max Temperature (average for month,\
      \ units = C)\n  tmin - Min Temperature (average for month, units = C)\n  vap\
      \ - Vapor pressure (average for month, units = kPa)\n  ws - Wind speed (average\
      \ for month, units = m/s)\n  vpd - Vapor Pressure Deficit (average for month,\
      \ units = kPa)\n  PDSI - Palmer Drought Severity Index (at end of month, units\
      \ = unitless)"


```