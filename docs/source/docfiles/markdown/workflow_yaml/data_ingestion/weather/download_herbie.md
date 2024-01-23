# data_ingestion/weather/download_herbie

```yaml

name: download_herbie
sources:
  user_input:
  - list_herbie.input_item
sinks:
  forecast: download_herbie.forecast
parameters:
  model: hrrr
  product: null
  frequency: 1
  forecast_lead_times:
  - 0
  - 1
  - 1
  search_text: :TMP:2 m
tasks:
  list_herbie:
    op: list_herbie
    parameters:
      model: '@from(model)'
      product: '@from(product)'
      frequency: '@from(frequency)'
      forecast_lead_times: '@from(forecast_lead_times)'
      search_text: '@from(search_text)'
  download_herbie:
    op: download_herbie
edges:
- origin: list_herbie.product
  destination:
  - download_herbie.herbie_product
description:
  short_description: Downloads forecast data for provided location & time range using
    herbie python package.
  long_description: Herbie is a python package that downloads recent and archived
    numerical weather prediction (NWP) model outputs from different cloud archive
    sources. Its most popular capability is to download HRRR model data. NWP data
    in GRIB2 format can be read with xarray+cfgrib. Model data Herbie can retrieve
    includes the High Resolution Rapid Refresh (HRRR), Rapid Refresh (RAP), Global
    Forecast System (GFS), National Blend of Models (NBM), Rapid Refresh Forecast
    System - Prototype (RRFS), and ECMWF open data forecast products (ECMWF).
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    forecast: Grib file with the requested forecast.
  parameters:
    model: Model name as defined in the models template folder. CASE INSENSITIVE Below
      are examples of model types 'hrrr' HRRR contiguous United States model 'hrrrak'
      HRRR Alaska model (alias 'alaska') 'rap' RAP model 'gfs' Global Forecast System
      (atmosphere) 'gfs_wave' Global Forecast System (wave) 'rrfs' Rapid Refresh Forecast
      System prototype for more information see https://herbie.readthedocs.io/en/latest/user_guide/model_info.html
    product: Output variable product file type (sfc (surface fields), prs (pressure
      fields), nat (native fields), subh (subhourly fields)). Not specifying this
      will use the first product in model template file.
    frequency: frequency in hours of the forecast
    forecast_lead_times: Forecast lead time in the format [start_time, end_time, increment]
      (in hours)
    search_text: It's a regular expression used to search on GRIB2 Index files and
      allow you to download just the layer of the file required instead of complete
      file. For more information on search_text refer to below url. https://blaylockbk.github.io/Herbie/_build/html/user_guide/searchString.html


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>forecast]
    tsk1{{list_herbie}}
    tsk2{{download_herbie}}
    tsk1{{list_herbie}} -- product/herbie_product --> tsk2{{download_herbie}}
    inp1>user_input] -- input_item --> tsk1{{list_herbie}}
    tsk2{{download_herbie}} -- forecast --> out1>forecast]
```