# data_ingestion/weather/download_herbie

Downloads forecast data for provided location & time range using herbie python package. Herbie is a python package that downloads recent and archived numerical weather prediction (NWP) model outputs from different cloud archive sources. Its most popular capability is to download HRRR model data. NWP data in GRIB2 format can be read with xarray+cfgrib. Model data Herbie can retrieve includes the High Resolution Rapid Refresh (HRRR), Rapid Refresh (RAP), Global Forecast System (GFS), National Blend of Models (NBM), Rapid Refresh Forecast System - Prototype (RRFS), and ECMWF open data forecast products (ECMWF).

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

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **forecast**: Grib file with the requested forecast.

## Parameters

- **model**: Model name as defined in the models template folder. CASE INSENSITIVE Below are examples of model types 'hrrr' HRRR contiguous United States model 'hrrrak' HRRR Alaska model (alias 'alaska') 'rap' RAP model 'gfs' Global Forecast System (atmosphere) 'gfs_wave' Global Forecast System (wave) 'rrfs' Rapid Refresh Forecast System prototype for more information see https://herbie.readthedocs.io/en/latest/user_guide/model_info.html

- **product**: Output variable product file type (sfc (surface fields), prs (pressure fields), nat (native fields), subh (subhourly fields)). Not specifying this will use the first product in model template file.

- **frequency**: frequency in hours of the forecast

- **forecast_lead_times**: Forecast lead time in the format [start_time, end_time, increment] (in hours). This parameter can be None, and in this case see parameter 'forecast_start_date' for more details. You cannot specify 'forecast_lead_times' and 'forecast_start_date' at the same time.

- **forecast_start_date**: latest datetime (in the format "%Y-%m-%d %H:%M") for which analysis (zero lead time) are retrieved. After this datetime, forecasts with progressively increasing lead times are retrieved. If this parameter is set to None and 'forecast_lead_times' is also set to None, then the workflow returns analysis (zero lead time) up to the latest analysis available, and from that point it returns forecasts with progressively increasing lead times.

- **search_text**: It's a regular expression used to search on GRIB2 Index files and allow you to download just the layer of the file required instead of complete file. For more information on search_text refer to below url. https://blaylockbk.github.io/Herbie/_build/html/user_guide/searchString.html

## Tasks

- **list_herbie**: Lists herbie products.

- **download_herbie**: Download herbie grib files.

## Workflow Yaml

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
  forecast_lead_times: null
  forecast_start_date: null
  search_text: :TMP:2 m
tasks:
  list_herbie:
    op: list_herbie
    parameters:
      model: '@from(model)'
      product: '@from(product)'
      frequency: '@from(frequency)'
      forecast_lead_times: '@from(forecast_lead_times)'
      forecast_start_date: '@from(forecast_start_date)'
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
      (in hours). This parameter can be None, and in this case see parameter 'forecast_start_date'
      for more details. You cannot specify 'forecast_lead_times' and 'forecast_start_date'
      at the same time.
    forecast_start_date: latest datetime (in the format "%Y-%m-%d %H:%M") for which
      analysis (zero lead time) are retrieved. After this datetime, forecasts with
      progressively increasing lead times are retrieved. If this parameter is set
      to None and 'forecast_lead_times' is also set to None, then the workflow returns
      analysis (zero lead time) up to the latest analysis available, and from that
      point it returns forecasts with progressively increasing lead times.
    search_text: It's a regular expression used to search on GRIB2 Index files and
      allow you to download just the layer of the file required instead of complete
      file. For more information on search_text refer to below url. https://blaylockbk.github.io/Herbie/_build/html/user_guide/searchString.html


```