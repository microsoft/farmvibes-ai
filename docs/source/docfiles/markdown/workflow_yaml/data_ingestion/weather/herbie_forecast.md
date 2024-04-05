# data_ingestion/weather/herbie_forecast

Downloads forecast observations for provided location & time range using herbie python package. Herbie is a python package that downloads recent and archived numerical weather prediction (NWP) model outputs from different cloud archive sources. Its most popular capability is to download HRRR model data. NWP data in GRIB2 format can be read with xarray+cfgrib. Model data Herbie can retrieve includes the High Resolution Rapid Refresh (HRRR), Rapid Refresh (RAP), Global Forecast System (GFS), National Blend of Models (NBM), Rapid Refresh Forecast System - Prototype (RRFS), and ECMWF open data forecast products (ECMWF).

```{mermaid}
    graph TD
    inp1>user_input]
    out1>weather_forecast]
    out2>forecast_range]
    tsk1{{forecast_range}}
    tsk2{{forecast_download}}
    tsk1{{forecast_range}} -- download_period/user_input --> tsk2{{forecast_download}}
    inp1>user_input] -- user_input --> tsk1{{forecast_range}}
    tsk2{{forecast_download}} -- weather_forecast --> out1>weather_forecast]
    tsk1{{forecast_range}} -- download_period --> out2>forecast_range]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **weather_forecast**: Downloaded Forecast observations, cleaned, interpolated and mapped to each hour.

- **forecast_range**: Time range of forecast observations.

## Parameters

- **forecast_lead_times**: Help to define forecast lead time in hours. Accept the input in range format. Example - (1, 25, 1) For more information refer below url. https://blaylockbk.github.io/Herbie/_build/html/reference_guide/_autosummary/herbie.archive.Herbie.html

- **search_text**: It's a regular expression used to search on GRIB2 Index files and allow you to download just the layer of the file required instead of complete file. For more information on search_text refer to below url. https://blaylockbk.github.io/Herbie/_build/html/user_guide/searchString.html

- **weather_type**: It's a user preferred text to represent weather parameter type (temperature, humidity, wind_speed etc). This is used as column name for the output returned by operator.

- **model**: Model name as defined in the models template folder. CASE INSENSITIVE Below are examples of model types 'hrrr' HRRR contiguous United States model 'hrrrak' HRRR Alaska model (alias 'alaska') 'rap' RAP model 'gfs' Global Forecast System (atmosphere) 'gfs_wave' Global Forecast System (wave) 'rrfs' Rapid Refresh Forecast System prototype

- **overwrite**: If true, look for GRIB2 file even if local copy exists. If false, use the local copy

- **product**: Output variable product file type (sfc (surface fields), prs (pressure fields), nat (native fields), subh (subhourly fields)). Not specifying this will use the first product in model template file.

## Tasks

- **forecast_range**: Splits input time range according to frequency and number of hours in lead time.

- **forecast_download**: Downloads forecast observations with Herbie.

## Workflow Yaml

```yaml

name: forecast_weather
sources:
  user_input:
  - forecast_range.user_input
sinks:
  weather_forecast: forecast_download.weather_forecast
  forecast_range: forecast_range.download_period
parameters:
  forecast_lead_times: null
  search_text: null
  weather_type: null
  model: null
  overwrite: null
  product: null
tasks:
  forecast_range:
    op: forecast_range_split
    op_dir: download_herbie
    parameters:
      forecast_lead_times: '@from(forecast_lead_times)'
      weather_type: '@from(weather_type)'
  forecast_download:
    op: forecast_weather
    op_dir: download_herbie
    parameters:
      model: '@from(model)'
      overwrite: '@from(overwrite)'
      product: '@from(product)'
      forecast_lead_times: '@from(forecast_lead_times)'
      search_text: '@from(search_text)'
      weather_type: '@from(weather_type)'
edges:
- origin: forecast_range.download_period
  destination:
  - forecast_download.user_input
description:
  short_description: Downloads forecast observations for provided location & time
    range using herbie python package.
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
    weather_forecast: Downloaded Forecast observations, cleaned, interpolated and
      mapped to each hour.
    forecast_range: Time range of forecast observations.
  parameters:
    model: Model name as defined in the models template folder. CASE INSENSITIVE Below
      are examples of model types 'hrrr' HRRR contiguous United States model 'hrrrak'
      HRRR Alaska model (alias 'alaska') 'rap' RAP model 'gfs' Global Forecast System
      (atmosphere) 'gfs_wave' Global Forecast System (wave) 'rrfs' Rapid Refresh Forecast
      System prototype
    overwrite: If true, look for GRIB2 file even if local copy exists. If false, use
      the local copy
    product: Output variable product file type (sfc (surface fields), prs (pressure
      fields), nat (native fields), subh (subhourly fields)). Not specifying this
      will use the first product in model template file.
    forecast_lead_times: Help to define forecast lead time in hours. Accept the input
      in range format. Example - (1, 25, 1) For more information refer below url.
      https://blaylockbk.github.io/Herbie/_build/html/reference_guide/_autosummary/herbie.archive.Herbie.html
    search_text: It's a regular expression used to search on GRIB2 Index files and
      allow you to download just the layer of the file required instead of complete
      file. For more information on search_text refer to below url. https://blaylockbk.github.io/Herbie/_build/html/user_guide/searchString.html
    weather_type: It's a user preferred text to represent weather parameter type (temperature,
      humidity, wind_speed etc). This is used as column name for the output returned
      by operator.


```