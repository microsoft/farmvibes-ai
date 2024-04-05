# data_ingestion/weather/get_forecast

Downloads weather forecast data from NOAA Global Forecast System (GFS) for the input time range. The workflow downloads global forecast data from the Planetary Computer with 13km resolution between grid points. The workflow requires a SAS token to access the blob storage, which can be found at https://planetarycomputer.microsoft.com/dataset/storage/noaa-gfs.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>forecast]
    tsk1{{preprocessing}}
    tsk2{{gfs_download}}
    tsk3{{read_forecast}}
    tsk1{{preprocessing}} -- time --> tsk2{{gfs_download}}
    tsk1{{preprocessing}} -- location --> tsk3{{read_forecast}}
    tsk2{{gfs_download}} -- global_forecast --> tsk3{{read_forecast}}
    inp1>user_input] -- user_input --> tsk1{{preprocessing}}
    tsk3{{read_forecast}} -- local_forecast --> out1>forecast]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **forecast**: Weather forecast data.

## Parameters

- **noaa_gfs_token**: SAS token to access blob storage.

## Tasks

- **preprocessing**: Gets the most relevant model date and forecast hour of product for the given input day, time and location.

- **gfs_download**: Downloads the global forecast for the given input time.

- **read_forecast**: Extracts the local data from a global forecast.

## Workflow Yaml

```yaml

name: get_forecast
sources:
  user_input:
  - preprocessing.user_input
sinks:
  forecast: read_forecast.local_forecast
parameters:
  noaa_gfs_token: null
tasks:
  preprocessing:
    op: gfs_preprocess
    op_dir: gfs_preprocess
    parameters:
      sas_token: '@from(noaa_gfs_token)'
  gfs_download:
    op: gfs_download
    op_dir: gfs_download
    parameters:
      sas_token: '@from(noaa_gfs_token)'
  read_forecast:
    op: read_grib_forecast
    op_dir: read_grib_forecast
edges:
- origin: preprocessing.time
  destination:
  - gfs_download.time
- origin: preprocessing.location
  destination:
  - read_forecast.location
- origin: gfs_download.global_forecast
  destination:
  - read_forecast.global_forecast
description:
  short_description: Downloads weather forecast data from NOAA Global Forecast System
    (GFS) for the input time range.
  long_description: The workflow downloads global forecast data from the Planetary
    Computer with 13km resolution between grid points. The workflow requires a SAS
    token to access the blob storage, which can be found at https://planetarycomputer.microsoft.com/dataset/storage/noaa-gfs.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    forecast: Weather forecast data.
  parameters:
    noaa_gfs_token: SAS token to access blob storage.


```