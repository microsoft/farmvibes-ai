# data_ingestion/weather/get_ambient_weather

```yaml

name: get_ambient_weather
sources:
  user_input:
  - get_weather.user_input
sinks:
  weather: get_weather.weather
parameters:
  api_key: null
  app_key: null
  limit: -1
  feed_interval: null
tasks:
  get_weather:
    op: download_ambient_weather
    op_dir: download_ambient_weather
    parameters:
      api_key: '@from(api_key)'
      app_key: '@from(app_key)'
      limit: '@from(limit)'
      feed_interval: '@from(feed_interval)'
edges: null
description:
  short_description: Downloads weather data from an Ambient Weather station.
  long_description: The workflow connects to the Ambient Weather REST API and requests
    data for the input time range. The input geometry will be used to find a device
    inside the region. If not devices are found in the geometry, the workflow will
    fail. Connection to the API requires an API key and an App key.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    weather: Weather data from the station.
  parameters:
    api_key: Ambient Weather API key.
    app_key: Ambient Weather App key.
    limit: Maximum number of data points. If -1, do not limit.
    feed_interval: Interval between samples. Defined by the weather station.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>weather]
    tsk1{{get_weather}}
    inp1>user_input] -- user_input --> tsk1{{get_weather}}
    tsk1{{get_weather}} -- weather --> out1>weather]
```