# data_ingestion/weather/download_chirps

```yaml

name: chirps
sources:
  user_input:
  - list_chirps.input_item
sinks:
  product: download_chirps.downloaded_product
parameters:
  freq: daily
  res: p05
tasks:
  list_chirps:
    op: list_chirps
    parameters:
      freq: '@from(freq)'
      res: '@from(res)'
  download_chirps:
    op: download_chirps
edges:
- origin: list_chirps.chirps_products
  destination:
  - download_chirps.chirps_product
description:
  short_description: Downloads accumulated precipitation data from the CHIRPS dataset.
  long_description: null
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    product: TIFF file containing accumulated precipitation.
  parameters:
    freq: daily or monthly frequencies
    res: p05 for 0.05 degree resolution or p25 for 0.25 degree resolution, p25 is
      only available daily


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>product]
    tsk1{{list_chirps}}
    tsk2{{download_chirps}}
    tsk1{{list_chirps}} -- chirps_products/chirps_product --> tsk2{{download_chirps}}
    inp1>user_input] -- input_item --> tsk1{{list_chirps}}
    tsk2{{download_chirps}} -- downloaded_product --> out1>product]
```