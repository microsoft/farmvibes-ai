# data_ingestion/airbus/airbus_download

Downloads available AirBus imagery for the input geometry and time range. The workflow will check available imagery, using the AirBus API, that contains the input geometry and inside the input time range. Matching images will be purchased (if they are not already in the user's library) and downloaded. This workflow requires an AirBus API key.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- airbus_products --> tsk2{{download}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    tsk2{{download}} -- downloaded_products --> out1>raster]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **raster**: AirBus raster.

## Parameters

- **api_key**: AirBus API key. Required to run the workflow.

## Tasks

- **list**: Lists available AirBus products for the input geometry and time range.

- **download**: Downloads the AirBus imagery from the listed product.

## Workflow Yaml

```yaml

name: airbus_download
sources:
  user_input:
  - list.input_item
sinks:
  raster: download.downloaded_products
parameters:
  api_key: null
tasks:
  list:
    op: list_airbus_products
    parameters:
      api_key: '@from(api_key)'
  download:
    op: download_airbus
    parameters:
      api_key: '@from(api_key)'
edges:
- origin: list.airbus_products
  destination:
  - download.airbus_products
description:
  short_description: Downloads available AirBus imagery for the input geometry and
    time range.
  long_description: The workflow will check available imagery, using the AirBus API,
    that contains the input geometry and inside the input time range. Matching images
    will be purchased (if they are not already in the user's library) and downloaded.
    This workflow requires an AirBus API key.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: AirBus raster.
  parameters:
    api_key: AirBus API key. Required to run the workflow.


```