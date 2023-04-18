# data_ingestion/naip/download_naip

```yaml

name: download_naip
sources:
  user_input:
  - list.input_item
sinks:
  raster: download.downloaded_product
parameters:
  pc_key: null
tasks:
  list:
    op: list_naip_products
  download:
    op: download_naip
    parameters:
      api_key: '@from(pc_key)'
edges:
- origin: list.naip_products
  destination:
  - download.input_product
description:
  short_description: Downloads NAIP tiles that intersect with the input geometry and
    time range.
  long_description: null
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: NAIP tiles.
  parameters:
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- naip_products/input_product --> tsk2{{download}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    tsk2{{download}} -- downloaded_product --> out1>raster]
```