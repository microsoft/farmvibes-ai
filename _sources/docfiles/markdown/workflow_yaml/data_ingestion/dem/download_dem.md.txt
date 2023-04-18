# data_ingestion/dem/download_dem

```yaml

name: download_dem
sources:
  user_input:
  - list.input_items
sinks:
  raster: download.downloaded_product
parameters:
  pc_key: null
tasks:
  list:
    op: list_dem_products
  download:
    op: download_dem
    parameters:
      api_key: '@from(pc_key)'
edges:
- origin: list.dem_products
  destination:
  - download.input_product
description:
  short_description: Downloads digital elevation map tiles that intersect with the
    input geometry and time range.
  long_description: The workflow will download digital elevation maps from the USGS
    3DEP datasets (available for the United States) through the Planetary Computer.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: DEM raster.
  parameters:
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- dem_products/input_product --> tsk2{{download}}
    inp1>user_input] -- input_items --> tsk1{{list}}
    tsk2{{download}} -- downloaded_product --> out1>raster]
```