# data_ingestion/alos/alos_forest_extent_download

```yaml

name: alos_forest_extent_download
sources:
  user_input:
  - list.input_data
sinks:
  downloaded_product: download.raster
parameters:
  pc_key: null
tasks:
  list:
    op: list_alos_products
  download:
    op: download_alos
    parameters:
      pc_key: '@from(pc_key)'
edges:
- origin: list.alos_products
  destination:
  - download.product
description:
  short_description: Downloads Advanced Land Observing Satellite (ALOS) forest/non-forest
    classification map.
  long_description: The workflow lists all ALOS forest/non-forest classification products
    that intersect with the input geometry and time range (available range 2015-2020),
    then downloads the data for each of them. The data will be returned in the form
    of rasters.
  sources:
    user_input: Geometry of interest for which to download the ALOS forest/non-forest
      classification map.
  sinks:
    downloaded_product: Downloaded ALOS forest/non-forest classification map.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>downloaded_product]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- alos_products/product --> tsk2{{download}}
    inp1>user_input] -- input_data --> tsk1{{list}}
    tsk2{{download}} -- raster --> out1>downloaded_product]
```