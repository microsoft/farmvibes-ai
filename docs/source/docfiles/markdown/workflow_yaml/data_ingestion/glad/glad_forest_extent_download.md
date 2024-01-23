# data_ingestion/glad/glad_forest_extent_download

```yaml

name: glad_forest_extent_download
sources:
  input_item:
  - list.input_item
sinks:
  downloaded_product: download.downloaded_product
parameters: null
tasks:
  list:
    op: list_glad_products
  download:
    op: download_glad
    op_dir: download_glad_data
edges:
- origin: list.glad_products
  destination:
  - download.glad_product
description:
  short_description: Downloads Global Land Analysis (GLAD) forest extent data.
  long_description: The workflow will list all GLAD forest extent products that intersect
    with the input geometry and download the data for each of them. The data will
    be returned as rasters.
  sources:
    input_item: Geometry of interest for which to download the GLAD forest extent
      data.
  sinks:
    downloaded_product: Downloaded GLAD forest extent product.


```

```{mermaid}
    graph TD
    inp1>input_item]
    out1>downloaded_product]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- glad_products/glad_product --> tsk2{{download}}
    inp1>input_item] -- input_item --> tsk1{{list}}
    tsk2{{download}} -- downloaded_product --> out1>downloaded_product]
```