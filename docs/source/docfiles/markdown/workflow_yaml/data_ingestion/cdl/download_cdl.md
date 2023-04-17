# data_ingestion/cdl/download_cdl

```yaml

name: download_cdl
sources:
  user_input:
  - list_cdl.input_item
sinks:
  raster: download_cdl.cdl_raster
tasks:
  list_cdl:
    op: list_cdl_products
  download_cdl:
    op: download_cdl
    op_dir: download_cdl_data
edges:
- origin: list_cdl.cdl_products
  destination:
  - download_cdl.input_product
description:
  short_description: Downloads crop classes maps in the continental USA for the input
    time range.
  long_description: The workflow will download crop-specific land cover maps from
    the USDA Cropland Data Layer, available for the continental United States. The
    input geometry must intersect with the coverage area.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: CDL land cover raster.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list_cdl}}
    tsk2{{download_cdl}}
    tsk1{{list_cdl}} -- cdl_products/input_product --> tsk2{{download_cdl}}
    inp1>user_input] -- input_item --> tsk1{{list_cdl}}
    tsk2{{download_cdl}} -- cdl_raster --> out1>raster]
```