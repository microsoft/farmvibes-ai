# data_ingestion/landsat/preprocess_landsat

```yaml

name: preprocess_landsat
sources:
  user_input:
  - list.input_item
sinks:
  raster: stack.landsat_raster
parameters:
  pc_key: null
  qa_mask_value: 64
tasks:
  list:
    op: list_landsat_products_pc
  download:
    op: download_landsat_from_pc
    parameters:
      api_key: '@from(pc_key)'
  stack:
    op: stack_landsat
    parameters:
      qa_mask_value: '@from(qa_mask_value)'
edges:
- origin: list.landsat_products
  destination:
  - download.landsat_product
- origin: download.downloaded_product
  destination:
  - stack.landsat_product
description:
  short_description: Downloads and preprocesses LANDSAT tiles that intersect with
    the input geometry and time range.
  long_description: The workflow will download the tile bands from the Planetary Computer
    and stack them into a single raster at 30m resolution.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: LANDSAT rasters at 30m resolution.
  parameters:
    pc_key: Optional Planetary Computer API key.
    qa_mask_value: Bitmap for which pixel to be included. See documentation for each
      bit in https://www.usgs.gov/media/images/landsat-collection-2-pixel-quality-assessment-bit-index
      For example, the default value 64 (i.e. 1<<6 ) corresponds to "Clear" pixels


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk3{{stack}}
    tsk1{{list}} -- landsat_products/landsat_product --> tsk2{{download}}
    tsk2{{download}} -- downloaded_product/landsat_product --> tsk3{{stack}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    tsk3{{stack}} -- landsat_raster --> out1>raster]
```