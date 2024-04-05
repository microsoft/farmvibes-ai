# data_ingestion/modis/download_modis_vegetation_index

Downloads MODIS 16-day vegetation index products that intersect with the input geometry and time range. The workflow will download products at the chosen index and resolution. The products are available at a 16-day interval and pixel values are selected based on low clouds, low view angle, and highest index value. Vegetation index values range from (-2000 to 10000). For more information, see https://planetarycomputer.microsoft.com/dataset/modis-13Q1-061 and https://lpdaac.usgs.gov/products/mod13a1v061/ .

```{mermaid}
    graph TD
    inp1>user_input]
    out1>index]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- modis_products/product --> tsk2{{download}}
    inp1>user_input] -- input_data --> tsk1{{list}}
    tsk2{{download}} -- index --> out1>index]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **index**: Products containing the chosen index at the chosen resolution.

## Parameters

- **index**: Vegetation index that should be downloaded. Either 'evi' or 'ndvi'.

- **pc_key**: Optional Planetary Computer API key.

- **resolution_m**: Product resolution, in meters. Either 250 or 500.

## Tasks

- **list**: Lists MODIS vegetation products for input geometry, time range and resolution.

- **download**: Downloads selected index raster from Modis product.

## Workflow Yaml

```yaml

name: download_modis_vegetation_index
sources:
  user_input:
  - list.input_data
sinks:
  index: download.index
parameters:
  index: null
  pc_key: null
  resolution_m: null
tasks:
  list:
    op: list_modis_vegetation
    parameters:
      resolution: '@from(resolution_m)'
  download:
    op: download_modis_vegetation
    parameters:
      pc_key: '@from(pc_key)'
      index: '@from(index)'
edges:
- origin: list.modis_products
  destination:
  - download.product
description:
  short_description: Downloads MODIS 16-day vegetation index products that intersect
    with the input geometry and time range.
  long_description: The workflow will download products at the chosen index and resolution.
    The products are available at a 16-day interval and pixel values are selected
    based on low clouds, low view angle, and highest index value. Vegetation index
    values range from (-2000 to 10000). For more information, see https://planetarycomputer.microsoft.com/dataset/modis-13Q1-061
    and https://lpdaac.usgs.gov/products/mod13a1v061/ .
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    index: Products containing the chosen index at the chosen resolution.
  parameters:
    index: Vegetation index that should be downloaded. Either 'evi' or 'ndvi'.
    pc_key: Optional Planetary Computer API key.
    resolution_m: Product resolution, in meters. Either 250 or 500.


```