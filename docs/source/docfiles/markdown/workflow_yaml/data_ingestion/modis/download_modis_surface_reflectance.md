# data_ingestion/modis/download_modis_surface_reflectance

```yaml

name: download_modis_surface_reflectance
sources:
  user_input:
  - list.input_data
sinks:
  raster: download.raster
parameters:
  pc_key: null
  resolution_m: null
tasks:
  list:
    op: list_modis_sr
    parameters:
      resolution: '@from(resolution_m)'
  download:
    op: download_modis_sr
    parameters:
      pc_key: '@from(pc_key)'
edges:
- origin: list.modis_products
  destination:
  - download.product
description:
  short_description: Downloads MODIS 8-day surface reflectance rasters that intersect
    with the input geometry and time range.
  long_description: The workflow will download MODIS raster images either at 250m
    or 500m resolution. The products are available at a 8-day interval and pixel values
    are selected based on low clouds, low view angle, and highest index value. Notice
    that only bands 1, 2 and quality control are available on 250m. For more information,
    see https://planetarycomputer.microsoft.com/dataset/modis-09Q1-061 https://planetarycomputer.microsoft.com/dataset/modis-09A1-061
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: Products containing MODIS reflectance bands and data.
  parameters:
    pc_key: Optional Planetary Computer API key.
    resolution_m: Product resolution, in meters. Either 250 or 500.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- modis_products/product --> tsk2{{download}}
    inp1>user_input] -- input_data --> tsk1{{list}}
    tsk2{{download}} -- raster --> out1>raster]
```