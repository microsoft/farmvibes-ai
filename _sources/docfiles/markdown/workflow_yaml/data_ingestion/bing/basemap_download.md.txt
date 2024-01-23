# data_ingestion/bing/basemap_download

```yaml

name: basemap_download
sources:
  input_geometry:
  - list.user_input
sinks:
  basemaps: download.basemap
parameters:
  api_key: null
  zoom_level: null
tasks:
  list:
    op: list_bing_maps
    parameters:
      api_key: '@from(api_key)'
      zoom_level: '@from(zoom_level)'
  download:
    op: download_bing_basemap
    parameters:
      api_key: '@from(api_key)'
edges:
- origin: list.products
  destination:
  - download.input_product
description:
  short_description: Downloads Bing Maps basemaps.
  long_description: The workflow will list all tiles intersecting with the input geometry
    for a given zoom level and download a basemap for each of them using Bing Maps
    API. The basemap tiles will be returned as individual rasters.
  sources:
    input_geometry: Geometry of interest for which to download the basemap tiles.
  sinks:
    basemaps: Downloaded basemaps.


```

```{mermaid}
    graph TD
    inp1>input_geometry]
    out1>basemaps]
    tsk1{{list}}
    tsk2{{download}}
    tsk1{{list}} -- products/input_product --> tsk2{{download}}
    inp1>input_geometry] -- user_input --> tsk1{{list}}
    tsk2{{download}} -- basemap --> out1>basemaps]
```