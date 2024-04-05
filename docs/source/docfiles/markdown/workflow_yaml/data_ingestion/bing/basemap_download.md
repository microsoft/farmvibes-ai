# data_ingestion/bing/basemap_download

Downloads Bing Maps basemaps. The workflow will list all tiles intersecting with the input geometry for a given zoom level and download a basemap for each of them using Bing Maps API. The basemap tiles will be returned as individual rasters.

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

## Sources

- **input_geometry**: Geometry of interest for which to download the basemap tiles.

## Sinks

- **basemaps**: Downloaded basemaps.

## Parameters

- **api_key**: Required BingMaps API key.

- **zoom_level**: Zoom level of interest, ranging from 0 to 20. For instance, a zoom level of 1 corresponds to a resolution of 78271.52 m/pixel, a zoom level of 10 corresponds to 152.9 m/pixel, and a zoom level of 19 corresponds to 0.3 m/pixel. For more information on zoom levels and their corresponding scale and resolution, please refer to the BingMaps API documentation at https://learn.microsoft.com/en-us/bingmaps/articles/understanding-scale-and-resolution

## Tasks

- **list**: Lists BingMaps basemap tile products intersecting the input geometry for a given `zoom_level`.

- **download**: Downloads a basemap tile represented by a BingMapsProduct using BingMapsAPI.

## Workflow Yaml

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