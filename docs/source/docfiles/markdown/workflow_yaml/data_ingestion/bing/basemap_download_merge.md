# data_ingestion/bing/basemap_download_merge

```yaml

name: basemap_download_merge
sources:
  input_geometry:
  - basemap_download.input_geometry
sinks:
  merged_basemap: merge.raster
parameters:
  api_key: null
  zoom_level: null
  merge_resolution: highest
tasks:
  basemap_download:
    workflow: data_ingestion/bing/basemap_download
    parameters:
      api_key: '@from(api_key)'
      zoom_level: '@from(zoom_level)'
  to_sequence:
    op: list_to_sequence
  merge:
    op: merge_rasters
    parameters:
      resolution: '@from(merge_resolution)'
edges:
- origin: basemap_download.basemaps
  destination:
  - to_sequence.list_rasters
- origin: to_sequence.rasters_seq
  destination:
  - merge.raster_sequence
description:
  short_description: Downloads Bing Maps basemap tiles and merges them into a single
    raster.
  long_description: The workflow will list all tiles intersecting with the input geometry
    for a given zoom level, and download a basemap for each of them using Bing Maps
    API. The basemaps will be merged into a single raster with the union of the geometries
    of all tiles.
  sources:
    input_geometry: Geometry of interest for which to download the basemap tiles.
  sinks:
    merged_basemap: Merged basemap raster.


```

```{mermaid}
    graph TD
    inp1>input_geometry]
    out1>merged_basemap]
    tsk1{{basemap_download}}
    tsk2{{to_sequence}}
    tsk3{{merge}}
    tsk1{{basemap_download}} -- basemaps/list_rasters --> tsk2{{to_sequence}}
    tsk2{{to_sequence}} -- rasters_seq/raster_sequence --> tsk3{{merge}}
    inp1>input_geometry] -- input_geometry --> tsk1{{basemap_download}}
    tsk3{{merge}} -- raster --> out1>merged_basemap]
```