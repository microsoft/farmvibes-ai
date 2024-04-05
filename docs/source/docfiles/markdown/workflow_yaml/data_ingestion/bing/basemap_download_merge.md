# data_ingestion/bing/basemap_download_merge

Downloads Bing Maps basemap tiles and merges them into a single raster. The workflow will list all tiles intersecting with the input geometry for a given zoom level, and download a basemap for each of them using Bing Maps API. The basemaps will be merged into a single raster with the union of the geometries of all tiles.

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

## Sources

- **input_geometry**: Geometry of interest for which to download the basemap tiles.

## Sinks

- **merged_basemap**: Merged basemap raster.

## Parameters

- **api_key**: Required BingMaps API key.

- **zoom_level**: Zoom level of interest, ranging from 0 to 20. For instance, a zoom level of 1 corresponds to a resolution of 78271.52 m/pixel, a zoom level of 10 corresponds to 152.9 m/pixel, and a zoom level of 19 corresponds to 0.3 m/pixel. For more information on zoom levels and their corresponding scale and resolution, please refer to the BingMaps API documentation at https://learn.microsoft.com/en-us/bingmaps/articles/understanding-scale-and-resolution

- **merge_resolution**: Determines how the resolution of the output raster is defined. One of 'equal' (breaks if the resolution of the sequence rasters are not the same), 'lowest' (uses the lowest resolution among rasters), 'highest' (uses the highest resolution among rasters), or 'average' (averages the resolution of all rasters in the sequence).

## Tasks

- **basemap_download**: Downloads Bing Maps basemaps.

- **to_sequence**: Combines a list of Rasters into a RasterSequence.

- **merge**: Merges rasters in a sequence to a single raster.

## Workflow Yaml

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