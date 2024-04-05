# data_ingestion/osm_road_geometries

Downloads road geometry for input region from Open Street Maps. The workflow downloads information from Open Street Maps for the target region and generates geometries for roads that intercept the input region bounding box.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>roads]
    tsk1{{download}}
    inp1>user_input] -- input_region --> tsk1{{download}}
    tsk1{{download}} -- roads --> out1>roads]
```

## Sources

- **user_input**: List of external references.

## Sinks

- **roads**: Geometry collection with road geometries that intercept the input region bounding box.

## Parameters

- **network_type**: Type of roads that will be selected. One of:
  - 'drive_service': get drivable streets, including service roads.
  - 'walk': get all streets and paths that pedestrians can use (this network type ignores
    one-way directionality).
  - 'bike': get all streets and paths that cyclists can use.
  - 'all': download all non-private OSM streets and paths (this is the default network type
    unless you specify a different one).
  - 'all_private': download all OSM streets and paths, including private-access ones.
  - 'drive': get drivable public streets (but not service roads).
For more information see https://osmnx.readthedocs.io/en/stable/index.html.

- **buffer_size**: Size of buffer, in meters, to search for nodes in OSM.

## Tasks

- **download**: Downloads road geometry for input region from Open Street Maps.

## Workflow Yaml

```yaml

name: osm_road_geometries
sources:
  user_input:
  - download.input_region
sinks:
  roads: download.roads
parameters:
  network_type: null
  buffer_size: null
tasks:
  download:
    op: download_road_geometries
    parameters:
      network_type: '@from(network_type)'
      buffer_size: '@from(buffer_size)'
description:
  short_description: Downloads road geometry for input region from Open Street Maps.
  long_description: The workflow downloads information from Open Street Maps for the
    target region and generates geometries for roads that intercept the input region
    bounding box.
  sources:
    user_input: List of external references.
  sinks:
    roads: Geometry collection with road geometries that intercept the input region
      bounding box.
  parameters:
    network_type: "Type of roads that will be selected. One of:\n  - 'drive_service':\
      \ get drivable streets, including service roads.\n  - 'walk': get all streets\
      \ and paths that pedestrians can use (this network type ignores\n    one-way\
      \ directionality).\n  - 'bike': get all streets and paths that cyclists can\
      \ use.\n  - 'all': download all non-private OSM streets and paths (this is the\
      \ default network type\n    unless you specify a different one).\n  - 'all_private':\
      \ download all OSM streets and paths, including private-access ones.\n  - 'drive':\
      \ get drivable public streets (but not service roads).\nFor more information\
      \ see https://osmnx.readthedocs.io/en/stable/index.html."
    buffer_size: Size of buffer, in meters, to search for nodes in OSM.


```