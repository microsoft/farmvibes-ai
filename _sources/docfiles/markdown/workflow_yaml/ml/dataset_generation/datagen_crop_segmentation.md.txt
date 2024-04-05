# ml/dataset_generation/datagen_crop_segmentation

Generates a dataset for crop segmentation, based on NDVI raster and Crop Data Layer (CDL) maps. The workflow generates SpaceEye cloud-free data for the input region and time range and computes NDVI over those. It also downloads CDL maps for the years comprised in the time range.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>ndvi]
    out2>cdl]
    tsk1{{spaceeye}}
    tsk2{{ndvi}}
    tsk3{{cdl}}
    tsk1{{spaceeye}} -- raster --> tsk2{{ndvi}}
    inp1>user_input] -- user_input --> tsk1{{spaceeye}}
    inp1>user_input] -- user_input --> tsk3{{cdl}}
    tsk2{{ndvi}} -- index_raster --> out1>ndvi]
    tsk3{{cdl}} -- raster --> out2>cdl]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **ndvi**: NDVI rasters.

- **cdl**: CDL map for the years comprised in the input time range.

## Parameters

- **pc_key**: Optional Planetary Computer API key.

## Tasks

- **spaceeye**: Runs the SpaceEye cloud removal pipeline using an interpolation-based algorithm, yielding daily cloud-free images for the input geometry and time range.

- **ndvi**: Computes an index from the bands of an input raster.

- **cdl**: Downloads crop classes maps in the continental USA for the input time range.

## Workflow Yaml

```yaml

name: datagen_crop_segmentation
sources:
  user_input:
  - spaceeye.user_input
  - cdl.user_input
sinks:
  ndvi: ndvi.index_raster
  cdl: cdl.raster
parameters:
  pc_key: null
tasks:
  spaceeye:
    workflow: data_ingestion/spaceeye/spaceeye_interpolation
    parameters:
      pc_key: '@from(pc_key)'
  ndvi:
    workflow: data_processing/index/index
    parameters:
      index: ndvi
  cdl:
    workflow: data_ingestion/cdl/download_cdl
edges:
- origin: spaceeye.raster
  destination:
  - ndvi.raster
description:
  short_description: Generates a dataset for crop segmentation, based on NDVI raster
    and Crop Data Layer (CDL) maps.
  long_description: The workflow generates SpaceEye cloud-free data for the input
    region and time range and computes NDVI over those. It also downloads CDL maps
    for the years comprised in the time range.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    ndvi: NDVI rasters.
    cdl: CDL map for the years comprised in the input time range.
  parameters:
    pc_key: Optional Planetary Computer API key.


```