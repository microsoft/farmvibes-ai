# data_ingestion/glad/glad_forest_extent_download_merge

Downloads the tiles from Global Land Analysis (GLAD) forest data that intersect with the user input geometry and time range, and merges them into a single raster. The workflow lists the GLAD forest products that intersect with the input geometry and time range, and downloads the filtered products. The downloaded products are merged into a single raster and classified. The result tiles have pixel values categorized into two classes - 0 (non-forest) and 1 (forest). This workflow uses the same forest definition as the Food and Agriculture Organization of the United Nations (FAO).

```{mermaid}
    graph TD
    inp1>input_item]
    out1>merged_product]
    out2>categorical_raster]
    tsk1{{glad_forest_extent_download}}
    tsk2{{group_rasters_by_time}}
    tsk3{{merge}}
    tsk1{{glad_forest_extent_download}} -- downloaded_product/rasters --> tsk2{{group_rasters_by_time}}
    tsk2{{group_rasters_by_time}} -- raster_groups/raster_sequence --> tsk3{{merge}}
    inp1>input_item] -- input_item --> tsk1{{glad_forest_extent_download}}
    tsk3{{merge}} -- raster --> out1>merged_product]
    tsk1{{glad_forest_extent_download}} -- downloaded_product --> out2>categorical_raster]
```

## Sources

- **input_item**: Geometry of interest for which to download the GLAD forest extent data.

## Sinks

- **merged_product**: Merged GLAD forest extent product to geometry of interest.

- **categorical_raster**: Raster with the GLAD forest extent data.

## Tasks

- **glad_forest_extent_download**: Downloads Global Land Analysis (GLAD) forest extent data.

- **group_rasters_by_time**: This op groups rasters in time according to 'criterion'.

- **merge**: Merges rasters in a sequence to a single raster.

## Workflow Yaml

```yaml

name: glad_forest_extent_download_merge
sources:
  input_item:
  - glad_forest_extent_download.input_item
parameters: null
sinks:
  merged_product: merge.raster
  categorical_raster: glad_forest_extent_download.downloaded_product
tasks:
  glad_forest_extent_download:
    workflow: data_ingestion/glad/glad_forest_extent_download
  group_rasters_by_time:
    op: group_rasters_by_time
    parameters:
      criterion: year
  merge:
    op: merge_rasters
edges:
- origin: glad_forest_extent_download.downloaded_product
  destination:
  - group_rasters_by_time.rasters
- origin: group_rasters_by_time.raster_groups
  destination:
  - merge.raster_sequence
description:
  short_description: Downloads the tiles from Global Land Analysis (GLAD) forest data
    that intersect with the user input geometry and time range, and merges them into
    a single raster.
  long_description: The workflow lists the GLAD forest products that intersect with
    the input geometry and time range, and downloads the filtered products. The downloaded
    products are merged into a single raster and classified. The result tiles have
    pixel values categorized into two classes - 0 (non-forest) and 1 (forest). This
    workflow uses the same forest definition as the Food and Agriculture Organization
    of the United Nations (FAO).
  sources:
    input_item: Geometry of interest for which to download the GLAD forest extent
      data.
  sinks:
    merged_product: Merged GLAD forest extent product to geometry of interest.
    categorical_raster: Raster with the GLAD forest extent data.


```