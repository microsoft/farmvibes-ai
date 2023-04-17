# data_ingestion/sentinel2/preprocess_s2

```yaml

name: preprocess_s2
sources:
  user_input:
  - list.input_item
  - filter.bounds_items
sinks:
  raster: merge.output_raster
  mask: merge.output_mask
parameters:
  min_tile_cover: null
  max_tiles_per_time: null
  pc_key: null
  dl_timeout: null
tasks:
  list:
    op: list_sentinel2_products_pc
    op_dir: list_sentinel2_products
  filter:
    op: select_necessary_coverage_items
    parameters:
      min_cover: '@from(min_tile_cover)'
      max_items: '@from(max_tiles_per_time)'
  download:
    op: download_stack_sentinel2
    parameters:
      api_key: '@from(pc_key)'
      timeout_s: '@from(dl_timeout)'
  group:
    op: group_sentinel2_orbits
  merge:
    op: merge_sentinel2_orbits
edges:
- origin: list.sentinel_products
  destination:
  - filter.items
- origin: filter.filtered_items
  destination:
  - download.sentinel_product
- origin: download.raster
  destination:
  - group.rasters
- origin: download.cloud
  destination:
  - group.masks
- origin: group.raster_groups
  destination:
  - merge.raster_group
- origin: group.mask_groups
  destination:
  - merge.mask_group
description:
  short_description: Downloads and preprocesses Sentinel-2 imagery that covers the
    input geometry and time range.
  long_description: This workflow selects a minimum set of tiles that covers the input
    geometry, downloads Sentinel-2 imagery for the selected time range, and preprocesses
    it by generating a single multi-band raster at 10m resolution.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    raster: Sentinel-2 L2A rasters with all bands resampled to 10m resolution.
    mask: Cloud mask at 10m resolution from the product's quality indicators.
  parameters:
    min_tile_cover: Minimum RoI coverage to consider a set of tiles sufficient.
    max_tiles_per_time: Maximum number of tiles used to cover the RoI in each date.
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>raster]
    out2>mask]
    tsk1{{list}}
    tsk2{{filter}}
    tsk3{{download}}
    tsk4{{group}}
    tsk5{{merge}}
    tsk1{{list}} -- sentinel_products/items --> tsk2{{filter}}
    tsk2{{filter}} -- filtered_items/sentinel_product --> tsk3{{download}}
    tsk3{{download}} -- raster/rasters --> tsk4{{group}}
    tsk3{{download}} -- cloud/masks --> tsk4{{group}}
    tsk4{{group}} -- raster_groups/raster_group --> tsk5{{merge}}
    tsk4{{group}} -- mask_groups/mask_group --> tsk5{{merge}}
    inp1>user_input] -- input_item --> tsk1{{list}}
    inp1>user_input] -- bounds_items --> tsk2{{filter}}
    tsk5{{merge}} -- output_raster --> out1>raster]
    tsk5{{merge}} -- output_mask --> out2>mask]
```