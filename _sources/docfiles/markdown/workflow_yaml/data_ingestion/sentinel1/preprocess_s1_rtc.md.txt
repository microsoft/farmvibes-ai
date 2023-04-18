# data_ingestion/sentinel1/preprocess_s1_rtc

```yaml

name: preprocess_s1_rtc
sources:
  user_input:
  - merge_geom_tr.time_range
  s2_products:
  - union.items
  - filter.bounds_items
  - tile.sentinel2_products
sinks:
  raster: merge.merged_product
parameters:
  pc_key: null
  min_cover: 0.4
  dl_timeout: null
tasks:
  union:
    op: merge_geometries
  merge_geom_tr:
    op: merge_geometry_and_time_range
  list:
    op: list_sentinel1_products_pc
    op_dir: list_sentinel1_products
    parameters:
      collection: rtc
  filter:
    op: select_necessary_coverage_items
    parameters:
      min_cover: '@from(min_cover)'
      group_attribute: orbit_number
  download:
    op: download_sentinel1_rtc
    parameters:
      api_key: '@from(pc_key)'
      timeout_s: '@from(dl_timeout)'
  tile:
    op: tile_sentinel1_rtc
    op_dir: tile_sentinel1
  group:
    op: group_sentinel1_orbits
  merge:
    op: merge_sentinel1_orbits
edges:
- origin: union.merged
  destination:
  - merge_geom_tr.geometry
- origin: merge_geom_tr.merged
  destination:
  - list.input_item
- origin: list.sentinel_products
  destination:
  - filter.items
- origin: filter.filtered_items
  destination:
  - download.sentinel_product
- origin: download.downloaded_product
  destination:
  - tile.sentinel1_products
- origin: tile.tiled_products
  destination:
  - group.rasters
- origin: group.raster_groups
  destination:
  - merge.raster_group
description:
  short_description: Downloads and preprocesses tiles of Sentinel-1 imagery that intersect
    with the input Sentinel-2 products in the input time range.
  long_description: The workflow fetches Sentinel-1 tiles that intersects with the
    Sentinel-2 products, downloads and preprocesses them, and produces Sentinel-1
    rasters in the Sentinel-2 tiling system.
  sources:
    user_input: Time range of interest.
    s2_products: Sentinel-2 products whose geometries are used to select Sentinel-1
      tiles.
  sinks:
    raster: Sentinel-1 rasters in the Sentinel-2 tiling system.
  parameters:
    pc_key: Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    inp2>s2_products]
    out1>raster]
    tsk1{{union}}
    tsk2{{merge_geom_tr}}
    tsk3{{list}}
    tsk4{{filter}}
    tsk5{{download}}
    tsk6{{tile}}
    tsk7{{group}}
    tsk8{{merge}}
    tsk1{{union}} -- merged/geometry --> tsk2{{merge_geom_tr}}
    tsk2{{merge_geom_tr}} -- merged/input_item --> tsk3{{list}}
    tsk3{{list}} -- sentinel_products/items --> tsk4{{filter}}
    tsk4{{filter}} -- filtered_items/sentinel_product --> tsk5{{download}}
    tsk5{{download}} -- downloaded_product/sentinel1_products --> tsk6{{tile}}
    tsk6{{tile}} -- tiled_products/rasters --> tsk7{{group}}
    tsk7{{group}} -- raster_groups/raster_group --> tsk8{{merge}}
    inp1>user_input] -- time_range --> tsk2{{merge_geom_tr}}
    inp2>s2_products] -- items --> tsk1{{union}}
    inp2>s2_products] -- bounds_items --> tsk4{{filter}}
    inp2>s2_products] -- sentinel2_products --> tsk6{{tile}}
    tsk8{{merge}} -- merged_product --> out1>raster]
```