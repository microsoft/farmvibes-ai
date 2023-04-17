# data_processing/merge/match_merge_to_ref

```yaml

name: match_merge_to_ref
sources:
  rasters:
  - pair.rasters2
  ref_rasters:
  - pair.rasters1
  - group.group_by
sinks:
  match_rasters: merge.raster
parameters:
  resampling: bilinear
tasks:
  pair:
    op: pair_intersecting_rasters
  match:
    op: match_raster_to_ref
    parameters:
      resampling: '@from(resampling)'
  group:
    op: group_rasters_by_geometries
  merge:
    op: merge_rasters
    parameters:
      resampling: '@from(resampling)'
edges:
- origin: pair.paired_rasters1
  destination:
  - match.ref_raster
- origin: pair.paired_rasters2
  destination:
  - match.raster
- origin: match.output_raster
  destination:
  - group.rasters
- origin: group.raster_groups
  destination:
  - merge.raster_sequence
description:
  short_description: Resamples input rasters to the reference rasters' grid.
  long_description: The workflow will produce input and reference raster pairs with
    intersecting geometries. For each pair, the input raster is resampled to match
    the reference raster's grid. Afterwards, all resampled rasters are groupped if
    they are contained in a reference raster geometry, and each raster group is matched
    into single raster. The output should contain the information available in the
    input rasters, gridded according to the reference rasters.
  sources:
    rasters: Input rasters that will be resampled.
    ref_rasters: Reference rasters.
  sinks:
    match_rasters: Rasters with information from the input rasters on the reference
      grid.
  parameters:
    resampling: 'Type of resampling when reprojecting the rasters. See [link=https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling]
      rasterio documentation: https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling[/]
      for all available resampling options.'


```

```{mermaid}
    graph TD
    inp1>rasters]
    inp2>ref_rasters]
    out1>match_rasters]
    tsk1{{pair}}
    tsk2{{match}}
    tsk3{{group}}
    tsk4{{merge}}
    tsk1{{pair}} -- paired_rasters1/ref_raster --> tsk2{{match}}
    tsk1{{pair}} -- paired_rasters2/raster --> tsk2{{match}}
    tsk2{{match}} -- output_raster/rasters --> tsk3{{group}}
    tsk3{{group}} -- raster_groups/raster_sequence --> tsk4{{merge}}
    inp1>rasters] -- rasters2 --> tsk1{{pair}}
    inp2>ref_rasters] -- rasters1 --> tsk1{{pair}}
    inp2>ref_rasters] -- group_by --> tsk3{{group}}
    tsk4{{merge}} -- raster --> out1>match_rasters]
```