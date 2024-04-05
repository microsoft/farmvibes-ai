# farm_ai/land_cover_mapping/conservation_practices

Identifies conservation practices (terraces and grassed waterways) using elevation data. The workflow classifies pixels in terraces or grassed waterways. It starts downloading NAIP and USGS 3DEP tiles. Then, it computes the elevation gradient using a Sobel filter. And it computes local clusters using an overlap clustering method. Then, it combines cluster and elevation tiles to compute the average elevation per cluster. Finally, it uses a CNN model to classify pixels in either terraces or grassed waterways.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>dem_raster]
    out2>naip_raster]
    out3>dem_gradient]
    out4>cluster]
    out5>average_elevation]
    out6>practices]
    tsk1{{naip}}
    tsk2{{cluster}}
    tsk3{{dem}}
    tsk4{{gradient}}
    tsk5{{match_grad}}
    tsk6{{match_elev}}
    tsk7{{avg_elev}}
    tsk8{{practice}}
    tsk1{{naip}} -- raster/user_input --> tsk3{{dem}}
    tsk1{{naip}} -- raster/input_raster --> tsk2{{cluster}}
    tsk1{{naip}} -- raster/ref_rasters --> tsk6{{match_elev}}
    tsk1{{naip}} -- raster/ref_rasters --> tsk5{{match_grad}}
    tsk3{{dem}} -- raster --> tsk4{{gradient}}
    tsk3{{dem}} -- raster/rasters --> tsk6{{match_elev}}
    tsk4{{gradient}} -- gradient/rasters --> tsk5{{match_grad}}
    tsk2{{cluster}} -- output_raster/input_cluster_raster --> tsk7{{avg_elev}}
    tsk6{{match_elev}} -- match_rasters/input_dem_raster --> tsk7{{avg_elev}}
    tsk7{{avg_elev}} -- output_raster/average_elevation --> tsk8{{practice}}
    tsk5{{match_grad}} -- match_rasters/elevation_gradient --> tsk8{{practice}}
    inp1>user_input] -- user_input --> tsk1{{naip}}
    tsk3{{dem}} -- raster --> out1>dem_raster]
    tsk1{{naip}} -- raster --> out2>naip_raster]
    tsk4{{gradient}} -- gradient --> out3>dem_gradient]
    tsk2{{cluster}} -- output_raster --> out4>cluster]
    tsk7{{avg_elev}} -- output_raster --> out5>average_elevation]
    tsk8{{practice}} -- output_raster --> out6>practices]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **dem_raster**: USGS 3DEP tiles that overlap the NAIP tiles that overlap the area of interest.

- **naip_raster**: NAIP tiles that overlap the area of interest.

- **dem_gradient**: A copy of the USGS 3DEP tiles where the pixel values are the gradient computed using the Sobel filter.

- **cluster**: A copy of the NAIP tiles with one band representing the output of the overlap clustering method. Each pixel has a value between one and four.

- **average_elevation**: A combination of the dem_gradient and cluster sinks, where each pixel value is the average elevation of all pixels that fall in the same cluster.

- **practices**: A copy of the NAIP tile with one band where each pixel value refers to a conservation practice (0 = none, 1 = terraces, 2 = grassed waterways).

## Parameters

- **clustering_iterations**: The number of iterations used in the overlap clustering method.

- **pc_key**: Optional Planetary Computer API key.

## Tasks

- **naip**: Downloads NAIP tiles that intersect with the input geometry and time range.

- **cluster**: Computes local clusters using an overlap clustering method.

- **dem**: Downloads digital elevation map tiles that intersect with the input geometry and time range.

- **gradient**: Computes the gradient of each band of the input raster with a Sobel operator.

- **match_grad**: Resamples input rasters to the reference rasters' grid.

- **match_elev**: Resamples input rasters to the reference rasters' grid.

- **avg_elev**: Computes average elevation per-class in overlapping windows, combining cluster and elevation tiles.

- **practice**: Classifies pixels in either terraces or grassed waterways using a CNN model.

## Workflow Yaml

```yaml

name: conservation_practices
sources:
  user_input:
  - naip.user_input
sinks:
  dem_raster: dem.raster
  naip_raster: naip.raster
  dem_gradient: gradient.gradient
  cluster: cluster.output_raster
  average_elevation: avg_elev.output_raster
  practices: practice.output_raster
parameters:
  clustering_iterations: null
  pc_key: null
tasks:
  naip:
    workflow: data_ingestion/naip/download_naip
    parameters:
      pc_key: '@from(pc_key)'
  cluster:
    op: compute_raster_cluster
    parameters:
      number_iterations: '@from(clustering_iterations)'
  dem:
    workflow: data_ingestion/dem/download_dem
    parameters:
      pc_key: '@from(pc_key)'
  gradient:
    workflow: data_processing/gradient/raster_gradient
  match_grad:
    workflow: data_processing/merge/match_merge_to_ref
  match_elev:
    workflow: data_processing/merge/match_merge_to_ref
  avg_elev:
    op: compute_raster_class_windowed_average
  practice:
    op: compute_conservation_practice
edges:
- origin: naip.raster
  destination:
  - dem.user_input
  - cluster.input_raster
  - match_elev.ref_rasters
  - match_grad.ref_rasters
- origin: dem.raster
  destination:
  - gradient.raster
  - match_elev.rasters
- origin: gradient.gradient
  destination:
  - match_grad.rasters
- origin: cluster.output_raster
  destination:
  - avg_elev.input_cluster_raster
- origin: match_elev.match_rasters
  destination:
  - avg_elev.input_dem_raster
- origin: avg_elev.output_raster
  destination:
  - practice.average_elevation
- origin: match_grad.match_rasters
  destination:
  - practice.elevation_gradient
description:
  short_description: Identifies conservation practices (terraces and grassed waterways)
    using elevation data.
  long_description: The workflow classifies pixels in terraces or grassed waterways.
    It starts downloading NAIP and USGS 3DEP tiles. Then, it computes the elevation
    gradient using a Sobel filter. And it computes local clusters using an overlap
    clustering method. Then, it combines cluster and elevation tiles to compute the
    average elevation per cluster. Finally, it uses a CNN model to classify pixels
    in either terraces or grassed waterways.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    dem_raster: USGS 3DEP tiles that overlap the NAIP tiles that overlap the area
      of interest.
    naip_raster: NAIP tiles that overlap the area of interest.
    dem_gradient: A copy of the USGS 3DEP tiles where the pixel values are the gradient
      computed using the Sobel filter.
    cluster: A copy of the NAIP tiles with one band representing the output of the
      overlap clustering method. Each pixel has a value between one and four.
    average_elevation: A combination of the dem_gradient and cluster sinks, where
      each pixel value is the average elevation of all pixels that fall in the same
      cluster.
    practices: A copy of the NAIP tile with one band where each pixel value refers
      to a conservation practice (0 = none, 1 = terraces, 2 = grassed waterways).
  parameters:
    clustering_iterations: The number of iterations used in the overlap clustering
      method.
    pc_key: Optional Planetary Computer API key.


```