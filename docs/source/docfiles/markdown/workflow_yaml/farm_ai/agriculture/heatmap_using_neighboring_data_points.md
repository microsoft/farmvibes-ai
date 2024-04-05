# farm_ai/agriculture/heatmap_using_neighboring_data_points

Creates heatmap using the neighbors by performing spatial interpolation operations. It utilizes soil information collected at optimal sensor/sample locations and downloaded sentinel satellite imagery. The optimal location of nutrient samples are identified using workflow <farm_ai/sensor/optimal_locations>. The quantity of samples defines the accuracy of the heatmap generation. During the research performed testing on a 100 acre farm using sample count of approximately 20, 80, 130, 600. The research concluded that a sample count of 20 provided decent results, also accuracy of nutrient information improved with increase in sample count.

```{mermaid}
    graph TD
    inp1>input_raster]
    inp2>input_samples]
    inp3>input_sample_clusters]
    out1>result]
    tsk1{{download_samples}}
    tsk2{{download_sample_clusters}}
    tsk3{{soil_sample_heatmap}}
    tsk1{{download_samples}} -- geometry/samples --> tsk3{{soil_sample_heatmap}}
    tsk2{{download_sample_clusters}} -- geometry/samples_boundary --> tsk3{{soil_sample_heatmap}}
    inp1>input_raster] -- raster --> tsk3{{soil_sample_heatmap}}
    inp2>input_samples] -- user_input --> tsk1{{download_samples}}
    inp3>input_sample_clusters] -- user_input --> tsk2{{download_sample_clusters}}
    tsk3{{soil_sample_heatmap}} -- result --> out1>result]
```

## Sources

- **input_raster**: Sentinel-2 raster.

- **input_samples**: Sensor samples with nutrient information.

- **input_sample_clusters**: Clusters boundaries of sensor samples locations.

## Sinks

- **result**: Zip file containing heatmap output as shape files.

## Parameters

- **attribute_name**: Nutrient property name in sensor samples geojson file. For example: CARBON (C), Nitrogen (N), Phosphorus (P) etc.,

- **simplify**: Replace small polygons in input with value of their largest neighbor after converting from raster to vector. Accepts 'simplify' or 'convex' or 'none'.

- **tolerance**: All parts of a [simplified geometry](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html) will be no more than tolerance distance from the original. It has the same units as the coordinate reference system of the GeoSeries. For example, using tolerance=100 in a projected CRS with meters as units means a distance of 100 meters in reality.

- **algorithm**: Algorithm used to identify nearest neighbors. Accepts 'cluster overlap' or 'nearest neighbor' or 'kriging neighbor'.

- **resolution**: Defines the output resolution as the ratio of input raster resolution. For example, if resolution is 5, the output heatmap is 5 times coarser than input raster.

- **bins**: it defines the number of equal-width bins in the given range.Refer to this article to learn more about bins https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

## Tasks

- **download_samples**: Adds user geometries into the cluster storage, allowing for them to be used on workflows.

- **download_sample_clusters**: Adds user geometries into the cluster storage, allowing for them to be used on workflows.

- **soil_sample_heatmap**: Generate heatmap for nutrients using satellite or spaceEye imagery.

## Workflow Yaml

```yaml

name: heatmap_using_neighboring_data_points
sources:
  input_raster:
  - soil_sample_heatmap.raster
  input_samples:
  - download_samples.user_input
  input_sample_clusters:
  - download_sample_clusters.user_input
sinks:
  result: soil_sample_heatmap.result
parameters:
  attribute_name: null
  simplify: null
  tolerance: null
  algorithm: null
  resolution: null
  bins: null
tasks:
  download_samples:
    workflow: data_ingestion/user_data/ingest_geometry
  download_sample_clusters:
    workflow: data_ingestion/user_data/ingest_geometry
  soil_sample_heatmap:
    op: soil_sample_heatmap_using_neighbors
    op_dir: heatmap_sensor
    parameters:
      attribute_name: '@from(attribute_name)'
      simplify: '@from(simplify)'
      tolerance: '@from(tolerance)'
      algorithm: '@from(algorithm)'
      resolution: '@from(resolution)'
      bins: '@from(bins)'
edges:
- origin: download_samples.geometry
  destination:
  - soil_sample_heatmap.samples
- origin: download_sample_clusters.geometry
  destination:
  - soil_sample_heatmap.samples_boundary
description:
  short_description: Creates heatmap using the neighbors by performing spatial interpolation
    operations. It utilizes soil information collected at optimal sensor/sample locations
    and downloaded sentinel satellite imagery.
  long_description: The optimal location of nutrient samples are identified using
    workflow <farm_ai/sensor/optimal_locations>. The quantity of samples defines the
    accuracy of the heatmap generation. During the research performed testing on a
    100 acre farm using sample count of approximately 20, 80, 130, 600. The research
    concluded that a sample count of 20 provided decent results, also accuracy of
    nutrient information improved with increase in sample count.
  sources:
    input_raster: Sentinel-2 raster.
    input_samples: Sensor samples with nutrient information.
    input_sample_clusters: Clusters boundaries of sensor samples locations.
  sinks:
    result: Zip file containing heatmap output as shape files.
  parameters:
    attribute_name: 'Nutrient property name in sensor samples geojson file. For example:
      CARBON (C), Nitrogen (N), Phosphorus (P) etc.,'
    simplify: Replace small polygons in input with value of their largest neighbor
      after converting from raster to vector. Accepts 'simplify' or 'convex' or 'none'.
    tolerance: All parts of a [simplified geometry](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html)
      will be no more than tolerance distance from the original. It has the same units
      as the coordinate reference system of the GeoSeries. For example, using tolerance=100
      in a projected CRS with meters as units means a distance of 100 meters in reality.
    algorithm: Algorithm used to identify nearest neighbors. Accepts 'cluster overlap'
      or 'nearest neighbor' or 'kriging neighbor'.
    resolution: Defines the output resolution as the ratio of input raster resolution.
      For example, if resolution is 5, the output heatmap is 5 times coarser than
      input raster.
    bins: it defines the number of equal-width bins in the given range.Refer to this
      article to learn more about bins https://numpy.org/doc/stable/reference/generated/numpy.histogram.html


```