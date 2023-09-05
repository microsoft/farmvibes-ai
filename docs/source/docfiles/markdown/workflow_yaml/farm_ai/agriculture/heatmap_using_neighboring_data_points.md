# farm_ai/agriculture/heatmap_using_neighboring_data_points

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
  distribute_output: null
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
      distribute_output: '@from(distribute_output)'
      bins: '@from(bins)'
edges:
- origin: download_samples.geometry
  destination:
  - soil_sample_heatmap.samples
- origin: download_sample_clusters.geometry
  destination:
  - soil_sample_heatmap.samples_boundary
description:
  short_description: Create heatmap using the neighbors by performing spatial interpolation
    operations. It utilize soil information collected at optimal sensor/sample locations
    and downloaded sentinel satellite imagery.
  long_description: The optimal location of nutrient samples are identified using
    workflow <farm_ai/sensor/optimal_locations>. The quantity of samples define the
    accuracy of the heatmap generation. During the research performed testing using
    sample count approximately 20, 80, 130, 600. The research concluded samples count
    20 provided decent results, also accuracy of nutrient information improved with
    increase in sample count.
  sources:
    input_raster: sentinel 2 satellite imagery.
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
    distribute_output: Increases the output variance to avoid output polygon in shape
      file grouped into single large polygon.
    algorithm: Algorithm used to identify nearest neighbors. Accepts 'cluster overlap'
      or 'nearest neighbor' or 'kriging neighbor'.
    resolution: Resolution of the heatmap, units of resolution should match units
      of input raster.
    bins: it defines the number of equal-width bins in the given range.Refer to this
      article to learn more about bins https://numpy.org/doc/stable/reference/generated/numpy.histogram.html


```

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