# farm_ai/sensor/optimal_locations

Identify optimal locations by performing clustering operation using Gaussian Mixture model on computed raster indices. The clustering operation separate computed raster indices values into n groups of equal variance, each group assigned a location and that location is considered as a
optimal locations. The sample locations generated provide information of latitude and longitude. The optimal location can be utilized to install sensors and collect
soil information. The index parameter used as input to run the computed index workflow internally using the input raster submitted. The selection of index parameter varies
based on requirement. The workflow supports all the indices supported by spyndex library (https://github.com/awesome-spectral-indices/awesome-spectral-indices#vegetation).
Below provided various indices that are used to identify optimal locations and generated a nutrients heatmap.
Enhanced Vegetation Index (EVI) - EVI is designed to minimize the influence of soil brightness and atmospheric conditions on vegetation assessment. It is calculated
using the red, blue, and near-infrared (NIR) bands. EVI is particularly useful for monitoring vegetation in regions with high canopy cover and in areas where atmospheric
interference is significant. This indices also used in notebook (notebooks/heatmaps/nutrients_using_neighbors.ipynb) that derive nutrient information for Carbon, Nitrogen,
and Phosphorus.
Photochemical Reflectance Index (PRI) - It is a vegetation index used to assess the light-use efficiency of plants in terms of photosynthesis and their response to
changes in light conditions, particularly variations in the blue and red parts of the electromagnetic spectrum. This index also used in notebook
(notebooks/heatmaps/nutrients_using_neighbors.ipynb) that derive nutrient information for pH.
The number of sample locations generated depend on input parameters submitted. Tune n_clusters and sieve_size parameters to generate more or less location data points.
For a 100 acre farm, 
- 20 sample locations are generated using n_clusters=5 and sieve_size=10.
- 30 sample locations are generated using n_clusters=5 and sieve_size=20.
- 80 sample locations are generated using n_clusters=5 and sieve_size=5.
- 130 sample locations are generated using n_clusters=8 and sieve_size=5.

```{mermaid}
    graph TD
    inp1>user_input]
    inp2>input_raster]
    out1>result]
    tsk1{{compute_index}}
    tsk2{{find_samples}}
    tsk1{{compute_index}} -- index_raster/raster --> tsk2{{find_samples}}
    inp1>user_input] -- user_input --> tsk2{{find_samples}}
    inp2>input_raster] -- raster --> tsk1{{compute_index}}
    tsk2{{find_samples}} -- locations --> out1>result]
```

## Sources

- **input_raster**: List of computed raster indices generated using the sentinel 2 satellite imagery.

- **user_input**: DataVibe with time range information.

## Sinks

- **result**: Zip file containing sample locations in a shape file (.shp) format.

## Parameters

- **n_clusters**: number of clusters used to generate sample locations.

- **sieve_size**: Group the nearest neighbor pixel values.

- **index**: Index used to generate sample locations.

## Tasks

- **compute_index**: Computes an index from the bands of an input raster.

- **find_samples**: Find minimum soil sample locations by grouping indices values that are derived from satellite or spaceEye imagery bands.

## Workflow Yaml

```yaml

name: optimal_locations
sources:
  user_input:
  - find_samples.user_input
  input_raster:
  - compute_index.raster
sinks:
  result: find_samples.locations
parameters:
  n_clusters: null
  sieve_size: null
  index: null
tasks:
  compute_index:
    workflow: data_processing/index/index
    parameters:
      index: '@from(index)'
  find_samples:
    op: find_soil_sample_locations
    op_dir: minimum_samples
    parameters:
      n_clusters: '@from(n_clusters)'
      sieve_size: '@from(sieve_size)'
edges:
- origin: compute_index.index_raster
  destination:
  - find_samples.raster
description:
  short_description: Identify optimal locations by performing clustering operation
    using Gaussian Mixture model on computed raster indices.
  long_description: "The clustering operation separate computed raster indices values\
    \ into n groups of equal variance, each group assigned a location and that location\
    \ is considered as a\noptimal locations. The sample locations generated provide\
    \ information of latitude and longitude. The optimal location can be utilized\
    \ to install sensors and collect\nsoil information. The index parameter used as\
    \ input to run the computed index workflow internally using the input raster submitted.\
    \ The selection of index parameter varies\nbased on requirement. The workflow\
    \ supports all the indices supported by spyndex library (https://github.com/awesome-spectral-indices/awesome-spectral-indices#vegetation).\n\
    Below provided various indices that are used to identify optimal locations and\
    \ generated a nutrients heatmap.\nEnhanced Vegetation Index (EVI) - EVI is designed\
    \ to minimize the influence of soil brightness and atmospheric conditions on vegetation\
    \ assessment. It is calculated\nusing the red, blue, and near-infrared (NIR) bands.\
    \ EVI is particularly useful for monitoring vegetation in regions with high canopy\
    \ cover and in areas where atmospheric\ninterference is significant. This indices\
    \ also used in notebook (notebooks/heatmaps/nutrients_using_neighbors.ipynb) that\
    \ derive nutrient information for Carbon, Nitrogen,\nand Phosphorus.\nPhotochemical\
    \ Reflectance Index (PRI) - It is a vegetation index used to assess the light-use\
    \ efficiency of plants in terms of photosynthesis and their response to\nchanges\
    \ in light conditions, particularly variations in the blue and red parts of the\
    \ electromagnetic spectrum. This index also used in notebook\n(notebooks/heatmaps/nutrients_using_neighbors.ipynb)\
    \ that derive nutrient information for pH.\nThe number of sample locations generated\
    \ depend on input parameters submitted. Tune n_clusters and sieve_size parameters\
    \ to generate more or less location data points.\nFor a 100 acre farm, \n- 20\
    \ sample locations are generated using n_clusters=5 and sieve_size=10.\n- 30 sample\
    \ locations are generated using n_clusters=5 and sieve_size=20.\n- 80 sample locations\
    \ are generated using n_clusters=5 and sieve_size=5.\n- 130 sample locations are\
    \ generated using n_clusters=8 and sieve_size=5."
  sources:
    input_raster: List of computed raster indices generated using the sentinel 2 satellite
      imagery.
    user_input: DataVibe with time range information.
  sinks:
    result: Zip file containing sample locations in a shape file (.shp) format.
  parameters:
    n_clusters: number of clusters used to generate sample locations.
    sieve_size: Group the nearest neighbor pixel values.
    index: Index used to generate sample locations.


```