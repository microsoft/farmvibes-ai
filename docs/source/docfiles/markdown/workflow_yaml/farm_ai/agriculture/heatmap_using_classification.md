# farm_ai/agriculture/heatmap_using_classification

The workflow generates a nutrient heatmap for samples provided by user by downloading the samples from user input. The samples provided are related with farm boundary and have required nutrient information to create a heatmap.

```{mermaid}
    graph TD
    inp1>input_samples]
    inp2>input_raster]
    out1>result]
    tsk1{{download_samples}}
    tsk2{{soil_sample_heatmap_classification}}
    tsk1{{download_samples}} -- geometry/samples --> tsk2{{soil_sample_heatmap_classification}}
    inp1>input_samples] -- user_input --> tsk1{{download_samples}}
    inp2>input_raster] -- input_raster --> tsk2{{soil_sample_heatmap_classification}}
    tsk2{{soil_sample_heatmap_classification}} -- result --> out1>result]
```

## Sources

- **input_raster**: Input raster for index computation.

- **input_samples**: External references to sensor samples for nutrients.

## Sinks

- **result**: Zip file containing cluster geometries.

## Parameters

- **attribute_name**: Nutrient property name in sensor samples geojson file. For example CARBON (C), Nitrogen (N), Phosphorus (P) etc.,

- **buffer**: Offset distance from sample to perform interpolate operations with raster.

- **index**: Type of index to be used to generate heatmap. For example - evi, pri etc.,

- **bins**: Possible number of groups used to move value to nearest group using [numpy histogram](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) and to pre-process the data to support model training with classification .

- **simplify**: Replace small polygons in input with value of their largest neighbor after converting from raster to vector. Accepts 'simplify' or 'convex' or 'none'.

- **tolerance**: All parts of a [simplified geometry](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html) will be no more than tolerance distance from the original. It has the same units as the coordinate reference system of the GeoSeries. For example, using tolerance=100 in a projected CRS with meters as units means a distance of 100 meters in reality.

- **data_scale**: Accepts True or False. Default is False. On True, it scale data using [StandardScalar] (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) from scikit-learn package.  It Standardize features by removing the mean and scaling to unit variance.

- **max_depth**: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

- **n_estimators**: The number of trees in the forest. For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

- **random_state**: Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Tasks

- **download_samples**: Adds user geometries into the cluster storage, allowing for them to be used on workflows.

- **soil_sample_heatmap_classification**: Utilizes input Sentinel-2 satellite imagery & the sensor samples as labeled data that contain nutrient information (Nitrogen, Carbon, pH, Phosphorus) to train a model using Random Forest classifier. The inference operation predicts nutrients in soil for the chosen farm boundary.


## Workflow Yaml

```yaml

name: heatmap_using_classification
sources:
  input_samples:
  - download_samples.user_input
  input_raster:
  - soil_sample_heatmap_classification.input_raster
sinks:
  result: soil_sample_heatmap_classification.result
parameters:
  attribute_name: null
  buffer: null
  index: null
  bins: null
  simplify: null
  tolerance: null
  data_scale: null
  max_depth: null
  n_estimators: null
  random_state: null
tasks:
  download_samples:
    workflow: data_ingestion/user_data/ingest_geometry
  soil_sample_heatmap_classification:
    workflow: data_processing/heatmap/classification
    parameters:
      attribute_name: '@from(attribute_name)'
      buffer: '@from(buffer)'
      index: '@from(index)'
      bins: '@from(bins)'
      simplify: '@from(simplify)'
      tolerance: '@from(tolerance)'
      data_scale: '@from(data_scale)'
      max_depth: '@from(max_depth)'
      n_estimators: '@from(n_estimators)'
      random_state: '@from(random_state)'
edges:
- origin: download_samples.geometry
  destination:
  - soil_sample_heatmap_classification.samples
description:
  short_description: The workflow generates a nutrient heatmap for samples provided
    by user by downloading the samples from user input.
  long_description: The samples provided are related with farm boundary and have required
    nutrient information to create a heatmap.
  sources:
    input_raster: Input raster for index computation.
    input_samples: External references to sensor samples for nutrients.
  sinks:
    result: Zip file containing cluster geometries.
  parameters: null


```