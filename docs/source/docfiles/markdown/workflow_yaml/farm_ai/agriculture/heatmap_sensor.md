# farm_ai/agriculture/heatmap_sensor

```yaml

name: heatmap_sensor
sources:
  input_samples:
  - download_samples.user_input
  input_raster:
  - compute_index.raster
sinks:
  result: soil_sample_heatmap.result
parameters:
  attribute_name: null
  buffer: null
  index: null
  bins: null
  simplify: null
  tolerance: null
  data_scale: null
  distribute_output: null
  max_depth: null
  n_estimators: null
  random_state: null
tasks:
  compute_index:
    workflow: data_processing/index/index
    parameters:
      index: '@from(index)'
  download_samples:
    workflow: data_ingestion/user_data/ingest_geometry
  soil_sample_heatmap:
    op: soil_sample_heatmap
    op_dir: heatmap_sensor
    parameters:
      attribute_name: '@from(attribute_name)'
      buffer: '@from(buffer)'
      bins: '@from(bins)'
      simplify: '@from(simplify)'
      tolerance: '@from(tolerance)'
      data_scale: '@from(data_scale)'
      distribute_output: '@from(distribute_output)'
      max_depth: '@from(max_depth)'
      n_estimators: '@from(n_estimators)'
      random_state: '@from(random_state)'
edges:
- origin: compute_index.index_raster
  destination:
  - soil_sample_heatmap.raster
- origin: download_samples.geometry
  destination:
  - soil_sample_heatmap.samples
description:
  short_description: Utilizes input Sentinel-2 satellite imagery & the sensor samples
    as labeled data that contain nutrient information (Nitrogen, Carbon, pH, Phosphorus)
    to train a model using Random Forest classifier. The inference operation predicts
    nutrients in soil for the chosen farm boundary.
  long_description: The workflow generates a heatmap for selected nutrient. It relies
    on sample soil data that contain information of nutrients. The quantity of samples
    define the accuracy of the heat map generation. During the research performed
    testing with samples spaced at 200 feet, 100 feet and 50 feet. The 50 feet sample
    spaced distance provided results matching to the ground truth. Generating heatmap
    with this approach reduce the number of samples. It utilizes the logic below behind
    the scenes to generate heatmap. - Read the sentinel raster provided. - Download
    sensor samples for the input url provided. - Compute indices using the spyndex
    python package - Clip the satellite imagery & sensor samples using farm boundary.
    - Perform spatial interpolation to find raster pixels within the offset distance
    from sample location and assign the value of nutrients to group of pixels. - Classify
    the data based on number of bins. - Train the model using Random Forest classifier.
    - Predict the nutrients using the satellite imagery. - Generate a shape file using
    the predicted outputs.
  sources:
    input_raster: Input raster for index computation.
    input_samples: External references to sensor samples for nutrients.
  sinks:
    result: Zip file containing cluster geometries.
  parameters:
    attribute_name:
      Nutrient property name in sensor samples geojson file. For example: CARBON (C),
        Nitrogen (N), Phosphorus (P) etc.,
    buffer: Offset distance from sample to perform interpolate operations with raster.
    index: Type of index to be used to generate heatmap. For example - evi, pri etc.,
    bins: Possible number of groups used to move value to nearest group using [numpy
      histogram](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html)
      and to pre-process the data to support model training with classification .
    simplify: Replace small polygons in input with value of their largest neighbor
      after converting from raster to vector. Accepts 'simplify' or 'convex' or 'none'.
    tolerance: All parts of a [simplified geometry](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html)
      will be no more than tolerance distance from the original. It has the same units
      as the coordinate reference system of the GeoSeries. For example, using tolerance=100
      in a projected CRS with meters as units means a distance of 100 meters in reality.
    data_scale: Accepts True or False. Default is False. On True, it scale data using
      [StandardScalar] (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
      from scikit-learn package.  It Standardize features by removing the mean and
      scaling to unit variance.
    distribute_output: Increases the output variance to avoid output polygon in shape
      file grouped into single large polygon.
    max_depth: The maximum depth of the tree. If None, then nodes are expanded until
      all leaves are pure or until all leaves contain less than min_samples_split
      samples. For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    n_estimators: The number of trees in the forest. For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    random_state: Controls both the randomness of the bootstrapping of the samples
      used when building trees (if bootstrap=True) and the sampling of the features
      to consider when looking for the best split at each node (if max_features <
      n_features). For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


```

```{mermaid}
    graph TD
    inp1>input_samples]
    inp2>input_raster]
    out1>result]
    tsk1{{compute_index}}
    tsk2{{download_samples}}
    tsk3{{soil_sample_heatmap}}
    tsk1{{compute_index}} -- index_raster/raster --> tsk3{{soil_sample_heatmap}}
    tsk2{{download_samples}} -- geometry/samples --> tsk3{{soil_sample_heatmap}}
    inp1>input_samples] -- user_input --> tsk2{{download_samples}}
    inp2>input_raster] -- raster --> tsk1{{compute_index}}
    tsk3{{soil_sample_heatmap}} -- result --> out1>result]
```