# data_processing/heatmap/classification

Utilizes input Sentinel-2 satellite imagery & the sensor samples as labeled data that contain nutrient information (Nitrogen, Carbon, pH, Phosphorus) to train a model using Random Forest classifier. The inference operation predicts nutrients in soil for the chosen farm boundary.
 The workflow generates a heatmap for selected nutrient. It relies on sample soil data that
contain information of nutrients. The quantity of samples define the accuracy of the heat map
generation. During the research performed testing with samples spaced at 200 feet, 100 feet and
50 feet. The 50 feet sample spaced distance provided results matching to the ground truth.
Generating heatmaps with this approach reduces the number of samples. It utilizes the logic
below behind the scenes to generate heatmap.
  - Read the sentinel raster provided.
  - Sensor samples needs to be uploaded into prescriptions entity in Azure
    data manager for Agriculture (ADMAg). ADMAg is having hierarchy to hold
    information of Party, Field, Seasons, Crop etc. Prior to
    uploading prescriptions, it is required to build hierarchy and
    a `prescription_map_id`. All prescriptions uploaded to ADMAg are
    related to farm hierarchy through `prescription_map_id`. Please refer to
    https://learn.microsoft.com/en-us/rest/api/data-manager-for-agri/ for
    more information on ADMAg.
  - Compute indices using the spyndex python package.
  - Clip the satellite imagery & sensor samples using farm boundary.
  - Perform spatial interpolation to find raster pixels within the offset distance
    from sample location and assign the value of nutrients to group of pixels.
  - Classify the data based on number of bins.
  - Train the model using Random Forest classifier.
  - Predict the nutrients using the satellite imagery.
  - Generate a shape file using the predicted outputs.

```{mermaid}
    graph TD
    inp1>input_raster]
    inp2>samples]
    out1>result]
    tsk1{{compute_index}}
    tsk2{{soil_sample_heatmap}}
    tsk1{{compute_index}} -- index_raster/raster --> tsk2{{soil_sample_heatmap}}
    inp1>input_raster] -- raster --> tsk1{{compute_index}}
    inp2>samples] -- samples --> tsk2{{soil_sample_heatmap}}
    tsk2{{soil_sample_heatmap}} -- result --> out1>result]
```

## Sources

- **input_raster**: Input raster for index computation.

- **samples**: External references to sensor samples for nutrients.

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

- **compute_index**: Computes an index from the bands of an input raster.

- **soil_sample_heatmap**: Generate heatmap for nutrients using satellite or spaceEye imagery.

## Workflow Yaml

```yaml

name: heatmap_intermediate
sources:
  input_raster:
  - compute_index.raster
  samples:
  - soil_sample_heatmap.samples
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
  max_depth: null
  n_estimators: null
  random_state: null
tasks:
  compute_index:
    workflow: data_processing/index/index
    parameters:
      index: '@from(index)'
  soil_sample_heatmap:
    op: soil_sample_heatmap_using_classification
    op_dir: heatmap_sensor
    parameters:
      attribute_name: '@from(attribute_name)'
      buffer: '@from(buffer)'
      bins: '@from(bins)'
      simplify: '@from(simplify)'
      tolerance: '@from(tolerance)'
      data_scale: '@from(data_scale)'
      max_depth: '@from(max_depth)'
      n_estimators: '@from(n_estimators)'
      random_state: '@from(random_state)'
edges:
- origin: compute_index.index_raster
  destination:
  - soil_sample_heatmap.raster
description:
  short_description: 'Utilizes input Sentinel-2 satellite imagery & the sensor samples
    as labeled data that contain nutrient information (Nitrogen, Carbon, pH, Phosphorus)
    to train a model using Random Forest classifier. The inference operation predicts
    nutrients in soil for the chosen farm boundary.

    '
  long_description: "The workflow generates a heatmap for selected nutrient. It relies\
    \ on sample soil data that\ncontain information of nutrients. The quantity of\
    \ samples define the accuracy of the heat map\ngeneration. During the research\
    \ performed testing with samples spaced at 200 feet, 100 feet and\n50 feet. The\
    \ 50 feet sample spaced distance provided results matching to the ground truth.\n\
    Generating heatmaps with this approach reduces the number of samples. It utilizes\
    \ the logic\nbelow behind the scenes to generate heatmap.\n  - Read the sentinel\
    \ raster provided.\n  - Sensor samples needs to be uploaded into prescriptions\
    \ entity in Azure\n    data manager for Agriculture (ADMAg). ADMAg is having hierarchy\
    \ to hold\n    information of Party, Field, Seasons, Crop etc. Prior to\n    uploading\
    \ prescriptions, it is required to build hierarchy and\n    a `prescription_map_id`.\
    \ All prescriptions uploaded to ADMAg are\n    related to farm hierarchy through\
    \ `prescription_map_id`. Please refer to\n    https://learn.microsoft.com/en-us/rest/api/data-manager-for-agri/\
    \ for\n    more information on ADMAg.\n  - Compute indices using the spyndex python\
    \ package.\n  - Clip the satellite imagery & sensor samples using farm boundary.\n\
    \  - Perform spatial interpolation to find raster pixels within the offset distance\n\
    \    from sample location and assign the value of nutrients to group of pixels.\n\
    \  - Classify the data based on number of bins.\n  - Train the model using Random\
    \ Forest classifier.\n  - Predict the nutrients using the satellite imagery.\n\
    \  - Generate a shape file using the predicted outputs."
  sources:
    input_raster: Input raster for index computation.
    samples: External references to sensor samples for nutrients.
  sinks:
    result: Zip file containing cluster geometries.
  parameters:
    attribute_name: Nutrient property name in sensor samples geojson file. For example
      CARBON (C), Nitrogen (N), Phosphorus (P) etc.,
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
    max_depth: The maximum depth of the tree. If None, then nodes are expanded until
      all leaves are pure or until all leaves contain less than min_samples_split
      samples. For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    n_estimators: The number of trees in the forest. For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    random_state: Controls both the randomness of the bootstrapping of the samples
      used when building trees (if bootstrap=True) and the sampling of the features
      to consider when looking for the best split at each node (if max_features <
      n_features). For more details refer to (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


```