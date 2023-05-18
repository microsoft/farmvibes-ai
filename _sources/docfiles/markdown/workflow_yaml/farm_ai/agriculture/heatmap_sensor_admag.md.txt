# farm_ai/agriculture/heatmap_sensor_admag

```yaml

name: heatmap_sensor
sources:
  admag_input:
  - prescriptions.admag_input
  input_raster:
  - compute_index.raster
sinks:
  result: soil_sample_heatmap.result
parameters:
  base_url: null
  client_id: null
  client_secret: null
  authority: null
  default_scope: null
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
  prescriptions:
    workflow: data_ingestion/admag/prescriptions
    parameters:
      base_url: '@from(base_url)'
      client_id: '@from(client_id)'
      client_secret: '@from(client_secret)'
      authority: '@from(authority)'
      default_scope: '@from(default_scope)'
  compute_index:
    workflow: data_processing/index/index
    parameters:
      index: '@from(index)'
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
- origin: prescriptions.response
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
    the scenes to generate heatmap. - Read the sentinel raster provided. - Sensor
    samples needs to be uploaded into prescriptions entity in Azure data manager for
    Agriculture (ADMAG). ADMAG is having hierarchy to hold information of Farmer,
    Field, Seasons, Crop, Boundary etc. Prior to uploading prescriptions, it is required
    to build hierarchy and a prescription_map_id. All prescriptions uploaded to ADMAG
    are related to farm hierarchy through prescription_map_id. Please refer to https://learn.microsoft.com/en-us/rest/api/data-manager-for-agri/
    for more information on ADMAG. - Compute indices using the spyndex python package.
    - Clip the satellite imagery & sensor samples using farm boundary. - Perform spatial
    interpolation to find raster pixels within the offset distance from sample location
    and assign the value of nutrients to group of pixels. - Classify the data based
    on number of bins. - Train the model using Random Forest classifier. - Predict
    the nutrients using the satellite imagery. - Generate a shape file using the predicted
    outputs.
  sources:
    input_raster: Input raster for index computation.
    admag_input: Required inputs to download prescriptions from admag.
  sinks:
    result: Zip file containing cluster geometries.
  parameters:
    base_url: URL to access the registered app
    client_id: Value uniquely identifies registered application in the Microsoft identity
      platform. Visit url https://learn.microsoft.com/en-us/azure/active-directory/develop/quickstart-register-app
      to register the app.
    client_secret: Sometimes called an application password, a client secret is a
      string value your app can use in place of a certificate to identity itself.
    authority: The endpoint URIs for your app are generated automatically when you
      register or configure your app. It is used by client to obtain authorization
      from the resource owner
    default_scope: URL for default azure OAuth2 permissions
    attribute_name: Nutrient property name in sensor samples geojson file. For example
      - CARBON (C), Nitrogen (N), Phosphorus (P) etc.,
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
    inp1>admag_input]
    inp2>input_raster]
    out1>result]
    tsk1{{prescriptions}}
    tsk2{{compute_index}}
    tsk3{{soil_sample_heatmap}}
    tsk2{{compute_index}} -- index_raster/raster --> tsk3{{soil_sample_heatmap}}
    tsk1{{prescriptions}} -- response/samples --> tsk3{{soil_sample_heatmap}}
    inp1>admag_input] -- admag_input --> tsk1{{prescriptions}}
    inp2>input_raster] -- raster --> tsk2{{compute_index}}
    tsk3{{soil_sample_heatmap}} -- result --> out1>result]
```