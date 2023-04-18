# data_ingestion/spaceeye/spaceeye_preprocess_ensemble

```yaml

name: spaceeye_preprocess_ensemble
sources:
  user_input:
  - s2.user_input
  - s1.user_input
sinks:
  s2_raster: s2.raster
  s1_raster: s1.raster
  cloud_mask: s2.mask
parameters:
  pc_key: null
tasks:
  s2:
    workflow: data_ingestion/sentinel2/preprocess_s2_ensemble_masks
    parameters:
      pc_key: '@from(pc_key)'
  s1:
    workflow: data_ingestion/sentinel1/preprocess_s1
    parameters:
      pc_key: '@from(pc_key)'
edges:
- origin: s2.raster
  destination:
  - s1.s2_products
description:
  short_description: Runs the SpaceEye preprocessing pipeline with an ensemble of
    cloud segmentation models.
  long_description: The workflow fetches both Sentinel-1 and Sentinel-2 tiles that
    cover the input geometry and time range and preprocesses them, it also computes
    improved cloud masks using cloud and shadow segmentation models. Cloud probabilities
    are computed with an ensemble of five models.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    s2_raster: Sentinel-2 rasters.
    s1_raster: Sentinel-1 rasters.
    cloud_mask: Cloud and cloud shadow mask.
  parameters:
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>s2_raster]
    out2>s1_raster]
    out3>cloud_mask]
    tsk1{{s2}}
    tsk2{{s1}}
    tsk1{{s2}} -- raster/s2_products --> tsk2{{s1}}
    inp1>user_input] -- user_input --> tsk1{{s2}}
    inp1>user_input] -- user_input --> tsk2{{s1}}
    tsk1{{s2}} -- raster --> out1>s2_raster]
    tsk2{{s1}} -- raster --> out2>s1_raster]
    tsk1{{s2}} -- mask --> out3>cloud_mask]
```