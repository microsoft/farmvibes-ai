# data_ingestion/sentinel2/cloud_ensemble

```yaml

name: cloud_ensemble
sources:
  sentinel_raster:
  - cloud1.sentinel_raster
  - cloud2.sentinel_raster
  - cloud3.sentinel_raster
  - cloud4.sentinel_raster
  - cloud5.sentinel_raster
sinks:
  cloud_probability: ensemble.cloud_probability
tasks:
  cloud1:
    op: compute_cloud_prob
    parameters:
      model_path: cloud_model1_cpu.onnx
  cloud2:
    op: compute_cloud_prob
    parameters:
      model_path: cloud_model2_cpu.onnx
  cloud3:
    op: compute_cloud_prob
    parameters:
      model_path: cloud_model3_cpu.onnx
  cloud4:
    op: compute_cloud_prob
    parameters:
      model_path: cloud_model4_cpu.onnx
  cloud5:
    op: compute_cloud_prob
    parameters:
      model_path: cloud_model5_cpu.onnx
  ensemble:
    op: ensemble_cloud_prob
edges:
- origin: cloud1.cloud_probability
  destination:
  - ensemble.cloud1
- origin: cloud2.cloud_probability
  destination:
  - ensemble.cloud2
- origin: cloud3.cloud_probability
  destination:
  - ensemble.cloud3
- origin: cloud4.cloud_probability
  destination:
  - ensemble.cloud4
- origin: cloud5.cloud_probability
  destination:
  - ensemble.cloud5
description:
  short_description: Computes the cloud probability of a Sentinel-2 L2A raster using
    an ensemble of five cloud segmentation models.
  long_description: The workflow computes cloud probabilities for each model independently,
    and averages them to obtain a single probability map.
  sources:
    sentinel_raster: Sentinel-2 L2A raster.
  sinks:
    cloud_probability: Cloud probability map.


```

```{mermaid}
    graph TD
    inp1>sentinel_raster]
    out1>cloud_probability]
    tsk1{{cloud1}}
    tsk2{{cloud2}}
    tsk3{{cloud3}}
    tsk4{{cloud4}}
    tsk5{{cloud5}}
    tsk6{{ensemble}}
    tsk1{{cloud1}} -- cloud_probability/cloud1 --> tsk6{{ensemble}}
    tsk2{{cloud2}} -- cloud_probability/cloud2 --> tsk6{{ensemble}}
    tsk3{{cloud3}} -- cloud_probability/cloud3 --> tsk6{{ensemble}}
    tsk4{{cloud4}} -- cloud_probability/cloud4 --> tsk6{{ensemble}}
    tsk5{{cloud5}} -- cloud_probability/cloud5 --> tsk6{{ensemble}}
    inp1>sentinel_raster] -- sentinel_raster --> tsk1{{cloud1}}
    inp1>sentinel_raster] -- sentinel_raster --> tsk2{{cloud2}}
    inp1>sentinel_raster] -- sentinel_raster --> tsk3{{cloud3}}
    inp1>sentinel_raster] -- sentinel_raster --> tsk4{{cloud4}}
    inp1>sentinel_raster] -- sentinel_raster --> tsk5{{cloud5}}
    tsk6{{ensemble}} -- cloud_probability --> out1>cloud_probability]
```