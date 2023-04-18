# data_ingestion/sentinel2/improve_cloud_mask

```yaml

name: improve_cloud_mask
sources:
  s2_raster:
  - cloud.sentinel_raster
  - shadow.sentinel_raster
  product_mask:
  - merge.product_mask
sinks:
  mask: merge.merged_cloud_mask
parameters:
  cloud_thr: null
  shadow_thr: null
  in_memory: null
  cloud_model: null
  shadow_model: null
tasks:
  cloud:
    op: compute_cloud_prob
    parameters:
      in_memory: '@from(in_memory)'
      model_path: '@from(cloud_model)'
  shadow:
    op: compute_shadow_prob
    parameters:
      in_memory: '@from(in_memory)'
      model_path: '@from(shadow_model)'
  merge:
    op: merge_cloud_masks_simple
    op_dir: merge_cloud_masks
    parameters:
      cloud_prob_threshold: '@from(cloud_thr)'
      shadow_prob_threshold: '@from(shadow_thr)'
edges:
- origin: cloud.cloud_probability
  destination:
  - merge.cloud_probability
- origin: shadow.shadow_probability
  destination:
  - merge.shadow_probability
description:
  short_description: Improves cloud masks by merging the product cloud mask with cloud
    and shadow masks computed by machine learning segmentation models.
  long_description: This workflow computes cloud and shadow probabilities using segmentation
    models, thresholds them, and merges the models' masks with the product mask.
  sources:
    s2_raster: Sentinel-2 L2A raster.
    product_mask: Cloud mask obtained from the product's quality indicators.
  sinks:
    mask: Improved cloud mask.
  parameters:
    cloud_thr: Confidence threshold to assign a pixel as cloud.
    shadow_thr: Confidence threshold to assign a pixel as shadow.
    in_memory: Whether to load the whole raster in memory when running predictions.
      Uses more memory (~4GB/worker) but speeds up inference for fast models.
    cloud_model: "ONNX file for the cloud model. Available models are 'cloud_model{idx}_cpu.onnx'\
      \ with idx \u2208 {1, 2} being FPN-based models, which are more accurate but\
      \ slower, and idx \u2208 {3, 4, 5} being cheaplab models, which are less accurate\
      \ but faster."
    shadow_model: ONNX file for the shadow model. 'shadow.onnx' is the only currently
      available model.


```

```{mermaid}
    graph TD
    inp1>s2_raster]
    inp2>product_mask]
    out1>mask]
    tsk1{{cloud}}
    tsk2{{shadow}}
    tsk3{{merge}}
    tsk1{{cloud}} -- cloud_probability --> tsk3{{merge}}
    tsk2{{shadow}} -- shadow_probability --> tsk3{{merge}}
    inp1>s2_raster] -- sentinel_raster --> tsk1{{cloud}}
    inp1>s2_raster] -- sentinel_raster --> tsk2{{shadow}}
    inp2>product_mask] -- product_mask --> tsk3{{merge}}
    tsk3{{merge}} -- merged_cloud_mask --> out1>mask]
```