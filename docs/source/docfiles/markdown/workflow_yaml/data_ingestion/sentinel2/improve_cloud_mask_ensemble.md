# data_ingestion/sentinel2/improve_cloud_mask_ensemble

Improves cloud masks by merging the product cloud mask with cloud and shadow masks computed by an ensemble of machine learning segmentation models. This workflow computes cloud and shadow probabilities using and ensemble of segmentation models, thresholds them, and merges the models' masks with the product mask.

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

## Sources

- **s2_raster**: Sentinel-2 L2A raster.

- **product_mask**: Cloud mask obtained from the product's quality indicators.

## Sinks

- **mask**: Improved cloud mask.

## Parameters

- **cloud_thr**: Confidence threshold to assign a pixel as cloud.

- **shadow_thr**: Confidence threshold to assign a pixel as shadow.

## Tasks

- **cloud**: Computes the cloud probability of a Sentinel-2 L2A raster using an ensemble of five cloud segmentation models.

- **shadow**: Computes shadow probabilities using a convolutional segmentation model for L2A.

- **merge**: Merges cloud, shadow and product cloud masks into a single mask.

## Workflow Yaml

```yaml

name: improve_cloud_mask_ensemble
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
tasks:
  cloud:
    workflow: data_ingestion/sentinel2/cloud_ensemble
  shadow:
    op: compute_shadow_prob
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
    and shadow masks computed by an ensemble of machine learning segmentation models.
  long_description: This workflow computes cloud and shadow probabilities using and
    ensemble of segmentation models, thresholds them, and merges the models' masks
    with the product mask.
  sources:
    s2_raster: Sentinel-2 L2A raster.
    product_mask: Cloud mask obtained from the product's quality indicators.
  sinks:
    mask: Improved cloud mask.
  parameters:
    cloud_thr: Confidence threshold to assign a pixel as cloud.
    shadow_thr: Confidence threshold to assign a pixel as shadow.


```