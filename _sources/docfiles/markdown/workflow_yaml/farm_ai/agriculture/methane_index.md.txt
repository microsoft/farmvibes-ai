# farm_ai/agriculture/methane_index

Computes methane index from ultra emitters for a region and date range. The workflow retrieves the relevant Sentinel-2 products with Planetary Computer (PC) API and crop the rasters for the region defined in user_input. All bands are normalized and an anti-aliasing guassian filter is applied to smooth and remove potential artifacts. An unsupervised K-Nearest Neighbor is applied to identify bands similar to band 12, and the index is computed by the difference between band 12 to the pixel-wise median of top K similar bands.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>index]
    out2>s2_raster]
    out3>cloud_mask]
    tsk1{{s2}}
    tsk2{{clip}}
    tsk3{{methane}}
    tsk1{{s2}} -- raster --> tsk2{{clip}}
    tsk2{{clip}} -- clipped_raster/raster --> tsk3{{methane}}
    inp1>user_input] -- user_input --> tsk1{{s2}}
    inp1>user_input] -- input_geometry --> tsk2{{clip}}
    tsk3{{methane}} -- index_raster --> out1>index]
    tsk1{{s2}} -- raster --> out2>s2_raster]
    tsk1{{s2}} -- mask --> out3>cloud_mask]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **index**: Methane index raster.

- **s2_raster**: Sentinel-2 raster.

- **cloud_mask**: Cloud mask.

## Parameters

- **pc_key**: Optional Planetary Computer API key.

## Tasks

- **s2**: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using cloud and shadow segmentation models.

- **clip**: Performs a soft clip on an input raster based on a provided reference geometry.

- **methane**: Computes an index from the bands of an input raster.

## Workflow Yaml

```yaml

name: methane_index
sources:
  user_input:
  - s2.user_input
  - clip.input_geometry
sinks:
  index: methane.index_raster
  s2_raster: s2.raster
  cloud_mask: s2.mask
parameters:
  pc_key: null
tasks:
  s2:
    workflow: data_ingestion/sentinel2/preprocess_s2_improved_masks
    parameters:
      pc_key: '@from(pc_key)'
  clip:
    workflow: data_processing/clip/clip
  methane:
    workflow: data_processing/index/index
    parameters:
      index: methane
edges:
- origin: s2.raster
  destination:
  - clip.raster
- origin: clip.clipped_raster
  destination:
  - methane.raster
description:
  short_description: Computes methane index from ultra emitters for a region and date
    range.
  long_description: The workflow retrieves the relevant Sentinel-2 products with Planetary
    Computer (PC) API and crop the rasters for the region defined in user_input. All
    bands are normalized and an anti-aliasing guassian filter is applied to smooth
    and remove potential artifacts. An unsupervised K-Nearest Neighbor is applied
    to identify bands similar to band 12, and the index is computed by the difference
    between band 12 to the pixel-wise median of top K similar bands.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    index: Methane index raster.
    s2_raster: Sentinel-2 raster.
    cloud_mask: Cloud mask.
  parameters:
    pc_key: Optional Planetary Computer API key.


```