# farm_ai/agriculture/canopy_cover

```yaml

name: canopy_cover
sources:
  user_input:
  - ndvi_summary.user_input
  - canopy_summary_timeseries.input_geometry
sinks:
  ndvi: ndvi_summary.compute_ndvi.compute_index.index
  estimated_canopy_cover: canopy.estimated_canopy_cover
  ndvi_timeseries: ndvi_summary.timeseries
  canopy_timeseries: canopy_summary_timeseries.timeseries
parameters:
  pc_key: null
tasks:
  ndvi_summary:
    workflow: farm_ai/agriculture/ndvi_summary
    parameters:
      pc_key: '@from(pc_key)'
  canopy:
    op: estimate_canopy_cover
  canopy_summary_timeseries:
    workflow: data_processing/timeseries/timeseries_masked_aggregation
edges:
- origin: ndvi_summary.compute_ndvi.compute_index.index
  destination:
  - canopy.indices
- origin: canopy.estimated_canopy_cover
  destination:
  - canopy_summary_timeseries.raster
- origin: ndvi_summary.s2.cloud.merge.merged_cloud_mask
  destination:
  - canopy_summary_timeseries.mask
description:
  short_description: Estimates pixel-wise canopy cover for a region and date.
  long_description: The workflow retrieves the relevant Sentinel-2 products with Planetary
    Computer (PC) API, and computes the NDVI for each available tile and date. It
    applies a linear regressor trained with polynomial features (up to the 3rd degree)
    on top of the index raster to estimate canopy cover. The coeficients and intercept
    of the regressor were obtained beforehand using as ground-truth masked/annotated
    drone imagery, and are used for inference in this workflow.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    ndvi: NDVI raster.
    estimated_canopy_cover: Raster with pixel-wise canopy cover estimation;
    ndvi_timeseries: Aggregated NDVI statistics of the retrieved tiles within the
      input geometry and time range.
    canopy_timeseries: Aggregated canopy cover statistics.
  parameters:
    pc_key: Optional Planetary Computer API key.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>ndvi]
    out2>estimated_canopy_cover]
    out3>ndvi_timeseries]
    out4>canopy_timeseries]
    tsk1{{ndvi_summary}}
    tsk2{{canopy}}
    tsk3{{canopy_summary_timeseries}}
    tsk1{{ndvi_summary}} -- index/indices --> tsk2{{canopy}}
    tsk2{{canopy}} -- estimated_canopy_cover/raster --> tsk3{{canopy_summary_timeseries}}
    tsk1{{ndvi_summary}} -- merged_cloud_mask/mask --> tsk3{{canopy_summary_timeseries}}
    inp1>user_input] -- user_input --> tsk1{{ndvi_summary}}
    inp1>user_input] -- input_geometry --> tsk3{{canopy_summary_timeseries}}
    tsk1{{ndvi_summary}} -- index --> out1>ndvi]
    tsk2{{canopy}} -- estimated_canopy_cover --> out2>estimated_canopy_cover]
    tsk1{{ndvi_summary}} -- timeseries --> out3>ndvi_timeseries]
    tsk3{{canopy_summary_timeseries}} -- timeseries --> out4>canopy_timeseries]
```