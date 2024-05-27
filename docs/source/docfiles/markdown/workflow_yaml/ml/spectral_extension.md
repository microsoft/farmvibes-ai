# ml/spectral_extension

Generates high-resolution Sentinel-2 bands by combining UAV and Sentinel-2 data. The workflow will download a user-specified UAV raster, download and resample the corresponding Sentinel-2 raster, and run the spectral extension model to generate 8 Sentinel-2 bands at 0.125m resolution. The input raster should contain three bands (RGB) at 0.125m/px resolution in the range 0-255.

```{mermaid}
    graph TD
    inp1>raster]
    out1>s2_rasters]
    out2>matched_raster]
    out3>extended_raster]
    tsk1{{ingest_raster}}
    tsk2{{s2}}
    tsk3{{select}}
    tsk4{{match}}
    tsk5{{sequence}}
    tsk6{{compute_onnx}}
    tsk1{{ingest_raster}} -- downloaded/user_input --> tsk2{{s2}}
    tsk1{{ingest_raster}} -- downloaded/ref_raster --> tsk4{{match}}
    tsk1{{ingest_raster}} -- downloaded/rasters1 --> tsk5{{sequence}}
    tsk2{{s2}} -- raster/rasters --> tsk3{{select}}
    tsk3{{select}} -- sequence/raster --> tsk4{{match}}
    tsk4{{match}} -- output_raster/rasters2 --> tsk5{{sequence}}
    tsk5{{sequence}} -- sequence/input_raster --> tsk6{{compute_onnx}}
    inp1>raster] -- input_ref --> tsk1{{ingest_raster}}
    tsk2{{s2}} -- raster --> out1>s2_rasters]
    tsk4{{match}} -- output_raster --> out2>matched_raster]
    tsk6{{compute_onnx}} -- output_raster --> out3>extended_raster]
```

## Sources

- **raster**: The UAV input raster with three bands (red, green, blue, in this order) at 0.125m resolution.

## Sinks

- **s2_rasters**: The original Sentinel-2 raster used in the spectral extension.

- **matched_raster**: Sentinel-2 data resampled to the UAV raster's grid (low-resolution).

- **extended_raster**: The generated raster, containing 8 of the 12 Sentinel-2 bands.

## Parameters

- **resampling**: Resampling to use when reprojecting the Sentinel-2 data into the UAV raster's grid.

## Tasks

- **ingest_raster**: Downloads the raster from the input reference's url.

- **s2**: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range.

- **select**: Selects "num" entries from a Raster list so that the output sequence has a fixed length.

- **match**: Resamples the input `raster` to match the grid of `ref_raster`.

- **sequence**: Create a raster sequence from two lists of rasters.

- **compute_onnx**: Processes a sequence of rasters with an ONNX model.

## Workflow Yaml

```yaml

name: spectral_extension
sources:
  raster:
  - ingest_raster.input_ref
sinks:
  s2_rasters: s2.raster
  matched_raster: match.output_raster
  extended_raster: compute_onnx.output_raster
parameters:
  resampling: nearest
tasks:
  ingest_raster:
    op: download_raster_from_ref
    op_dir: download_from_ref
  s2:
    workflow: data_ingestion/sentinel2/preprocess_s2
  select:
    op: select_sequence_from_list
    op_dir: select_sequence
    parameters:
      num: 1
      criterion: first
  match:
    op: match_raster_to_ref
    parameters:
      resampling: '@from(resampling)'
  sequence:
    op: create_raster_sequence
  compute_onnx:
    op: compute_onnx_from_sequence
    op_dir: compute_onnx
    parameters:
      model_file: /opt/terravibes/ops/resources/spectral_extension_model/spectral_extension.onnx
      nodata: 0
edges:
- origin: ingest_raster.downloaded
  destination:
  - s2.user_input
  - match.ref_raster
  - sequence.rasters1
- origin: s2.raster
  destination:
  - select.rasters
- origin: select.sequence
  destination:
  - match.raster
- origin: match.output_raster
  destination:
  - sequence.rasters2
- origin: sequence.sequence
  destination:
  - compute_onnx.input_raster
description:
  short_description: Generates high-resolution Sentinel-2 bands by combining UAV and
    Sentinel-2 data.
  long_description: The workflow will download a user-specified UAV raster, download
    and resample the corresponding Sentinel-2 raster, and run the spectral extension
    model to generate 8 Sentinel-2 bands at 0.125m resolution. The input raster should
    contain three bands (RGB) at 0.125m/px resolution in the range 0-255.
  sources:
    raster: The UAV input raster with three bands (red, green, blue, in this order)
      at 0.125m resolution.
  sinks:
    s2_rasters: The original Sentinel-2 raster used in the spectral extension.
    matched_raster: Sentinel-2 data resampled to the UAV raster's grid (low-resolution).
    extended_raster: The generated raster, containing 8 of the 12 Sentinel-2 bands.
  parameters:
    resampling: Resampling to use when reprojecting the Sentinel-2 data into the UAV
      raster's grid.


```