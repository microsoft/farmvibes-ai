# ml/crop_segmentation

```yaml

name: crop_segmentation
sources:
  user_input:
  - spaceeye.user_input
sinks:
  segmentation: inference.output_raster
parameters:
  pc_key: null
  model_file: null
  model_bands: 37
tasks:
  spaceeye:
    workflow: data_ingestion/spaceeye/spaceeye_interpolation
    parameters:
      pc_key: '@from(pc_key)'
  ndvi:
    workflow: data_processing/index/index
    parameters:
      index: ndvi
  group:
    op: select_sequence_from_list
    op_dir: select_sequence
    parameters:
      num: '@from(model_bands)'
      criterion: regular
  inference:
    op: compute_onnx_from_sequence
    op_dir: compute_onnx
    parameters:
      model_file: '@from(model_file)'
      window_size: 256
      overlap: 0.25
      num_workers: 4
edges:
- origin: spaceeye.raster
  destination:
  - ndvi.raster
- origin: ndvi.index_raster
  destination:
  - group.rasters
- origin: group.sequence
  destination:
  - inference.input_raster
description:
  short_description: Runs a crop segmentation model based on NDVI from SpaceEye imagery
    along the year.
  long_description: The workflow generates SpaceEye cloud-free data for the input
    region and time range and computes NDVI over those. NDVI values sampled regularly
    along the year are stacked as bands and used as input to the crop segmentation
    model.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    segmentation: Crop segmentation map at 10m resolution.
  parameters:
    pc_key: Optional Planetary Computer API key.
    model_file: Path to the ONNX file containing the model architecture and weights.
    model_bands: Number of NDVI bands to stack as the model input.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>segmentation]
    tsk1{{spaceeye}}
    tsk2{{ndvi}}
    tsk3{{group}}
    tsk4{{inference}}
    tsk1{{spaceeye}} -- raster --> tsk2{{ndvi}}
    tsk2{{ndvi}} -- index_raster/rasters --> tsk3{{group}}
    tsk3{{group}} -- sequence/input_raster --> tsk4{{inference}}
    inp1>user_input] -- user_input --> tsk1{{spaceeye}}
    tsk4{{inference}} -- output_raster --> out1>segmentation]
```