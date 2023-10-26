# data_processing/chunk_onnx/chunk_onnx_sequence

```yaml

name: chunk_onnx_sequence
sources:
  rasters:
  - chunk_raster.rasters
  - compute_onnx.input_raster
sinks:
  raster: combine_chunks.raster
parameters:
  model_file: null
  step: 100
tasks:
  chunk_raster:
    op: chunk_sequence_raster
    op_dir: chunk_raster
    parameters:
      step_y: '@from(step)'
      step_x: '@from(step)'
  compute_onnx:
    op: compute_onnx_from_chunks
    op_dir: compute_onnx
    parameters:
      model_file: '@from(model_file)'
      window_size: '@from(step)'
  combine_chunks:
    op: combine_chunks
edges:
- origin: chunk_raster.chunk_series
  destination:
  - compute_onnx.chunk
- origin: compute_onnx.output_raster
  destination:
  - combine_chunks.chunks
description:
  short_description: Run an Onnx model over all rasters in the input to produce a
    single raster.
  long_description: This workflow is intended to run an Onnx model on all input rasters
    to produce a single raster output. This can be used, for instance, to compute
    time-series analysis of a list of rasters that span multiple times. The analysis
    can be any computation that can be expressed as an Onnx model (for an example,
    see notebooks/crop_cycles/crop_cycles.ipynb). In order to run the model in parallel
    (and avoid running out of memory if the list of rasters is large), the input rasters
    are divided spatially into chunks (that span all times). The Onnx model is applied
    to these chunks and then combined back to produce the final output.
  sources:
    rasters: Input rasters.
  sinks:
    raster: Result of the Onnx model run.
  parameters:
    model_file: An Onnx model which needs to be deployed with "farmvibes-ai local
      add-onnx" command
    step: Size of the chunk in pixels


```

```{mermaid}
    graph TD
    inp1>rasters]
    out1>raster]
    tsk1{{chunk_raster}}
    tsk2{{compute_onnx}}
    tsk3{{combine_chunks}}
    tsk1{{chunk_raster}} -- chunk_series/chunk --> tsk2{{compute_onnx}}
    tsk2{{compute_onnx}} -- output_raster/chunks --> tsk3{{combine_chunks}}
    inp1>rasters] -- rasters --> tsk1{{chunk_raster}}
    inp1>rasters] -- input_raster --> tsk2{{compute_onnx}}
    tsk3{{combine_chunks}} -- raster --> out1>raster]
```