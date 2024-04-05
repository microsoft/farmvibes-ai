# farm_ai/agriculture/weed_detection

Generates shape files for similarly colored regions in the input raster. The workflow retrieves a remote raster and trains a Gaussian Mixture Model (GMM) over a subset of the input data with a fixed number of components. The GMM is then used to cluster all images pixels. Clustered regions are converted to polygons with a minimum size threshold. These polygons are then simplified to smooth their borders. All polygons of a given cluster are written to a single shapefile. All files are then compressed and returned as a single zip archive.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>result]
    tsk1{{download_raster}}
    tsk2{{weed_detection}}
    tsk1{{download_raster}} -- raster --> tsk2{{weed_detection}}
    inp1>user_input] -- user_input --> tsk1{{download_raster}}
    tsk2{{weed_detection}} -- result --> out1>result]
```

## Sources

- **user_input**: External references to raster data.

## Sinks

- **result**: Zip file containing cluster geometries.

## Parameters

- **buffer**: Buffer size, in projected CRS, to apply to the input geometry before sampling training points. A negative number can be used to avoid sampling unwanted regions if the geometry is not very precise.

- **no_data**: Value to use as nodata when reading the raster. Uses the raster's internal nodata value if not provided.

- **clusters**: Number of clusters to use when segmenting the image.

- **sieve_size**: Area of the minimum connected region. Smaller regions will have their class assigned to the largest adjancent region.

- **simplify**: Method used to simplify the geometries. Accepts 'none', for no simplification, 'simplify', for tolerance-based simplification, and 'convex', for returning the convex hull.

- **tolerance**: Tolerance for simplifcation algorithm. Only applicable if simplification method is 'simplify'.

- **samples**: Number os samples to use during training.

- **bands**: List of band indices to use during training and inference.

- **alpha_index**: Positive index of alpha band, if used to filter out nodata values.

## Tasks

- **download_raster**: Adds user rasters into the cluster storage, allowing for them to be used on workflows.

- **weed_detection**: Trains a Gaussian Mixture Model (GMM), cluster all images pixels, and convert clustered regions into polygons.

## Workflow Yaml

```yaml

name: weed_detection
sources:
  user_input:
  - download_raster.user_input
sinks:
  result: weed_detection.result
parameters:
  buffer: null
  no_data: null
  clusters: null
  sieve_size: null
  simplify: null
  tolerance: null
  samples: null
  bands: null
  alpha_index: null
tasks:
  download_raster:
    workflow: data_ingestion/user_data/ingest_raster
  weed_detection:
    op: weed_detection
    parameters:
      buffer: '@from(buffer)'
      no_data: '@from(no_data)'
      clusters: '@from(clusters)'
      sieve_size: '@from(sieve_size)'
      simplify: '@from(simplify)'
      tolerance: '@from(tolerance)'
      samples: '@from(samples)'
      bands: '@from(bands)'
      alpha_index: '@from(alpha_index)'
edges:
- origin: download_raster.raster
  destination:
  - weed_detection.raster
description:
  short_description: Generates shape files for similarly colored regions in the input
    raster.
  long_description: The workflow retrieves a remote raster and trains a Gaussian Mixture
    Model (GMM) over a subset of the input data with a fixed number of components.
    The GMM is then used to cluster all images pixels. Clustered regions are converted
    to polygons with a minimum size threshold. These polygons are then simplified
    to smooth their borders. All polygons of a given cluster are written to a single
    shapefile. All files are then compressed and returned as a single zip archive.
  sources:
    user_input: External references to raster data.
  sinks:
    result: Zip file containing cluster geometries.
  parameters:
    buffer: Buffer size, in projected CRS, to apply to the input geometry before sampling
      training points. A negative number can be used to avoid sampling unwanted regions
      if the geometry is not very precise.
    no_data: Value to use as nodata when reading the raster. Uses the raster's internal
      nodata value if not provided.
    clusters: Number of clusters to use when segmenting the image.
    sieve_size: Area of the minimum connected region. Smaller regions will have their
      class assigned to the largest adjancent region.
    simplify: Method used to simplify the geometries. Accepts 'none', for no simplification,
      'simplify', for tolerance-based simplification, and 'convex', for returning
      the convex hull.
    tolerance: Tolerance for simplifcation algorithm. Only applicable if simplification
      method is 'simplify'.
    samples: Number os samples to use during training.
    bands: List of band indices to use during training and inference.
    alpha_index: Positive index of alpha band, if used to filter out nodata values.


```