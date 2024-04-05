# ml/driveway_detection

Detects driveways in front of houses. The workflow downloads road geometry from Open Street Maps and segments the front of houses in the input image using a machine learning model. It then uses the input image, segmentation map, road geometry, and input property boundaries to detect the presence of driveways in the front of each house.

```{mermaid}
    graph TD
    inp1>input_raster]
    inp2>property_boundaries]
    out1>properties]
    out2>driveways]
    tsk1{{segment}}
    tsk2{{osm}}
    tsk3{{detect}}
    tsk1{{segment}} -- segmentation_raster --> tsk3{{detect}}
    tsk2{{osm}} -- roads --> tsk3{{detect}}
    inp1>input_raster] -- input_raster --> tsk1{{segment}}
    inp1>input_raster] -- input_raster --> tsk3{{detect}}
    inp1>input_raster] -- user_input --> tsk2{{osm}}
    inp2>property_boundaries] -- property_boundaries --> tsk3{{detect}}
    tsk3{{detect}} -- properties_with_driveways --> out1>properties]
    tsk3{{detect}} -- driveways --> out2>driveways]
```

## Sources

- **input_raster**: Aerial imagery of the region of interest with RBG + NIR bands.

- **property_boundaries**: Property boundary information for the region of interest.

## Sinks

- **properties**: Boundaries of properties that contain a driveway.

- **driveways**: Regions of each property boundary where a driveway was detected.

## Parameters

- **min_region_area**: Minimum contiguous region that will be considered as a potential driveway, in meters.

- **ndvi_thr**: Only areas under this NDVI threshold will be considered for driveways.

- **car_size**: Expected size of a car, in pixels, defined as [height, width].

- **num_kernels**: Number of rotated kernels to try to fit a car inside a potential driveway region.

- **car_thr**: Ratio of pixels of a kernel that have to be inside a region in order to consider it a parkable spot.

## Tasks

- **segment**: Segments the front of houses in the input raster using a machine learning model.

- **osm**: Downloads road geometry for input region from Open Street Maps.

- **detect**: Detects driveways in the front of each house, using the input image, segmentation map, road geometry, and input property boundaries.

## Workflow Yaml

```yaml

name: driveway_detection
sources:
  input_raster:
  - segment.input_raster
  - detect.input_raster
  - osm.user_input
  property_boundaries:
  - detect.property_boundaries
sinks:
  properties: detect.properties_with_driveways
  driveways: detect.driveways
parameters:
  min_region_area: null
  ndvi_thr: null
  car_size: null
  num_kernels: null
  car_thr: null
tasks:
  segment:
    op: segment_driveway
  osm:
    workflow: data_ingestion/osm_road_geometries
    parameters:
      network_type: drive_service
      buffer_size: 100
  detect:
    op: detect_driveway
    parameters:
      min_region_area: '@from(min_region_area)'
      ndvi_thr: '@from(ndvi_thr)'
      car_size: '@from(car_size)'
      num_kernels: '@from(num_kernels)'
      car_thr: '@from(car_thr)'
edges:
- origin: segment.segmentation_raster
  destination:
  - detect.segmentation_raster
- origin: osm.roads
  destination:
  - detect.roads
description:
  short_description: Detects driveways in front of houses.
  long_description: The workflow downloads road geometry from Open Street Maps and
    segments the front of houses in the input image using a machine learning model.
    It then uses the input image, segmentation map, road geometry, and input property
    boundaries to detect the presence of driveways in the front of each house.
  sources:
    input_raster: Aerial imagery of the region of interest with RBG + NIR bands.
    property_boundaries: Property boundary information for the region of interest.
  sinks:
    properties: Boundaries of properties that contain a driveway.
    driveways: Regions of each property boundary where a driveway was detected.
  parameters:
    min_region_area: Minimum contiguous region that will be considered as a potential
      driveway, in meters.
    ndvi_thr: Only areas under this NDVI threshold will be considered for driveways.
    car_size: Expected size of a car, in pixels, defined as [height, width].
    num_kernels: Number of rotated kernels to try to fit a car inside a potential
      driveway region.
    car_thr: Ratio of pixels of a kernel that have to be inside a region in order
      to consider it a parkable spot.


```