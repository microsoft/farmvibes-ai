# farm_ai/water/irrigation_classification

Develops 30m pixel-wise irrigation probability map. The workflow retrieves LANDSAT 8 Surface Reflectance (SR) image tile and land surface elevation DEM data, and  runs four ops to compute irrigation probability map. The land surface elevation data source are 10m USGS DEM, or 30m Copernicus DEM; but Copernicus DEM is set as the default source in the workflow. Landsat Op compute_cloud_water_mask utilizes the qa_pixel band of image and NDVI index to generate mask of cloud cover and water bodies. Op compute_evaporative_fraction utilizes NDVI index, land surface temperature (LST), green and near infra-red bands, and DEM data to estimate evaporative flux (ETRF). Op compute_ngi_egi_layers utilizes NDVI index, ETRF estimates, green and near infra-red bands to generate NGI and EGI irrigation layers. Lastly op compute_irrigation_probability uses NGI and EGI layers along with LST band; and applies optimized logistic regression model to compute 30m pixel-wise irrigation probability map. The coeficients and intercept of the model were obtained beforehand using as ground-truth data from Nebraska state, USA for the year 2015.

```{mermaid}
    graph TD
    inp1>user_input]
    out1>landsat_bands]
    out2>ndvi]
    out3>cloud_water_mask]
    out4>dem]
    out5>evaporative_fraction]
    out6>ngi]
    out7>egi]
    out8>lst]
    out9>irrigation_probability]
    tsk1{{landsat}}
    tsk2{{ndvi}}
    tsk3{{merge_geom}}
    tsk4{{merge_geom_time_range}}
    tsk5{{cloud_water_mask}}
    tsk6{{dem}}
    tsk7{{match_dem}}
    tsk8{{evaporative_fraction}}
    tsk9{{ngi_egi_layers}}
    tsk10{{irrigation_probability}}
    tsk1{{landsat}} -- raster/items --> tsk3{{merge_geom}}
    tsk1{{landsat}} -- raster --> tsk2{{ndvi}}
    tsk1{{landsat}} -- raster/landsat_raster --> tsk5{{cloud_water_mask}}
    tsk1{{landsat}} -- raster/ref_rasters --> tsk7{{match_dem}}
    tsk1{{landsat}} -- raster/landsat_raster --> tsk8{{evaporative_fraction}}
    tsk1{{landsat}} -- raster/landsat_raster --> tsk9{{ngi_egi_layers}}
    tsk1{{landsat}} -- raster/landsat_raster --> tsk10{{irrigation_probability}}
    tsk2{{ndvi}} -- index/ndvi_raster --> tsk5{{cloud_water_mask}}
    tsk2{{ndvi}} -- index/ndvi_raster --> tsk8{{evaporative_fraction}}
    tsk2{{ndvi}} -- index/ndvi_raster --> tsk9{{ngi_egi_layers}}
    tsk3{{merge_geom}} -- merged/geometry --> tsk4{{merge_geom_time_range}}
    tsk4{{merge_geom_time_range}} -- merged/user_input --> tsk6{{dem}}
    tsk6{{dem}} -- raster/rasters --> tsk7{{match_dem}}
    tsk7{{match_dem}} -- match_rasters/dem_raster --> tsk8{{evaporative_fraction}}
    tsk8{{evaporative_fraction}} -- evaporative_fraction --> tsk9{{ngi_egi_layers}}
    tsk5{{cloud_water_mask}} -- cloud_water_mask/cloud_water_mask_raster --> tsk8{{evaporative_fraction}}
    tsk5{{cloud_water_mask}} -- cloud_water_mask/cloud_water_mask_raster --> tsk9{{ngi_egi_layers}}
    tsk5{{cloud_water_mask}} -- cloud_water_mask/cloud_water_mask_raster --> tsk10{{irrigation_probability}}
    tsk9{{ngi_egi_layers}} -- ngi --> tsk10{{irrigation_probability}}
    tsk9{{ngi_egi_layers}} -- egi --> tsk10{{irrigation_probability}}
    tsk9{{ngi_egi_layers}} -- lst --> tsk10{{irrigation_probability}}
    inp1>user_input] -- user_input --> tsk1{{landsat}}
    inp1>user_input] -- time_range --> tsk4{{merge_geom_time_range}}
    tsk1{{landsat}} -- raster --> out1>landsat_bands]
    tsk2{{ndvi}} -- index --> out2>ndvi]
    tsk5{{cloud_water_mask}} -- cloud_water_mask --> out3>cloud_water_mask]
    tsk7{{match_dem}} -- match_rasters --> out4>dem]
    tsk8{{evaporative_fraction}} -- evaporative_fraction --> out5>evaporative_fraction]
    tsk9{{ngi_egi_layers}} -- ngi --> out6>ngi]
    tsk9{{ngi_egi_layers}} -- egi --> out7>egi]
    tsk9{{ngi_egi_layers}} -- lst --> out8>lst]
    tsk10{{irrigation_probability}} -- irrigation_probability --> out9>irrigation_probability]
```

## Sources

- **user_input**: Time range and geometry of interest.

## Sinks

- **landsat_bands**: Raster of Landsat bands.

- **ndvi**: NDVI raster.

- **cloud_water_mask**: Mask of cloud cover and water bodies.

- **dem**: DEM raster. Options are CopernicusDEM30 and USGS3DEP.

- **evaporative_fraction**: Raster with estimates of evaporative fraction flux.

- **ngi**: Raster of NGI irrigation layer.

- **egi**: Raster of EGI irrigation layer.

- **lst**: Raster of land surface temperature.

- **irrigation_probability**: Raster of irrigation probability map in 30m resolution.

## Parameters

- **ndvi_threshold**: NDVI index threshold value for masking water bodies.

- **ndvi_hot_threshold**: Maximum NDVI index threshold value for selecting hot pixel.

- **coef_ngi**: Coefficient of NGI layer in optimized logistic regression model.

- **coef_egi**: Coefficient of EGI layer in optimized logistic regression model.

- **coef_lst**: Coefficient of land surface temperature band in optimized logistic regression model.

- **intercept**: Intercept value of optimized logistic regression model.

- **dem_resolution**: Spatial resolution of the DEM. 10m and 30m are available.

- **dem_provider**: Provider of the DEM. "USGS3DEP" and "CopernicusDEM30" are available.

- **pc_key**: Optional Planetary Computer API key.

## Tasks

- **landsat**: Downloads and preprocesses LANDSAT tiles that intersect with the input geometry and time range.

- **ndvi**: Computes `index` over the input raster.

- **merge_geom**: Create item with merged geometry from item list.

- **merge_geom_time_range**: Create item that contains the geometry from one item and the time range from another.

- **cloud_water_mask**: Merges landsat cloud mask and NDVI-based mask to produce a cloud water mask.

- **dem**: Downloads digital elevation map tiles that intersect with the input geometry and time range.

- **match_dem**: Resamples input rasters to the reference rasters' grid.

- **evaporative_fraction**: Computes evaporative fraction layer based on the percentile values of lst_dem (created by treating land surface temperature with dem) and ndvi layers. The source of constants used is "Senay, G.B.; Bohms, S.; Singh, R.K.; Gowda, P.H.; Velpuri, N.M.; Alemu, H.; Verdin, J.P. Operational Evapotranspiration Mapping Using Remote Sensing and Weather Datasets - A New Parameterization for the SSEB Approach. JAWRA J. Am. Water Resour. Assoc. 2013, 49, 577–591. The land surface elevation data source are 10m USGS DEM, and 30m Copernicus DEM; but Copernicus DEM is set as default source in the workflow.

- **ngi_egi_layers**: Computes NGI, EGI, and LST layers from landsat bands, ndvi layer, cloud water mask layer and evaporative fraction layer

- **irrigation_probability**: Computes irrigation probability values for each pixel in raster using optimized logistic regression model with ngi, egi, and lst rasters as input

## Workflow Yaml

```yaml

name: irrigation_classification
sources:
  user_input:
  - landsat.user_input
  - merge_geom_time_range.time_range
sinks:
  landsat_bands: landsat.raster
  ndvi: ndvi.index
  cloud_water_mask: cloud_water_mask.cloud_water_mask
  dem: match_dem.match_rasters
  evaporative_fraction: evaporative_fraction.evaporative_fraction
  ngi: ngi_egi_layers.ngi
  egi: ngi_egi_layers.egi
  lst: ngi_egi_layers.lst
  irrigation_probability: irrigation_probability.irrigation_probability
parameters:
  ndvi_threshold: 0.0
  ndvi_hot_threshold: 0.02
  coef_ngi: -0.50604148
  coef_egi: -0.93103156
  coef_lst: -0.14612046
  intercept: 1.99036986
  dem_resolution: 30
  dem_provider: CopernicusDEM30
  pc_key: null
tasks:
  landsat:
    workflow: data_ingestion/landsat/preprocess_landsat
    parameters:
      pc_key: '@from(pc_key)'
  ndvi:
    op: compute_index
  merge_geom:
    op: merge_geometries
  merge_geom_time_range:
    op: merge_geometry_and_time_range
  cloud_water_mask:
    op: compute_cloud_water_mask
    parameters:
      ndvi_threshold: '@from(ndvi_threshold)'
  dem:
    workflow: data_ingestion/dem/download_dem
    parameters:
      resolution: '@from(dem_resolution)'
      provider: '@from(dem_provider)'
  match_dem:
    workflow: data_processing/merge/match_merge_to_ref
  evaporative_fraction:
    op: compute_evaporative_fraction
    parameters:
      ndvi_hot_threshold: '@from(ndvi_hot_threshold)'
  ngi_egi_layers:
    op: compute_ngi_egi_layers
  irrigation_probability:
    op: compute_irrigation_probability
    parameters:
      coef_ngi: '@from(coef_ngi)'
      coef_egi: '@from(coef_egi)'
      coef_lst: '@from(coef_lst)'
      intercept: '@from(intercept)'
edges:
- origin: landsat.raster
  destination:
  - merge_geom.items
  - ndvi.raster
  - cloud_water_mask.landsat_raster
  - match_dem.ref_rasters
  - evaporative_fraction.landsat_raster
  - ngi_egi_layers.landsat_raster
  - irrigation_probability.landsat_raster
- origin: ndvi.index
  destination:
  - cloud_water_mask.ndvi_raster
  - evaporative_fraction.ndvi_raster
  - ngi_egi_layers.ndvi_raster
- origin: merge_geom.merged
  destination:
  - merge_geom_time_range.geometry
- origin: merge_geom_time_range.merged
  destination:
  - dem.user_input
- origin: dem.raster
  destination:
  - match_dem.rasters
- origin: match_dem.match_rasters
  destination:
  - evaporative_fraction.dem_raster
- origin: evaporative_fraction.evaporative_fraction
  destination:
  - ngi_egi_layers.evaporative_fraction
- origin: cloud_water_mask.cloud_water_mask
  destination:
  - evaporative_fraction.cloud_water_mask_raster
  - ngi_egi_layers.cloud_water_mask_raster
  - irrigation_probability.cloud_water_mask_raster
- origin: ngi_egi_layers.ngi
  destination:
  - irrigation_probability.ngi
- origin: ngi_egi_layers.egi
  destination:
  - irrigation_probability.egi
- origin: ngi_egi_layers.lst
  destination:
  - irrigation_probability.lst
description:
  short_description: Develops 30m pixel-wise irrigation probability map.
  long_description: The workflow retrieves LANDSAT 8 Surface Reflectance (SR) image
    tile and land surface elevation DEM data, and  runs four ops to compute irrigation
    probability map. The land surface elevation data source are 10m USGS DEM, or 30m
    Copernicus DEM; but Copernicus DEM is set as the default source in the workflow.
    Landsat Op compute_cloud_water_mask utilizes the qa_pixel band of image and NDVI
    index to generate mask of cloud cover and water bodies. Op compute_evaporative_fraction
    utilizes NDVI index, land surface temperature (LST), green and near infra-red
    bands, and DEM data to estimate evaporative flux (ETRF). Op compute_ngi_egi_layers
    utilizes NDVI index, ETRF estimates, green and near infra-red bands to generate
    NGI and EGI irrigation layers. Lastly op compute_irrigation_probability uses NGI
    and EGI layers along with LST band; and applies optimized logistic regression
    model to compute 30m pixel-wise irrigation probability map. The coeficients and
    intercept of the model were obtained beforehand using as ground-truth data from
    Nebraska state, USA for the year 2015.
  sources:
    user_input: Time range and geometry of interest.
  sinks:
    landsat_bands: Raster of Landsat bands.
    ndvi: NDVI raster.
    cloud_water_mask: Mask of cloud cover and water bodies.
    dem: DEM raster. Options are CopernicusDEM30 and USGS3DEP.
    evaporative_fraction: Raster with estimates of evaporative fraction flux.
    ngi: Raster of NGI irrigation layer.
    egi: Raster of EGI irrigation layer.
    lst: Raster of land surface temperature.
    irrigation_probability: Raster of irrigation probability map in 30m resolution.
  parameters:
    ndvi_threshold: NDVI index threshold value for masking water bodies.
    ndvi_hot_threshold: Maximum NDVI index threshold value for selecting hot pixel.
    coef_ngi: Coefficient of NGI layer in optimized logistic regression model.
    coef_egi: Coefficient of EGI layer in optimized logistic regression model.
    coef_lst: Coefficient of land surface temperature band in optimized logistic regression
      model.
    intercept: Intercept value of optimized logistic regression model.
    pc_key: Optional Planetary Computer API key.


```