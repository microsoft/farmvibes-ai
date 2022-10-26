# FarmVibes.AI Available Workflows

This document lists all the workflows available in FarmVibes.AI, grouping them in the following categories:

- **Data Ingestion**: workflows that download and preprocess data from a particular source, preparing data to be the starting point for most of the other workflows in the platform.
This includes raw data sources (e.g., Sentinel 1 and 2, LandSat, CropDataLayer) as well as the SpaceEye cloud-removal model;
- **Data Processing**: workflows that transform data into different data types (e.g., computing NDVI/MSAVI/Methane indexes, aggregating mean/max/min statistics of rasters, timeseries aggregation);
- **FarmAI**:  composed workflows (data ingestion + processing) whose outputs enable FarmAI scenarios (e.g., predicting conservation practices, estimating soil carbon sequestration, identifying methane leakage);
- **ML**: machine learning-related workflows to train, evaluate, and infer models within the FarmVibes.AI platform (e.g., dataset creation, inference);

---------

## data_ingestion

- `airbus/airbus_download.yaml`: Downloads available AirBus imagery for the input geometry and time range.

- `airbus/airbus_price.yaml`: Prices available AirBus imagery for the input geometry and time range.

- `cdl/download_cdl.yaml`: Downloads crop classes maps in the continental USA for the input time range.

- `dem/download_dem.yaml`: Downloads digital elevation map tiles that intersect with the input geometry and time range.

- `landsat/preprocess_landsat.yaml`: Downloads and preprocesses LANDSAT tiles that intersect with the input geometry and time range.

- `naip/download_naip.yaml`: Downloads NAIP tiles that intersect with the input geometry and time range.

- `osm_road_geometries.yaml`: Downloads road geometry for input region from Open Street Maps.

- `sentinel1/preprocess_s1.yaml`: Downloads and preprocesses tiles of Sentinel-1 imagery that intersect with the input Sentinel-2 products in the input time range.

- `sentinel2/cloud_ensemble.yaml`: Computes the cloud probability of a Sentinel-2 L2A raster using an ensemble of five cloud segmentation models.

- `sentinel2/improve_cloud_mask.yaml`: Improves cloud masks by merging the product cloud mask with cloud and shadow masks computed by machine learning segmentation models.

- `sentinel2/improve_cloud_mask_ensemble.yaml`: Improves cloud masks by merging the product cloud mask with cloud and shadow masks computed by an ensemble of machine learning segmentation models.

- `sentinel2/preprocess_s2.yaml`: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range.

- `sentinel2/preprocess_s2_ensemble_masks.yaml`: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using an ensemble of cloud and shadow segmentation models.

- `sentinel2/preprocess_s2_improved_masks.yaml`: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using cloud and shadow segmentation models.

- `soil/soilgrids.yaml`: Downloads digital soil mapping information from SoilGrids for the input geometry.

- `soil/usda.yaml`: Downloads USDA soil classification raster.

- `spaceeye/spaceeye.yaml`: Runs the SpaceEye cloud removal pipeline, yielding daily cloud-free images for the input geometry and time range.

- `spaceeye/spaceeye_interpolation.yaml`: Runs the SpaceEye cloud removal pipeline using an interpolation-based algorithm, yielding daily cloud-free images for the input geometry and time range.

- `spaceeye/spaceeye_preprocess.yaml`: Runs the SpaceEye preprocessing pipeline.

- `spaceeye/spaceeye_preprocess_ensemble.yaml`: Runs the SpaceEye preprocessing pipeline with an ensemble of cloud segmentation models.

- `user_data/ingest_geometry.yaml`: Adds user geometries into the cluster storage, allowing for them to be used on workflows.

- `user_data/ingest_raster.yaml`: Adds user rasters into the cluster storage, allowing for them to be used on workflows.

- `weather/download_chirps.yaml`: Downloads accumulated precipitation data from the CHIRPS dataset.

- `weather/download_era5.yaml`: Hourly estimated weather variables.

- `weather/get_ambient_weather.yaml`: Downloads weather data from an Ambient Weather station.

- `weather/get_forecast.yaml`: Downloads weather forecast data from NOAA Global Forecast System (GFS) for the input time range.


## data_processing

- `clip/clip.yaml`: Performs a soft clip on an input raster based on a provided reference geometry.

- `gradient/raster_gradient.yaml`: Computes the gradient of each band of the input raster with a Sobel operator.

- `index/index.yaml`: Computes an index (ndvi, evi, msavi, or methane) from an input raster.

- `linear_trend/chunked_linear_trend.yaml`: Computes the pixel-wise linear trend of a list of rasters (e.g. NDVI).

- `merge/match_merge_to_ref.yaml`: Resamples input rasters to the reference rasters' grid.

- `outlier/detect_outlier.yaml`: Fits a single-component Gaussian Mixture Model (GMM) over input data to detect outliers according to the threshold parameter.

- `threshold/threshold_raster.yaml`: Thresholds values of an input raster if higher than the threshold parameter.

- `timeseries/timeseries_aggregation.yaml`: Computes the mean, standard deviation, maximum, and minimum values of all regions of the raster and aggregates them into a timeseries.

- `timeseries/timeseries_masked_aggregation.yaml`: Computes the mean, standard deviation, maximum, and minimum values of all regions of the raster considered by the mask and aggregates them into a timeseries.


## farm_ai

- `agriculture/canopy_cover.yaml`: Estimates pixel-wise canopy cover for a region and date.

- `agriculture/change_detection.yaml`: Identifies changes/outliers over NDVI across dates.

- `agriculture/emergence_summary.yaml`: Calculates emergence statistics using thresholded MSAVI (mean, standard deviation, maximum and minimum) for the input geometry and time range.

- `agriculture/methane_index.yaml`: Computes methane index from ultra emitters for a region and date range.

- `agriculture/ndvi_summary.yaml`: Calculates NDVI statistics (mean, standard deviation, maximum and minimum) for the input geometry and time range.

- `agriculture/weed_detection.yaml`: Generates shape files for similarly colored regions in the input raster.

- `carbon_local/carbon_whatif.yaml`: Computes the offset amount of carbon that would be sequestered in a field by a what-if scenario using COMET.

- `land_cover_mapping/conservation_practices.yaml`: Identifies conservation practices (terraces and grassed waterways) using elevation data.

- `land_degradation/landsat_ndvi_trend.yaml`: Estimates a linear trend over NDVI computer over LANDSAT tiles that intersect with the input geometry and time range.

- `land_degradation/ndvi_linear_trend.yaml`: Computes the pixel-wise NDVI linear trend over the input raster.


## ml

- `crop_segmentation.yaml`: Runs a crop segmentation model based on NDVI from SpaceEye imagery along the year.

- `dataset_generation/datagen_crop_segmentation.yaml`: Generates a dataset for crop segmentation, based on NDVI raster and Crop Data Layer (CDL) maps.


