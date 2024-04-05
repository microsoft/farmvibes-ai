# Workflow List

We group FarmVibes.AI workflows in the following categories:

- **Data Ingestion**: workflows that download and preprocess data from a particular source, preparing data to be the starting point for most of the other workflows in the platform.
This includes raw data sources (e.g., Sentinel 1 and 2, LandSat, CropDataLayer) as well as the SpaceEye cloud-removal model;
- **Data Processing**: workflows that transform data into different data types (e.g., computing NDVI/MSAVI/Methane indexes, aggregating mean/max/min statistics of rasters, timeseries aggregation);
- **FarmAI**:  composed workflows (data ingestion + processing) whose outputs enable FarmAI scenarios (e.g., predicting conservation practices, estimating soil carbon sequestration, identifying methane leakage);
- **ForestAI**: composed workflows (data ingestion + processing) whose outputs enable ForestAI scenarios (e.g., detecting forest change, estimating forest extent);
- **ML**: machine learning-related workflows to train, evaluate, and infer models within the FarmVibes.AI platform (e.g., dataset creation, inference);

Below is a list of all available workflows within the FarmVibes.AI platform. For each of them, we provide a brief description and a link to the corresponding documentation page.

---------

## data_ingestion

- [`admag/admag_seasonal_field` 📄](workflow_yaml/data_ingestion/admag/admag_seasonal_field.md): Generates SeasonalFieldInformation using ADMAg (Microsoft Azure Data Manager for Agriculture).

- [`admag/prescriptions` 📄](workflow_yaml/data_ingestion/admag/prescriptions.md): Fetches prescriptions using ADMAg (Microsoft Azure Data Manager for Agriculture).

- [`airbus/airbus_download` 📄](workflow_yaml/data_ingestion/airbus/airbus_download.md): Downloads available AirBus imagery for the input geometry and time range.

- [`airbus/airbus_price` 📄](workflow_yaml/data_ingestion/airbus/airbus_price.md): Prices available AirBus imagery for the input geometry and time range.

- [`alos/alos_forest_extent_download` 📄](workflow_yaml/data_ingestion/alos/alos_forest_extent_download.md): Downloads Advanced Land Observing Satellite (ALOS) forest/non-forest classification map.

- [`alos/alos_forest_extent_download_merge` 📄](workflow_yaml/data_ingestion/alos/alos_forest_extent_download_merge.md): Downloads Advanced Land Observing Satellite (ALOS) forest/non-forest classification map and merges it into a single raster.

- [`bing/basemap_download` 📄](workflow_yaml/data_ingestion/bing/basemap_download.md): Downloads Bing Maps basemaps.

- [`bing/basemap_download_merge` 📄](workflow_yaml/data_ingestion/bing/basemap_download_merge.md): Downloads Bing Maps basemap tiles and merges them into a single raster.

- [`cdl/download_cdl` 📄](workflow_yaml/data_ingestion/cdl/download_cdl.md): Downloads crop classes maps in the continental USA for the input time range.

- [`dem/download_dem` 📄](workflow_yaml/data_ingestion/dem/download_dem.md): Downloads digital elevation map tiles that intersect with the input geometry and time range.

- [`gedi/download_gedi` 📄](workflow_yaml/data_ingestion/gedi/download_gedi.md): Downloads GEDI products for the input region and time range.

- [`gedi/download_gedi_rh100` 📄](workflow_yaml/data_ingestion/gedi/download_gedi_rh100.md): Downloads L2B GEDI products and extracts RH100 variables.

- [`glad/glad_forest_extent_download` 📄](workflow_yaml/data_ingestion/glad/glad_forest_extent_download.md): Downloads Global Land Analysis (GLAD) forest extent data.

- [`glad/glad_forest_extent_download_merge` 📄](workflow_yaml/data_ingestion/glad/glad_forest_extent_download_merge.md): Downloads the tiles from Global Land Analysis (GLAD) forest data that intersect with the user input geometry and time range, and merges them into a single raster.

- [`gnatsgo/download_gnatsgo` 📄](workflow_yaml/data_ingestion/gnatsgo/download_gnatsgo.md): Downloads gNATSGO raster data that intersect with the input geometry and time range.

- [`hansen/hansen_forest_change_download` 📄](workflow_yaml/data_ingestion/hansen/hansen_forest_change_download.md): Downloads and merges Global Forest Change (Hansen) rasters that intersect the user-provided geometry/time range.

- [`landsat/preprocess_landsat` 📄](workflow_yaml/data_ingestion/landsat/preprocess_landsat.md): Downloads and preprocesses LANDSAT tiles that intersect with the input geometry and time range.

- [`modis/download_modis_surface_reflectance` 📄](workflow_yaml/data_ingestion/modis/download_modis_surface_reflectance.md): Downloads MODIS 8-day surface reflectance rasters that intersect with the input geometry and time range.

- [`modis/download_modis_vegetation_index` 📄](workflow_yaml/data_ingestion/modis/download_modis_vegetation_index.md): Downloads MODIS 16-day vegetation index products that intersect with the input geometry and time range.

- [`naip/download_naip` 📄](workflow_yaml/data_ingestion/naip/download_naip.md): Downloads NAIP tiles that intersect with the input geometry and time range.

- [`osm_road_geometries` 📄](workflow_yaml/data_ingestion/osm_road_geometries.md): Downloads road geometry for input region from Open Street Maps.

- [`sentinel1/preprocess_s1` 📄](workflow_yaml/data_ingestion/sentinel1/preprocess_s1.md): Downloads and preprocesses tiles of Sentinel-1 imagery that intersect with the input Sentinel-2 products in the input time range.

- [`sentinel2/cloud_ensemble` 📄](workflow_yaml/data_ingestion/sentinel2/cloud_ensemble.md): Computes the cloud probability of a Sentinel-2 L2A raster using an ensemble of five cloud segmentation models.

- [`sentinel2/improve_cloud_mask` 📄](workflow_yaml/data_ingestion/sentinel2/improve_cloud_mask.md): Improves cloud masks by merging the product cloud mask with cloud and shadow masks computed by machine learning segmentation models.

- [`sentinel2/improve_cloud_mask_ensemble` 📄](workflow_yaml/data_ingestion/sentinel2/improve_cloud_mask_ensemble.md): Improves cloud masks by merging the product cloud mask with cloud and shadow masks computed by an ensemble of machine learning segmentation models.

- [`sentinel2/preprocess_s2` 📄](workflow_yaml/data_ingestion/sentinel2/preprocess_s2.md): Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range.

- [`sentinel2/preprocess_s2_ensemble_masks` 📄](workflow_yaml/data_ingestion/sentinel2/preprocess_s2_ensemble_masks.md): Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using an ensemble of cloud and shadow segmentation models.

- [`sentinel2/preprocess_s2_improved_masks` 📄](workflow_yaml/data_ingestion/sentinel2/preprocess_s2_improved_masks.md): Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using cloud and shadow segmentation models.

- [`soil/soilgrids` 📄](workflow_yaml/data_ingestion/soil/soilgrids.md): Downloads digital soil mapping information from SoilGrids for the input geometry.

- [`soil/usda` 📄](workflow_yaml/data_ingestion/soil/usda.md): Downloads USDA soil classification raster.

- [`spaceeye/spaceeye` 📄](workflow_yaml/data_ingestion/spaceeye/spaceeye.md): Runs the SpaceEye cloud removal pipeline, yielding daily cloud-free images for the input geometry and time range.

- [`spaceeye/spaceeye_inference` 📄](workflow_yaml/data_ingestion/spaceeye/spaceeye_inference.md): Performs SpaceEye inference to generate daily cloud-free images given Sentinel data and cloud masks.

- [`spaceeye/spaceeye_interpolation` 📄](workflow_yaml/data_ingestion/spaceeye/spaceeye_interpolation.md): Runs the SpaceEye cloud removal pipeline using an interpolation-based algorithm, yielding daily cloud-free images for the input geometry and time range.

- [`spaceeye/spaceeye_interpolation_inference` 📄](workflow_yaml/data_ingestion/spaceeye/spaceeye_interpolation_inference.md): Performs temporal damped interpolation to generate daily cloud-free images given Sentinel-2 data and cloud masks.

- [`spaceeye/spaceeye_preprocess` 📄](workflow_yaml/data_ingestion/spaceeye/spaceeye_preprocess.md): Runs the SpaceEye preprocessing pipeline.

- [`spaceeye/spaceeye_preprocess_ensemble` 📄](workflow_yaml/data_ingestion/spaceeye/spaceeye_preprocess_ensemble.md): Runs the SpaceEye preprocessing pipeline with an ensemble of cloud segmentation models.

- [`user_data/ingest_geometry` 📄](workflow_yaml/data_ingestion/user_data/ingest_geometry.md): Adds user geometries into the cluster storage, allowing for them to be used on workflows.

- [`user_data/ingest_raster` 📄](workflow_yaml/data_ingestion/user_data/ingest_raster.md): Adds user rasters into the cluster storage, allowing for them to be used on workflows.

- [`user_data/ingest_smb` 📄](workflow_yaml/data_ingestion/user_data/ingest_smb.md): Adds user rasters into the cluster storage from an SMB share, allowing for them to be used on workflows.

- [`weather/download_chirps` 📄](workflow_yaml/data_ingestion/weather/download_chirps.md): Downloads accumulated precipitation data from the CHIRPS dataset.

- [`weather/download_era5` 📄](workflow_yaml/data_ingestion/weather/download_era5.md): Hourly estimated weather variables.

- [`weather/download_era5_monthly` 📄](workflow_yaml/data_ingestion/weather/download_era5_monthly.md): Monthly estimated weather variables.

- [`weather/download_gridmet` 📄](workflow_yaml/data_ingestion/weather/download_gridmet.md): Daily surface meteorological properties from GridMET.

- [`weather/download_herbie` 📄](workflow_yaml/data_ingestion/weather/download_herbie.md): Downloads forecast data for provided location & time range using herbie python package.

- [`weather/download_terraclimate` 📄](workflow_yaml/data_ingestion/weather/download_terraclimate.md): Monthly climate and hydroclimate properties from TerraClimate.

- [`weather/get_ambient_weather` 📄](workflow_yaml/data_ingestion/weather/get_ambient_weather.md): Downloads weather data from an Ambient Weather station.

- [`weather/get_forecast` 📄](workflow_yaml/data_ingestion/weather/get_forecast.md): Downloads weather forecast data from NOAA Global Forecast System (GFS) for the input time range.

- [`weather/herbie_forecast` 📄](workflow_yaml/data_ingestion/weather/herbie_forecast.md): Downloads forecast observations for provided location & time range using herbie python package.


## data_processing

- [`chunk_onnx/chunk_onnx` 📄](workflow_yaml/data_processing/chunk_onnx/chunk_onnx.md): Runs an Onnx model over all rasters in the input to produce a single raster.

- [`chunk_onnx/chunk_onnx_sequence` 📄](workflow_yaml/data_processing/chunk_onnx/chunk_onnx_sequence.md): Runs an Onnx model over all rasters in the input to produce a single raster.

- [`clip/clip` 📄](workflow_yaml/data_processing/clip/clip.md): Performs a soft clip on an input raster based on a provided reference geometry.

- [`gradient/raster_gradient` 📄](workflow_yaml/data_processing/gradient/raster_gradient.md): Computes the gradient of each band of the input raster with a Sobel operator.

- [`heatmap/classification` 📄](workflow_yaml/data_processing/heatmap/classification.md): Utilizes input Sentinel-2 satellite imagery & the sensor samples as labeled data that contain nutrient information (Nitrogen, Carbon, pH, Phosphorus) to train a model using Random Forest classifier. The inference operation predicts nutrients in soil for the chosen farm boundary.


- [`index/index` 📄](workflow_yaml/data_processing/index/index.md): Computes an index from the bands of an input raster.

- [`linear_trend/chunked_linear_trend` 📄](workflow_yaml/data_processing/linear_trend/chunked_linear_trend.md): Computes the pixel-wise linear trend of a list of rasters (e.g. NDVI).

- [`merge/match_merge_to_ref` 📄](workflow_yaml/data_processing/merge/match_merge_to_ref.md): Resamples input rasters to the reference rasters' grid.

- [`outlier/detect_outlier` 📄](workflow_yaml/data_processing/outlier/detect_outlier.md): Fits a single-component Gaussian Mixture Model (GMM) over input data to detect outliers according to the threshold parameter.

- [`threshold/threshold_raster` 📄](workflow_yaml/data_processing/threshold/threshold_raster.md): Thresholds values of the input raster if higher than the threshold parameter.

- [`timeseries/timeseries_aggregation` 📄](workflow_yaml/data_processing/timeseries/timeseries_aggregation.md): Computes the mean, standard deviation, maximum, and minimum values of all regions of the raster and aggregates them into a timeseries.

- [`timeseries/timeseries_masked_aggregation` 📄](workflow_yaml/data_processing/timeseries/timeseries_masked_aggregation.md): Computes the mean, standard deviation, maximum, and minimum values of all regions of the raster considered by the mask and aggregates them into a timeseries.


## farm_ai

- [`agriculture/canopy_cover` 📄](workflow_yaml/farm_ai/agriculture/canopy_cover.md): Estimates pixel-wise canopy cover for a region and date.

- [`agriculture/change_detection` 📄](workflow_yaml/farm_ai/agriculture/change_detection.md): Identifies changes/outliers over NDVI across dates.

- [`agriculture/emergence_summary` 📄](workflow_yaml/farm_ai/agriculture/emergence_summary.md): Calculates emergence statistics using thresholded MSAVI (mean, standard deviation, maximum and minimum) for the input geometry and time range.

- [`agriculture/green_house_gas_fluxes` 📄](workflow_yaml/farm_ai/agriculture/green_house_gas_fluxes.md): Computes Green House Fluxes for a region and date range

- [`agriculture/heatmap_using_classification` 📄](workflow_yaml/farm_ai/agriculture/heatmap_using_classification.md): The workflow generates a nutrient heatmap for samples provided by user by downloading the samples from user input.

- [`agriculture/heatmap_using_classification_admag` 📄](workflow_yaml/farm_ai/agriculture/heatmap_using_classification_admag.md): This workflow integrate the ADMAG API to download prescriptions and generate heatmap.

- [`agriculture/heatmap_using_neighboring_data_points` 📄](workflow_yaml/farm_ai/agriculture/heatmap_using_neighboring_data_points.md): Creates heatmap using the neighbors by performing spatial interpolation operations. It utilizes soil information collected at optimal sensor/sample locations and downloaded sentinel satellite imagery.

- [`agriculture/methane_index` 📄](workflow_yaml/farm_ai/agriculture/methane_index.md): Computes methane index from ultra emitters for a region and date range.

- [`agriculture/ndvi_summary` 📄](workflow_yaml/farm_ai/agriculture/ndvi_summary.md): Calculates NDVI statistics (mean, standard deviation, maximum and minimum) for the input geometry and time range.

- [`agriculture/weed_detection` 📄](workflow_yaml/farm_ai/agriculture/weed_detection.md): Generates shape files for similarly colored regions in the input raster.

- [`carbon_local/admag_carbon_integration` 📄](workflow_yaml/farm_ai/carbon_local/admag_carbon_integration.md): Computes the offset amount of carbon that would be sequestered in a seasonal field using Microsoft Azure Data Manager for Agriculture (ADMAg) data.

- [`carbon_local/carbon_whatif` 📄](workflow_yaml/farm_ai/carbon_local/carbon_whatif.md): Computes the offset amount of carbon that would be sequestered in a seasonal field using the baseline (historical) and scenario (time range interested in) information.

- [`land_cover_mapping/conservation_practices` 📄](workflow_yaml/farm_ai/land_cover_mapping/conservation_practices.md): Identifies conservation practices (terraces and grassed waterways) using elevation data.

- [`land_degradation/landsat_ndvi_trend` 📄](workflow_yaml/farm_ai/land_degradation/landsat_ndvi_trend.md): Estimates a linear trend over NDVI computer over LANDSAT tiles that intersect with the input geometry and time range.

- [`land_degradation/ndvi_linear_trend` 📄](workflow_yaml/farm_ai/land_degradation/ndvi_linear_trend.md): Computes the pixel-wise NDVI linear trend over the input raster.

- [`segmentation/segment_basemap` 📄](workflow_yaml/farm_ai/segmentation/segment_basemap.md): Downloads basemap with BingMaps API and runs Segment Anything Model (SAM) over them with points and/or bounding boxes as prompts.

- [`segmentation/segment_s2` 📄](workflow_yaml/farm_ai/segmentation/segment_s2.md): Downloads Sentinel-2 imagery and runs Segment Anything Model (SAM) over them with points and/or bounding boxes as prompts.

- [`sensor/optimal_locations` 📄](workflow_yaml/farm_ai/sensor/optimal_locations.md): Identify optimal locations by performing clustering operation using Gaussian Mixture model on computed raster indices.

- [`water/irrigation_classification` 📄](workflow_yaml/farm_ai/water/irrigation_classification.md): Develops 30m pixel-wise irrigation probability map.


## forest_ai

- [`deforestation/alos_trend_detection` 📄](workflow_yaml/forest_ai/deforestation/alos_trend_detection.md): Detects increase/decrease trends in forest pixel levels over the user-input geometry and time range for the ALOS forest map.

- [`deforestation/ordinal_trend_detection` 📄](workflow_yaml/forest_ai/deforestation/ordinal_trend_detection.md): Detects increase/decrease trends in the pixel levels over the user-input geometry and time range.


## ml

- [`crop_segmentation` 📄](workflow_yaml/ml/crop_segmentation.md): Runs a crop segmentation model based on NDVI from SpaceEye imagery along the year.

- [`dataset_generation/datagen_crop_segmentation` 📄](workflow_yaml/ml/dataset_generation/datagen_crop_segmentation.md): Generates a dataset for crop segmentation, based on NDVI raster and Crop Data Layer (CDL) maps.

- [`driveway_detection` 📄](workflow_yaml/ml/driveway_detection.md): Detects driveways in front of houses.

- [`segment_anything/basemap_prompt_segmentation` 📄](workflow_yaml/ml/segment_anything/basemap_prompt_segmentation.md): Runs Segment Anything Model (SAM) over BingMaps basemap rasters with points and/or bounding boxes as prompts.

- [`segment_anything/s2_prompt_segmentation` 📄](workflow_yaml/ml/segment_anything/s2_prompt_segmentation.md): Runs Segment Anything Model (SAM) over Sentinel-2 rasters with points and/or bounding boxes as prompts.


