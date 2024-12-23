# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Core data model for FarmVibes.AI."""

from .airbus import AirbusPrice, AirbusProduct, AirbusRaster
from .core_types import (
    AssetVibe,
    BaseVibe,
    BaseVibeDict,
    BBox,
    CarbonOffsetInfo,
    ChipWindow,
    DataSummaryStatistics,
    DataVibe,
    ExternalReference,
    ExternalReferenceList,
    FoodFeatures,
    FoodVibe,
    GeometryCollection,
    GHGFlux,
    GHGProtocolVibe,
    OrdinalTrendTest,
    Point,
    ProteinSequence,
    RasterPixelCount,
    TimeRange,
    TimeSeries,
    TypeDictVibe,
    gen_guid,
    gen_hash_id,
)
from .farm import (
    ADMAgPrescription,
    ADMAgPrescriptionInput,
    ADMAgSeasonalFieldInput,
    FertilizerInformation,
    HarvestInformation,
    OrganicAmendmentInformation,
    SeasonalFieldInformation,
    TillageInformation,
)
from .products import (
    AlosProduct,
    ChirpsProduct,
    ClimatologyLabProduct,
    DemProduct,
    Era5Product,
    EsriLandUseLandCoverProduct,
    GEDIProduct,
    GLADProduct,
    GNATSGOProduct,
    HansenProduct,
    HerbieProduct,
    LandsatProduct,
    ModisProduct,
    NaipProduct,
)
from .rasters import (
    CategoricalRaster,
    ChunkLimits,
    CloudRaster,
    DemRaster,
    GNATSGORaster,
    LandsatRaster,
    ModisRaster,
    NaipRaster,
    Raster,
    RasterChunk,
    RasterIlluminance,
    RasterSequence,
    SamMaskRaster,
)
from .sentinel import (
    DownloadedSentinel1Product,
    DownloadedSentinel2Product,
    S2ProcessingLevel,
    Sentinel1Product,
    Sentinel1Raster,
    Sentinel1RasterOrbitGroup,
    Sentinel2CloudMask,
    Sentinel2CloudMaskOrbitGroup,
    Sentinel2CloudProbability,
    Sentinel2Product,
    Sentinel2Raster,
    Sentinel2RasterOrbitGroup,
    SentinelProduct,
    SpaceEyeRaster,
    TiledSentinel1Product,
)
from .utils import StacConverter
from .weather import GfsForecast, Grib, gen_forecast_time_hash_id
