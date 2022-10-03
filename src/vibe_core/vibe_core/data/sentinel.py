import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto
from typing import Any, Dict, List, Union

from dateutil.parser import parse as parse_date
from strenum import StrEnum

from . import AssetVibe, DataVibe
from .core_types import TimeRange, gen_guid
from .rasters import CategoricalRaster, CloudRaster, Raster, RasterSequence


class S2ProcessingLevel(StrEnum):
    L1C = auto()
    L2A = auto()


def discriminator_date(product_name: str) -> datetime:
    return parse_date(product_name.split("_")[-1])


# TODO: remove the generic dictionary for a list of actual information that we want to keep.
# consider having two representations, one for Sentinel 1 another for Sentinel 2.
@dataclass
class SentinelProduct(DataVibe):
    product_name: str
    orbit_number: int
    relative_orbit_number: int
    orbit_direction: str
    platform: str
    extra_info: Dict[str, Any]  # Allows for generic information to be stored.


@dataclass
class Sentinel1Product(SentinelProduct):
    sensor_mode: str
    polarisation_mode: str


@dataclass
class Sentinel2Product(SentinelProduct):
    tile_id: str
    processing_level: str


@dataclass
class CloudMask(CategoricalRaster, CloudRaster, Sentinel2Product):
    pass


@dataclass
class SentinelRaster(Raster, SentinelProduct):
    pass


@dataclass
class DownloadedSentinel2Product(Sentinel2Product):
    CLOUD_MASK: str = "cloud_mask"
    asset_map: Dict[str, str] = field(default_factory=dict)

    def add_downloaded_band(self, band_name: str, asset_path: str):
        band_guid = gen_guid()
        # Double check mime type
        self.asset_map[band_name] = band_guid
        # Check if this is also true for L1A
        asset_type = mimetypes.guess_type(asset_path)[0]
        if asset_type is None or asset_type not in ["image/tiff", "image/jp2"]:
            raise ValueError(
                f"TIFF and JP2 files supported for Sentinel2 downloads. Found {asset_type}."
            )
        self.assets.append(AssetVibe(asset_path, asset_type, band_guid))

    def _lookup_asset(self, guid: str) -> AssetVibe:
        def id_eq(x: AssetVibe):
            return x.id == guid

        return list(filter(id_eq, self.assets))[0]

    def add_downloaded_cloudmask(self, asset_path: str):
        cloud_guid = gen_guid()
        # Double check mime type
        self.asset_map[self.CLOUD_MASK] = cloud_guid
        self.assets.append(AssetVibe(asset_path, "application/gml+xml", cloud_guid))

    def get_downloaded_band(self, band_name: str) -> AssetVibe:
        guid = self.asset_map[band_name]
        return self._lookup_asset(guid)

    def get_downloaded_cloudmask(self) -> AssetVibe:
        guid = self.asset_map[self.CLOUD_MASK]
        return self._lookup_asset(guid)


@dataclass
class DownloadedSentinel1Product(Sentinel1Product):
    ZIP_FILE = "zip"
    asset_map: Dict[str, str] = field(default_factory=dict)

    def _lookup_asset(self, guid: str) -> AssetVibe:
        def id_eq(x: AssetVibe):
            return x.id == guid

        return list(filter(id_eq, self.assets))[0]

    def add_zip_asset(self, asset_path: str):
        zip_guid = gen_guid()
        # Double check mime type
        self.asset_map[self.ZIP_FILE] = zip_guid
        self.assets.append(AssetVibe(asset_path, "application/zip", zip_guid))

    def get_zip_asset(self) -> AssetVibe:
        guid = self.asset_map[self.ZIP_FILE]
        return self._lookup_asset(guid)


@dataclass
class Sentinel1Raster(Raster, Sentinel1Product):
    tile_id: str


@dataclass
class Sentinel2Raster(Raster, Sentinel2Product):
    def __post_init__(self):
        super().__post_init__()
        self.quantification_value = 10000


@dataclass
class Sentinel2CloudProbability(CloudRaster, Sentinel2Product):
    pass


@dataclass
class Sentinel2CloudMask(CloudMask, Sentinel2Product):
    pass


class SpaceEyeRaster(Sentinel2Raster):
    pass


@dataclass
class TiledSentinel1Product(DownloadedSentinel1Product):
    tile_id: str = ""

    def __post_init__(self):
        if not self.tile_id:
            raise ValueError("tile_id is a mandatory argument even though it isn't.")
        return super().__post_init__()


@dataclass
class Sentinel1RasterOrbitGroup(Sentinel1Raster):
    asset_map: Dict[str, str] = field(default_factory=dict)

    def add_raster(self, raster: Sentinel1Raster):
        asset = raster.raster_asset
        self.asset_map[asset.id] = raster.time_range[0].isoformat()
        self.assets.append(raster.raster_asset)

    def get_ordered_assets(self) -> List[AssetVibe]:
        return sorted(self.assets, key=lambda x: datetime.fromisoformat(self.asset_map[x.id]))


@dataclass
class Sentinel2RasterOrbitGroup(Sentinel2Raster):
    asset_map: Dict[str, str] = field(default_factory=dict)

    def add_raster(self, raster: Sentinel2Raster):
        asset = raster.raster_asset
        self.asset_map[asset.id] = discriminator_date(raster.product_name).isoformat()
        self.assets.append(raster.raster_asset)

    def get_ordered_assets(self) -> List[AssetVibe]:
        return sorted(
            self.assets, key=lambda x: datetime.fromisoformat(self.asset_map[x.id]), reverse=True
        )


@dataclass
class Sentinel2CloudMaskOrbitGroup(Sentinel2CloudMask):
    asset_map: Dict[str, str] = field(default_factory=dict)

    def add_raster(self, raster: Sentinel2CloudMask):
        asset = raster.raster_asset
        self.asset_map[asset.id] = discriminator_date(raster.product_name).isoformat()
        self.assets.append(raster.raster_asset)

    def get_ordered_assets(self) -> List[AssetVibe]:
        return sorted(
            self.assets, key=lambda x: datetime.fromisoformat(self.asset_map[x.id]), reverse=True
        )


@dataclass
class TileSequence(RasterSequence):
    write_time_range: TimeRange = field(default_factory=tuple)

    def __post_init__(self):
        super().__post_init__()
        if len(self.write_time_range) != 2:
            raise ValueError(
                "write_time_range must be a tuple of two datetime items,"
                f"found {self.write_time_range=}"
            )


@dataclass
class Sentinel1RasterTileSequence(TileSequence, Sentinel1Raster):
    pass


@dataclass
class Sentinel2RasterTileSequence(TileSequence, Sentinel2Raster):
    pass


@dataclass
class Sentinel2CloudMaskTileSequence(TileSequence, Sentinel2CloudMask):
    pass


@dataclass
class SpaceEyeRasterSequence(TileSequence, SpaceEyeRaster):
    pass


ListTileData = List[Union[Sentinel1Raster, Sentinel2Raster, Sentinel2CloudMask]]
TileSequenceData = Union[
    Sentinel1RasterTileSequence,
    Sentinel2RasterTileSequence,
    Sentinel2CloudMaskTileSequence,
]

Tile2Sequence = {
    Sentinel1Raster: Sentinel1RasterTileSequence,
    Sentinel2Raster: Sentinel2RasterTileSequence,
    Sentinel2CloudMask: Sentinel2CloudMaskTileSequence,
}

Sequence2Tile = {
    Sentinel1RasterTileSequence: Sentinel1Raster,
    Sentinel2RasterTileSequence: Sentinel2Raster,
    Sentinel2CloudMaskTileSequence: Sentinel2CloudMask,
    SpaceEyeRasterSequence: SpaceEyeRaster,
}
