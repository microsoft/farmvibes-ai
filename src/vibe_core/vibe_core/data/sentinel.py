# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data types and supporting functions for Sentinel data in FarmVibes.AI."""

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
    """Enum for Sentinel 2 processing levels."""

    L1C = auto()
    """Level 1C processing level."""
    L2A = auto()
    """Level 2A processing level."""


def discriminator_date(product_name: str) -> datetime:
    """Extract the date from a Sentinel-2 product name.

    Args:
        product_name: The name of the Sentinel-2 product.

    Returns:
        The date of the Sentinel-2 product as a datetime object.
    """
    return parse_date(product_name.split("_")[-1])


# TODO: remove the generic dictionary for a list of actual information that we want to keep.
# consider having two representations, one for Sentinel 1 another for Sentinel 2.
@dataclass
class SentinelProduct(DataVibe):
    """Represent a Sentinel product metadata (does not include the image data)."""

    product_name: str
    """The name of the Sentinel product."""
    orbit_number: int
    """The orbit number of the Sentinel product."""
    relative_orbit_number: int
    """The relative orbit number of the Sentinel product."""
    orbit_direction: str
    """The orbit direction of the Sentinel product."""
    platform: str
    """The platform of the Sentinel product."""
    extra_info: Dict[str, Any]  # Allows for generic information to be stored.
    """A dictionary with extra information about the Sentinel product."""


@dataclass
class Sentinel1Product(SentinelProduct):
    """Represent a Sentinel-1 product metadata."""

    sensor_mode: str
    """The sensor mode of the Sentinel-1 product."""
    polarisation_mode: str
    """The polarisation mode of the Sentinel-1 product."""


@dataclass
class Sentinel2Product(SentinelProduct):
    """Represent a Sentinel-2 product metadata."""

    tile_id: str
    """The tile ID of the Sentinel-2 product."""
    processing_level: str
    """The processing level of the Sentinel-2 product."""


@dataclass
class CloudMask(CategoricalRaster, CloudRaster, Sentinel2Product):
    """Represent a cloud mask raster for a Sentinel-2 product."""

    pass


@dataclass
class SentinelRaster(Raster, SentinelProduct):
    """Represent a raster for a Sentinel product."""

    pass


@dataclass
class DownloadedSentinel2Product(Sentinel2Product):
    """Represent a downloaded Sentinel-2 product."""

    CLOUD_MASK: str = "cloud_mask"
    """The key for the cloud mask asset in the asset map."""

    asset_map: Dict[str, str] = field(default_factory=dict)
    """A dictionary mapping the band name to the asset ID."""

    def add_downloaded_band(self, band_name: str, asset_path: str):
        """Add a downloaded band to the asset map of a :class:`DownloadedSentinel2Product` object.

        Args:
            band_name: The name of the band to add.
            asset_path: The path to the downloaded band file.

        Raises:
            ValueError: If the file type is not supported (types other than TIFF or JP2).
        """
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
        """Add a downloaded cloud mask to the asset map.

        Args:
            asset_path: The path to the downloaded cloud mask file.
        """
        cloud_guid = gen_guid()
        # Double check mime type
        self.asset_map[self.CLOUD_MASK] = cloud_guid
        self.assets.append(AssetVibe(asset_path, "application/gml+xml", cloud_guid))

    def get_downloaded_band(self, band_name: str) -> AssetVibe:
        """Return the downloaded band asset for the given band name.

        Args:
            band_name: The name of the band to return.

        Returns:
            The downloaded band asset.
        """
        guid = self.asset_map[band_name]
        return self._lookup_asset(guid)

    def get_downloaded_cloudmask(self) -> AssetVibe:
        """Retrieve the downloaded cloud mask asset.

        Returns:
            The downloaded cloud mask asset.
        """
        guid = self.asset_map[self.CLOUD_MASK]
        return self._lookup_asset(guid)


@dataclass
class DownloadedSentinel1Product(Sentinel1Product):
    """Represent a downloaded Sentinel-1 product."""

    ZIP_FILE = "zip"
    """The key for the zip asset in the asset map."""
    asset_map: Dict[str, str] = field(default_factory=dict)
    """A dictionary mapping the band name to the asset ID."""

    def _lookup_asset(self, guid: str) -> AssetVibe:
        def id_eq(x: AssetVibe):
            return x.id == guid

        return list(filter(id_eq, self.assets))[0]

    def add_zip_asset(self, asset_path: str):
        """Add a downloaded zip asset to the asset map of a :class:`DownloadedSentinel1Product`.

        Args:
            asset_path: The path to the downloaded zip file.
        """
        zip_guid = gen_guid()
        # Double check mime type
        self.asset_map[self.ZIP_FILE] = zip_guid
        self.assets.append(AssetVibe(asset_path, "application/zip", zip_guid))

    def get_zip_asset(self) -> AssetVibe:
        """Retrieve the downloaded zip asset.

        Returns:
            The downloaded zip asset.
        """
        guid = self.asset_map[self.ZIP_FILE]
        return self._lookup_asset(guid)


@dataclass
class Sentinel1Raster(Raster, Sentinel1Product):
    """Represent a raster for a Sentinel-1 product."""

    tile_id: str
    """The tile ID of the raster."""


@dataclass
class Sentinel2Raster(Raster, Sentinel2Product):
    """Represent a raster for a Sentinel-2 product."""

    def __post_init__(self):
        super().__post_init__()
        self.scale = 1e-4


@dataclass
class Sentinel2CloudProbability(CloudRaster, Sentinel2Product):
    """Represent a cloud probability raster for a Sentinel-2 product."""

    pass


@dataclass
class Sentinel2CloudMask(CloudMask, Sentinel2Product):
    """Represent a cloud mask raster for a Sentinel-2 product."""

    pass


class SpaceEyeRaster(Sentinel2Raster):
    """Represent a SpaceEye raster."""

    pass


@dataclass
class TiledSentinel1Product(DownloadedSentinel1Product):
    """Represent a tiled Sentinel-1 product."""

    tile_id: str = ""
    """The tile ID of the product."""

    def __post_init__(self):
        if not self.tile_id:
            raise ValueError("tile_id is a mandatory argument even though it isn't.")
        return super().__post_init__()


@dataclass
class Sentinel1RasterOrbitGroup(Sentinel1Raster):
    """Represent a group of Sentinel-1 raster orbits."""

    asset_map: Dict[str, str] = field(default_factory=dict)
    """A dictionary mapping the asset ID to the acquisition date."""

    def add_raster(self, raster: Sentinel1Raster):
        """Add a raster to the orbit group.

        Args:
            raster: The raster to add to the orbit group.
        """
        asset = raster.raster_asset
        self.asset_map[asset.id] = raster.time_range[0].isoformat()
        self.assets.append(raster.raster_asset)

    def get_ordered_assets(self) -> List[AssetVibe]:
        """Return the assets in the orbit group in ascending order of acquisition date.

        Returns:
            The list of sorted assets in the orbit group.
        """
        return sorted(self.assets, key=lambda x: datetime.fromisoformat(self.asset_map[x.id]))


@dataclass
class Sentinel2RasterOrbitGroup(Sentinel2Raster):
    """Represent a group of Sentinel-2 raster orbits."""

    asset_map: Dict[str, str] = field(default_factory=dict)
    """A dictionary mapping the asset ID to the acquisition date."""

    def add_raster(self, raster: Sentinel2Raster):
        """Add a raster to the orbit group.

        Args:
            raster: The raster to add to the orbit group.
        """
        asset = raster.raster_asset
        self.asset_map[asset.id] = discriminator_date(raster.product_name).isoformat()
        self.assets.append(raster.raster_asset)

    def get_ordered_assets(self) -> List[AssetVibe]:
        """Return the assets in the orbit group in ascending order of acquisition date.

        Returns:
            The list of sorted assets in the orbit group.
        """
        return sorted(
            self.assets, key=lambda x: datetime.fromisoformat(self.asset_map[x.id]), reverse=True
        )


@dataclass
class Sentinel2CloudMaskOrbitGroup(Sentinel2CloudMask):
    """Represent a group of Sentinel-2 cloud mask orbits."""

    asset_map: Dict[str, str] = field(default_factory=dict)
    """A dictionary mapping the asset ID to the acquisition date."""

    def add_raster(self, raster: Sentinel2CloudMask):
        """Add a raster to the orbit group.

        Args:
            raster: The raster to add to the orbit group.
        """
        asset = raster.raster_asset
        self.asset_map[asset.id] = discriminator_date(raster.product_name).isoformat()
        self.assets.append(raster.raster_asset)

    def get_ordered_assets(self) -> List[AssetVibe]:
        """Return the assets in the orbit group in ascending order of acquisition date.

        Returns:
            The list of sorted assets in the orbit group.
        """
        return sorted(
            self.assets, key=lambda x: datetime.fromisoformat(self.asset_map[x.id]), reverse=True
        )


@dataclass
class TileSequence(RasterSequence):
    """Represent a sequence of rasters for a tile."""

    write_time_range: TimeRange = field(default_factory=tuple)
    """The time range of the sequence."""

    def __post_init__(self):
        super().__post_init__()
        if len(self.write_time_range) != 2:
            raise ValueError(
                "write_time_range must be a tuple of two datetime items,"
                f"found {self.write_time_range=}"
            )


@dataclass
class Sentinel1RasterTileSequence(TileSequence, Sentinel1Raster):
    """Represent a sequence of Sentinel-1 rasters for a tile."""

    pass


@dataclass
class Sentinel2RasterTileSequence(TileSequence, Sentinel2Raster):
    """Represent a sequence of Sentinel-2 rasters for a tile."""

    pass


@dataclass
class Sentinel2CloudMaskTileSequence(TileSequence, Sentinel2CloudMask):
    """Represent a sequence of Sentinel-2 cloud masks for a tile."""

    pass


@dataclass
class SpaceEyeRasterSequence(TileSequence, SpaceEyeRaster):
    """Represent a sequence of SpaceEye rasters for a tile."""

    pass


TileData = Union[Sentinel1Raster, Sentinel2Raster, Sentinel2CloudMask]
"""
A type alias for any of the tile data classes (:class:`Sentinel1Raster`,
:class:`Sentinel2Raster`, and :class:`Sentinel2CloudMask`).
"""

ListTileData = List[TileData]
"""A type alias for a list of :const:`TileData`."""

TileSequenceData = Union[
    Sentinel1RasterTileSequence,
    Sentinel2RasterTileSequence,
    Sentinel2CloudMaskTileSequence,
]
"""
A type alias for any of the tile sequence data classes (:class:`Sentinel1RasterTileSequence`,
:class:`Sentinel2RasterTileSequence`, and :class:`Sentinel2CloudMaskTileSequence`).
"""

Tile2Sequence = {
    Sentinel1Raster: Sentinel1RasterTileSequence,
    Sentinel2Raster: Sentinel2RasterTileSequence,
    Sentinel2CloudMask: Sentinel2CloudMaskTileSequence,
}
"""A dictionary mapping the tile data classes to the tile sequence data classes."""

Sequence2Tile = {
    Sentinel1RasterTileSequence: Sentinel1Raster,
    Sentinel2RasterTileSequence: Sentinel2Raster,
    Sentinel2CloudMaskTileSequence: Sentinel2CloudMask,
    SpaceEyeRasterSequence: SpaceEyeRaster,
}
"""A dictionary mapping tile sequence data classes to tile data classes."""
