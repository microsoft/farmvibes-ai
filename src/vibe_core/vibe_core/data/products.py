import mimetypes
from dataclasses import dataclass, field
from typing import Dict, cast

from .core_types import AssetVibe, DataVibe, gen_guid

CDL_DOWNLOAD_URL = (
    "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{}_30m_cdls.zip"
)
"""The base URL for downloading CropDataLayer data.

:meta hide-value:
"""


@dataclass
class DemProduct(DataVibe):
    """Represents metadata information about a Digital Elevation Map (DEM) tile.

    The :class:`DemProduct` type is the expected output of a list-like operator
    and the expected input type of a download-like operator.
    """

    tile_id: str
    """The tile ID of the DEM tile."""

    resolution: int
    """The resolution of the DEM tile."""

    provider: str
    """The provider of the DEM tile."""


@dataclass
class NaipProduct(DataVibe):
    """Represents metadata information about a National Agricultural
    Imagery Program (NAIP) tile.

    The :class:`NaipProduct` type is the expected output of a list-like operator
    and the type of a download-like operator.
    """

    tile_id: str
    """The tile ID of the NAIP tile."""

    year: int
    """The year of the NAIP tile."""

    resolution: float
    """The resolution of the NAIP tile."""


@dataclass
class LandsatProduct(DataVibe):
    """Represents metadata information about a Landsat tile."""

    tile_id: str = ""
    """The tile ID of the Landsat tile."""

    asset_map: Dict[str, str] = field(default_factory=dict)
    """A dictionary mapping band names to asset IDs."""

    def add_downloaded_band(self, band_name: str, asset_path: str):
        """Adds a downloaded band to the asset map.

        :param band_name: The name of the band.

        :param asset_path: The path to the downloaded asset.
        """
        band_guid = gen_guid()
        self.asset_map[band_name] = band_guid
        self.assets.append(
            AssetVibe(asset_path, cast(str, mimetypes.guess_type(asset_path)[0]), band_guid)
        )

    def get_downloaded_band(self, band_name: str) -> AssetVibe:
        """Retrieves the downloaded band with the given name from the asset map.

        :param band_name: The name of the band to retrieve.

        :return: The downloaded band with the given name.
        :rtype: :class:`AssetVibe`

        :raises ValueError: If the band with the given name is not found or downloaded.
        """
        try:
            band_guid = self.asset_map[band_name]
        except KeyError:
            raise ValueError(f"Band {band_name} not found or downloaded")
        return next((a for a in self.assets if a.id == band_guid))


@dataclass
class ChirpsProduct(DataVibe):
    """Represents metadata information about a
    Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS) product.
    """

    url: str
    """The URL of the CHIRPS product."""


@dataclass
class CDLProduct(DataVibe):
    """Represents metadata information about a Crop Data Layer (CDL) product."""

    pass


@dataclass
class Era5Product(DataVibe):
    """Represents metadata information about an ERA5 product.

    :var item_id: The item ID of the ERA5 product.
    :var var: The variable of the ERA5 product.
    :var cds_request: A dictionary with the CDS request parameters.
    """

    item_id: str
    var: str
    cds_request: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class ModisProduct(DataVibe):
    """Represents metadata information about a
    Moderate Resolution Imaging Spectroradiometer (MODIS) product.
    """

    resolution: int
    """The resolution of the MODIS product."""


@dataclass
class GEDIProduct(DataVibe):
    """Represents metadata information about a
    Global Ecosystem Dynamics Investigation (GEDI) product.
    """

    product_name: str
    """The name of the GEDI product."""
    start_orbit: int
    """The start orbit of the GEDI product."""
    stop_orbit: int
    """The stop orbit of the GEDI product."""
    processing_level: str
    """The processing level of the GEDI product."""


@dataclass
class GNATSGOProduct(DataVibe):
    """Represents metadata information about a
    Gridded National Soil Survey Geographic Database (gNATSGO) product.
    """

    pass


@dataclass
class ClimatologyLabProduct(DataVibe):
    """Represents metadata information about a Climatology Lab product."""

    url: str
    """The URL of the Climatology Lab product."""
    variable: str
    """The variable of the Climatology Lab product."""


@dataclass
class EsriLandUseLandCoverProduct(DataVibe):
    """Represents metadata information about Esri LandUse/LandCover (9-class) dataset."""

    pass
