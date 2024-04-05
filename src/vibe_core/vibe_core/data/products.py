import mimetypes
import re
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
class AlosProduct(DataVibe):
    """Represents metadata information about an Advanced Land Observing Satellite (ALOS) product."""

    pass


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
class GLADProduct(DataVibe):
    """Represents metadata information about a Global Land Analysis (GLAD) product."""

    url: str
    """The URL of the GLAD product."""

    @property
    def tile_name(self) -> str:
        """The tile name of the GLAD product."""
        # Extract the tile name from the URL
        tile_name = self.url.split("/")[-1].split(".")[0]
        return tile_name


@dataclass
class HansenProduct(DataVibe):
    """
    Represents metadata information about a Hansen product.
    """

    asset_keys = ["treecover2000", "gain", "lossyear", "datamask", "first", "last"]
    """ The asset keys (dataset layers) for the Hansen products."""

    asset_url: str = field(default_factory=str)
    """ The URL of the Hansen product."""

    def __post_init__(self):
        super().__post_init__()
        valid = self.validate_url()
        if not valid:
            raise ValueError(f"Invalid URL: {self.asset_url}")

    def validate_url(self):
        # Urls are expected to be in the format:
        # 'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_treecover2000_20N_090W.tif'
        pattern = (
            r"https://storage\.googleapis\.com/earthenginepartners-hansen"
            r"/GFC-\d{4}-v\d+\.\d+/Hansen_GFC-\d{4}-v\d+\.\d+_\w+"
            r"_\d{2}[NS]_\d{3}[WE]\.tif"
        )
        match = re.match(pattern, self.asset_url)
        return bool(match)

    @staticmethod
    def extract_hansen_url_property(
        asset_url: str, regular_expression: str, property_name: str
    ) -> str:
        """Extracts the property from the base URL and the tile name."""

        # Use re.search to find the pattern in the URL
        match = re.search(regular_expression, asset_url)

        if match is None:
            raise ValueError(f"Could not extract {property_name} from {asset_url}")

        return match.group(1)

    @staticmethod
    def extract_tile_name(asset_url: str) -> str:
        """Extracts the tile name from the base URL and the tile name."""

        # Define the regex pattern for the tile name
        # The tile name is expected to be in the format: '20N_090W'
        pattern = r"(\d{2}[NS]_\d{3}[WE])"

        return HansenProduct.extract_hansen_url_property(asset_url, pattern, "tile name")

    @staticmethod
    def extract_last_year(asset_url: str) -> int:
        """Extracts the last year from the base URL and the tile name."""

        # Define the regex pattern for the last year - e.g., GFC-2022-v1.10 -> 2022
        pattern = r"GFC-(\d{4})-"

        return int(HansenProduct.extract_hansen_url_property(asset_url, pattern, "last year"))

    @staticmethod
    def extract_version(asset_url: str) -> str:
        """Extracts the version from the base URL and the tile name."""

        # Define the regex pattern for the version - e.g., GFC-2022-v1.10 -> v1.10
        pattern = r"GFC-\d{4}-(v\d+\.\d+)"

        return HansenProduct.extract_hansen_url_property(asset_url, pattern, "version")

    @staticmethod
    def extract_layer_name(asset_url: str) -> str:
        """Extracts the layer name from the base URL and the tile name."""

        # Define the regex pattern for the layer name
        pattern = r"_(\w+)_(\d{2}[NS]_\d{3}[WE])"

        return HansenProduct.extract_hansen_url_property(asset_url, pattern, "layer name")

    @property
    def tile_name(self) -> str:
        """The tile name of the Hansen product."""
        return self.extract_tile_name(self.asset_url)

    @property
    def last_year(self) -> int:
        """The last year of the Hansen product."""
        return self.extract_last_year(self.asset_url)

    @property
    def version(self) -> str:
        """The version of the Hansen product."""
        return self.extract_version(self.asset_url)

    @property
    def layer_name(self) -> str:
        """The layer name of the Hansen product."""
        return self.extract_layer_name(self.asset_url)


@dataclass
class EsriLandUseLandCoverProduct(DataVibe):
    """Represents metadata information about Esri LandUse/LandCover (9-class) dataset."""

    pass


@dataclass
class HerbieProduct(DataVibe):
    """Stands for Herbie products metadata
    https://herbie.readthedocs.io/en/latest/index.html
    """

    model: str
    """model name, e.g., 'hrrr', 'hrrrak', 'rap', 'gfs', 'gfs_wave', 'rrfs'"""
    product: str
    """product file type: 'sfc' (surface fields), 'prs' (pressure fields), 'nat' (native fields),
    'subh' (subhourly fields)
    """
    lead_time_hours: int
    """lead time in hours"""
    search_text: str
    """regular expression used to search on GRIB2 Index files"""


@dataclass
class BingMapsProduct(DataVibe):
    """
    Represents metadata of a BingMaps product.
    """

    url: str
    """The download URL of the product."""

    zoom_level: int
    """The zoom level of the product."""

    imagery_set: str
    """The imagery set of the product."""

    map_layer: str
    """The map layer of the product."""

    orientation: float
    """The orientation of the product."""
