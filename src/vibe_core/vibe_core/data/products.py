import mimetypes
from dataclasses import dataclass, field
from typing import Dict, cast

from .core_types import AssetVibe, DataVibe, gen_guid

CDL_DOWNLOAD_URL = (
    "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{}_30m_cdls.zip"
)


@dataclass
class DemProduct(DataVibe):
    """Holds metadata information about a DEM tile.

    The DemProduct type is the expected output of a list-like operator and the expected input
    type of a download-like operator.
    """

    tile_id: str
    resolution: int


@dataclass
class NaipProduct(DataVibe):
    """Holds metadata information about a NAIP tile.

    The NaipProduct type is the expected output of a list-like operator and the
    type of a download-like operator.
    """

    tile_id: str
    year: int
    resolution: float


@dataclass
class LandsatProduct(DataVibe):
    tile_id: str = ""
    asset_map: Dict[str, str] = field(default_factory=dict)

    def add_downloaded_band(self, band_name: str, asset_path: str):
        band_guid = gen_guid()
        self.asset_map[band_name] = band_guid
        self.assets.append(
            AssetVibe(asset_path, cast(str, mimetypes.guess_type(asset_path)[0]), band_guid)
        )

    def get_downloaded_band(self, band_name: str) -> AssetVibe:
        try:
            band_guid = self.asset_map[band_name]
        except KeyError:
            raise ValueError(f"Band {band_name} not found or downloaded")
        return next((a for a in self.assets if a.id == band_guid))


@dataclass
class ChirpsProduct(DataVibe):
    url: str


@dataclass
class CDLProduct(DataVibe):
    pass


@dataclass
class Era5Product(DataVibe):
    item_id: str
    var: str


@dataclass
class ModisVegetationProduct(DataVibe):
    resolution: int
