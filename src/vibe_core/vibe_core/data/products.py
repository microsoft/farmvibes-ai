from dataclasses import dataclass

from .core_types import DataVibe

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
    tile_id: str


@dataclass
class ChirpsProduct(DataVibe):
    url: str


@dataclass
class CDLProduct(DataVibe):
    pass
