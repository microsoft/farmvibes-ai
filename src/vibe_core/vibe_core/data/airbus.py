from dataclasses import dataclass
from typing import Any, Dict

from .core_types import DataVibe
from .rasters import Raster


@dataclass
class AirbusProduct(DataVibe):
    """
    Airbus product metadata obtained from the search API. Contains no image assets.
    """

    acquisition_id: str
    extra_info: Dict[str, Any]


@dataclass
class AirbusPrice(DataVibe):
    price: float


@dataclass
class AirbusRaster(Raster, AirbusProduct):
    """
    Airbus product downloaded with specific product type, radiometric processing, projection
    Contains slightly different metadata than `AirbusProduct`
    """

    pass
