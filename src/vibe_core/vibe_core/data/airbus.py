from dataclasses import dataclass
from typing import Any, Dict

from .core_types import DataVibe
from .rasters import Raster


@dataclass
class AirbusProduct(DataVibe):
    """
    Represents Airbus product metadata obtained from the search API.
    Contains no image assets.
    """

    acquisition_id: str
    """The ID of the acquisition."""

    extra_info: Dict[str, Any]
    """A dictionary with extra information about the product."""


@dataclass
class AirbusPrice(DataVibe):
    """Represents the price of an Airbus product."""

    price: float
    """The price of the product."""


@dataclass
class AirbusRaster(Raster, AirbusProduct):
    """
    Represents an Airbus raster product, downloaded with specific product type,
    radiometric processing, projection.
    """

    pass
