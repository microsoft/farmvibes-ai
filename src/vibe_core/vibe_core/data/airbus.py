"""AirBus data types."""

from dataclasses import dataclass
from typing import Any, Dict

from .core_types import DataVibe
from .rasters import Raster


@dataclass
class AirbusProduct(DataVibe):
    """Represent Airbus product metadata obtained from the search API. Contains no image assets."""

    acquisition_id: str
    """The ID of the acquisition."""

    extra_info: Dict[str, Any]
    """A dictionary with extra information about the product."""


@dataclass
class AirbusPrice(DataVibe):
    """Represent the price of an Airbus product."""

    price: float
    """The price of the product."""


@dataclass
class AirbusRaster(Raster, AirbusProduct):
    """Airbus raster product.

    Represent an Airbus raster, downloaded with specific product type, radiometric processing,
    projection.
    """

    pass
