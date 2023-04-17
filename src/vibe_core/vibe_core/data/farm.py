from dataclasses import dataclass
from typing import Any, Dict, List

from .core_types import BaseVibe, DataVibe


@dataclass
class ADMAgSeasonalFieldInput(BaseVibe):
    """
    Represents an ADMAg Seasonal Field input.
    """

    farmer_id: str
    """The ID of the farmer."""
    seasonal_field_id: str
    """The ID of the seasonal field."""
    boundary_id: str
    """The ID of the boundary."""


@dataclass
class TillageInformation:
    """Represents a tillage operation in a field ."""

    start_date: str
    """The start date of the tillage operation."""
    end_date: str
    """The end date of the tillage operation."""
    implement: str
    """The implement used for the tillage operation."""


@dataclass
class FertilizerInformation:
    """Represents fertilizer practices operation."""

    start_date: str
    """The start date of the practice."""

    end_date: str
    """The end date of the practice."""

    application_type: str
    """The type of fertilizer application."""

    total_nitrogen: float
    """The total amount of nitrogen applied."""
    enhanced_efficiency_phosphorus: str
    """The type of enhanced efficiency phosphorus used."""


@dataclass
class OrganicAmendmentInformation:
    """Represents an organic amendment practice operation."""

    start_date: str
    """The start date of the organic amendment practice."""
    end_date: str
    """The end date of the organic amendment practice."""
    organic_amendment_type: str
    """The type of organic amendment applied."""
    organic_amendment_amount: float
    """The amount of organic amendment applied."""
    organic_amendment_percent_nitrogen: float
    """The percent nitrogen of the organic amendment."""
    organic_amendment_carbon_nitrogen_ratio: float
    """The carbon to nitrogen ratio of the organic amendment."""


@dataclass
class HarvestInformation:
    """Represents a harvest operation in a field."""

    is_grain: bool
    """Whether the crop is a grain (True) or not (False)."""
    start_date: str
    """The start date of the harvest operation."""
    end_date: str
    """The end date of the harvest operation."""
    crop_yield: float
    """The yield of the crop, in kg/ha."""
    stray_stover_hay_removal: float
    """The amount of stray stover or hay removed from the field after harvest, in kg/ha."""


@dataclass
class SeasonalFieldInformation(DataVibe):
    """Represents seasonal field information for a farm."""

    crop_name: str
    """The name of the crop grown in the seasonal field."""

    crop_type: str
    """The type of the crop grown in the seasonal field."""

    properties: Dict[str, Any]
    """A dictionary of additional properties for the seasonal field."""

    fertilizers: List[FertilizerInformation]
    """A list of :class:`FertilizerInformation` objects representing the
    fertilizer practices in the seasonal field."""

    harvests: List[HarvestInformation]
    """A list of :class:`HarvestInformation` objects representing the harvests
    for the seasonal field."""

    tillages: List[TillageInformation]
    """A list of :class:`TillageInformation` objects representing the tillage operations
    for the seasonal field."""

    organic_amendments: List[OrganicAmendmentInformation]
    """A list of :class:`OrganicAmendmentInformation` objects representing the organic
    amendments for the seasonal field."""
