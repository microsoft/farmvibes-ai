from dataclasses import dataclass
from typing import Any, Dict, List

from .core_types import BaseVibe, DataVibe


@dataclass
class ADMAgSeasonalFieldInput(BaseVibe):
    farmer_id: str
    seasonal_field_id: str
    boundary_id: str


@dataclass
class TillageInformation:
    start_date: str
    end_date: str
    implement: str


@dataclass
class FertilizerInformation:
    start_date: str
    end_date: str
    application_type: str
    total_nitrogen: float
    enhanced_efficiency_phosphorus: str


@dataclass
class OrganicAmendmentInformation:
    start_date: str
    end_date: str
    organic_amendment_type: str
    organic_amendment_amount: float
    organic_amendment_percent_nitrogen: float
    organic_amendment_carbon_nitrogen_ratio: float


@dataclass
class HarvestInformation:
    is_grain: bool
    start_date: str
    end_date: str
    crop_yield: float
    stray_stover_hay_removal: float


@dataclass
class SeasonalFieldInformation(DataVibe):
    crop_name: str
    crop_type: str
    properties: Dict[str, Any]
    fertilizers: List[FertilizerInformation]
    harvests: List[HarvestInformation]
    tillages: List[TillageInformation]
    organic_amendments: List[OrganicAmendmentInformation]
