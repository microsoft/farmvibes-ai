from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .core_types import BaseVibe, DataVibe


class Name(BaseModel):
    id: int
    name: str


class Fertilizer(BaseModel):
    ammonium: int
    date: str
    eep: Name
    fertilizerType: Name
    id: str
    totalFertilizerApplied: str
    totalNitrogenApplied: str


class Omad(BaseModel):
    date: str
    id: str
    type: str
    amount: str
    percentN: str
    CNratio: str


class CHarvest(BaseModel):
    gfsrt: bool
    harvestDate: str
    id: str
    sshrr: str
    Yield: str = Field(alias="yield")

    class Config:
        allow_population_by_field_name = True


class CTillage(BaseModel):
    date: str
    id: str
    implement: Name


class Crop(BaseModel):
    id: Optional[str]
    name: str
    plantedDate: str
    type: str
    fertilizer: List[Fertilizer]
    harvest: List[CHarvest]
    tillage: List[CTillage]
    omad: Optional[List[Omad]]


@dataclass
class CBoundary:
    name: str
    location: Dict[str, Any]


@dataclass
class Historical:
    pre_1980: str
    crp_type: str
    crp_start: str
    crp_end: str
    year_1980_2000: str
    year_1980_2000_tillage: str


@dataclass
class Scenario:
    id: str
    farmId: str
    name: str
    history: Dict[str, Dict[str, Any]]
    container: str


@dataclass
class Baseline:
    id: str
    farmId: str
    name: str
    history: Dict[str, Dict[str, Any]]
    container: str


@dataclass
class FarmersInfo(BaseVibe):
    ids: List[str]


@dataclass
class BoundariesInfo(DataVibe):
    details: Dict[str, Any]


@dataclass
class WhatIfScenario(BaseVibe):
    scenarioName: str
    farmerId: str
    boundaryId: str
    year: str
    scenario: Crop


@dataclass
class CarbonOffsetInfo(DataVibe):
    carbon: str


@dataclass
class CarbonScenario(BaseVibe):
    boundary: CBoundary
    scenario: Scenario
    baseline: Baseline
    historical: Historical
