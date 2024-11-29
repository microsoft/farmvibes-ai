# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class MapUnit(BaseModel):
    id: str = Field(alias="@id")
    area: Union[None, str] = Field(alias="@area")
    year: Union[None, str] = Field(alias="Year")
    inputCrop: Union[None, str] = Field(alias="InputCrop")
    irrigated: Union[None, str] = Field(alias="Irrigated")
    agcprd: Union[None, str]
    abgdefac: Union[None, str]
    accrste_1_: Union[None, str]
    crpval: Union[None, str]
    rain: Union[None, str]
    cgrain: Union[None, str]
    cinput: Union[None, str]
    eupacc_1_: Union[None, str]
    fertot_1_1_: Union[None, str]
    fertac_1_: Union[None, str]
    irrtot: Union[None, str]
    metabe_1_1_: Union[None, str]
    metabe_2_1_: Union[None, str]
    nfixac: Union[None, str]
    omadae_1_: Union[None, str]
    petann: Union[None, str]
    stdede_1_: Union[None, str]
    struce_1_1_: Union[None, str]
    struce_2_1_: Union[None, str]
    tnetmn_1_: Union[None, str]
    tminrl_1_: Union[None, str]
    gromin_1_: Union[None, str]
    somse_1_: Union[None, str]
    somsc: Union[None, str]
    strmac_2_: Union[None, str]
    volpac: Union[None, str]
    aagdefac: Union[None, str]
    accrst: Union[None, str]
    aglivc: Union[None, str]
    bgdefac: Union[None, str]
    bglivcm: Union[None, str]
    crmvst: Union[None, str]
    crootc: Union[None, str]
    fbrchc: Union[None, str]
    frootcm: Union[None, str]
    metabc_1_: Union[None, str]
    metabc_2_: Union[None, str]
    omadac: Union[None, str]
    rlwodc: Union[None, str]
    stdedc: Union[None, str]
    strmac_1_: Union[None, str]
    strmac_6_: Union[None, str]
    strucc_1_: Union[None, str]
    n2oflux: Union[None, str]
    annppt: Union[None, str]
    noflux: Union[None, str]

    class Config:
        allow_population_by_field_name = True


class CarbonResponse(BaseModel):
    soilCarbon: str = Field(alias="SoilCarbon")
    biomassBurningCarbon: str = Field(alias="BiomassBurningCarbon")
    soilCarbonStock2000: str = Field(alias="SoilCarbonStock2000")
    soilCarbonStockBegin: str = Field(alias="SoilCarbonStockBegin")
    soilCarbonStockEnd: str = Field(alias="SoilCarbonStockEnd")

    class Config:
        allow_population_by_field_name = True


class Co2Response(BaseModel):
    limingCO2: str = Field(alias="LimingCO2")
    ureaFertilizationCO2: str = Field(alias="UreaFertilizationCO2")
    drainedOrganicSoilsCO2: str = Field(alias="DrainedOrganicSoilsCO2")

    class Config:
        allow_population_by_field_name = True


class N2OResponse(BaseModel):
    soilN2O: str = Field(alias="SoilN2O")
    soilN2O_Direct: str = Field(alias="SoilN2O_Direct")
    soilN2O_Indirect_Volatilization: str = Field(alias="SoilN2O_Indirect_Volatilization")
    soilN2O_Indirect_Leaching: str = Field(alias="SoilN2O_Indirect_Leaching")
    wetlandRiceCultivationN2O: str = Field(alias="WetlandRiceCultivationN2O")
    biomassBurningN2O: str = Field(alias="BiomassBurningN2O")
    drainedOrganicSoilsN2O: str = Field(alias="DrainedOrganicSoilsN2O")

    class Config:
        allow_population_by_field_name = True


class CH4Response(BaseModel):
    soilCH4: str = Field(alias="SoilCH4")
    wetlandRiceCultivationCH4: str = Field(alias="WetlandRiceCultivationCH4")
    biomassBurningCH4: str = Field(alias="BiomassBurningCH4")

    class Config:
        allow_population_by_field_name = True


class CometOutput(BaseModel):
    name: str = Field(alias="@name")
    carbon: CarbonResponse = Field(alias="Carbon")
    co2: Co2Response = Field(alias="CO2")
    n20: N2OResponse = Field(alias="N2O")
    ch4: CH4Response = Field(alias="CH4")

    class Config:
        allow_population_by_field_name = True


class ScenarioMapUnit(BaseModel):
    name: str = Field(alias="@name")
    mapUnit: Union[List[MapUnit], MapUnit] = Field(alias="MapUnit")

    class Config:
        allow_population_by_field_name = True


class ModelRunChild(BaseModel):
    name: str = Field(alias="@name")
    scenario: List[Union[ScenarioMapUnit, CometOutput]] = Field(alias="Scenario")

    class Config:
        allow_population_by_field_name = True


class ModelRun(BaseModel):
    modelRun: ModelRunChild = Field(alias="ModelRun")

    class Config:
        allow_population_by_field_name = True


class CometDay(BaseModel):
    cometEmailID: str = Field(alias="@cometEmailId")
    cFARMVersion: str = Field(alias="@CFARMVersion")
    cropland: ModelRun = Field(alias="Cropland")

    class Config:
        allow_population_by_field_name = True


class CometResponse(BaseModel):
    day: CometDay = Field(alias="Day")

    class Config:
        allow_population_by_field_name = True


class CarbonOffset(BaseModel):
    id: str
    data: Dict[str, Any]
