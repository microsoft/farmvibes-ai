# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from math import isclose
from typing import Dict, List, cast

import pytest
from shapely import geometry as shpg

from vibe_core.data import GHGFlux, GHGProtocolVibe
from vibe_dev.testing.op_tester import OpTester

YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_ghg_fluxes.yaml")


@pytest.fixture
def fake_ghg() -> GHGProtocolVibe:
    return GHGProtocolVibe(
        id="fake_id",
        time_range=(datetime(2020, 1, 1), datetime(2021, 1, 1)),
        geometry=shpg.mapping(shpg.box(-43.793839, -20.668953, -43.784183, -20.657266)),
        assets=[],
        cultivation_area=10,
        total_yield=50,  # average = 5
        soil_texture_class="sand",
        soil_clay_content=0.1,
        previous_land_use="native",
        current_land_use="conventional_crops",
        practice_adoption_period=10,
        burn_area=4,
        soil_management_area=2,
        synthetic_fertilizer_amount=100,
        synthetic_fertilizer_nitrogen_ratio=10 / 100.0,
        urea_amount=3,
        limestone_calcite_amount=11,
        limestone_dolomite_amount=22,
        gypsum_amount=33,
        organic_compound_amount=44,
        manure_amount=55,
        manure_birds_amount=66,
        organic_other_amount=77,
        diesel_amount=10,
        gasoline_amount=666,
        ethanol_amount=42,
        biome="BRAZIL_AMAZON_SAVANNA",
        transport_diesel_type="DIESEL_B10",
        transport_diesel_amount=790,
        green_manure_amount=22,
        green_manure_grass_amount=33,
        green_manure_legumes_amount=44,
    )


def test_ghg_fluxes(fake_ghg: GHGProtocolVibe):
    op_tester = OpTester(YAML_PATH)
    parameters = {"crop_type": "cotton"}
    op_tester.update_parameters(parameters)

    output = cast(Dict[str, List[GHGFlux]], op_tester.run(ghg=fake_ghg))
    assert output

    fluxes = {e.description: e.value for e in output["fluxes"]}

    assert isclose(fluxes["Fertilizer emissions, urea"], 0.06, abs_tol=0.01), fluxes[
        "Fertilizer emissions, urea"
    ]

    gypsum = [v for k, v in fluxes.items() if ", gypsum" in k][0]  # type: ignore
    assert isclose(gypsum, 0.29, abs_tol=0.01), gypsum

    assert isclose(
        fluxes["Fertilizer emissions, synthetic nitrogen fertilizer"], 0.34, abs_tol=0.01
    ), fluxes["Fertilizer emissions, synthetic nitrogen fertilizer"]

    s = "Fertilizer emissions, manure"
    f = [v for k, v in fluxes.items() if s in k][0]  # type: ignore
    assert isclose(f, 0.18, abs_tol=0.01), (s, f)

    flow = [v for k, v in fluxes.items() if "Flow emissions" in k][0]  # type: ignore
    assert isclose(flow, 0.17, abs_tol=0.001), flow

    atmospheric = [v for k, v in fluxes.items() if "Atmospheric emissions" in k][0]  # type: ignore
    assert isclose(atmospheric, 0.098, abs_tol=0.001), atmospheric

    residue = [v for k, v in fluxes.items() if "Residue decomposition" in k][0]  # type: ignore
    assert isclose(residue, 5.4672, abs_tol=0.001), residue

    assert isclose(fluxes["Soil management"], 146.67, abs_tol=0.1), fluxes["Soil management"]

    s = "Internal operations"
    internal = [v for k, v in fluxes.items() if s in k][0]  # type: ignore
    assert isclose(internal, 1.3027, abs_tol=0.001), (s, internal)

    s = "Initial carbon stock"
    assert isclose(fluxes[s], 863.76, abs_tol=1), fluxes[s]

    s = "Transportation / DIESEL_B10 / Biodiesel"
    assert isclose(fluxes[s], 2.1131, abs_tol=0.01), fluxes[s]

    s = "Biomass Burning (Cotton)"
    assert isclose(fluxes[s], 81.58, abs_tol=0.1), fluxes[s]

    s = "Carbon captured by Green Manure"
    assert isclose(fluxes[s], -18.35, abs_tol=0.1), fluxes[s]

    s = "Land use change"
    assert isclose(fluxes[s], 9.167, abs_tol=0.1), fluxes[s]
