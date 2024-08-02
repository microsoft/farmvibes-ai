# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from typing import List
from unittest.mock import Mock, patch

import pytest
from pyngrok.exception import PyngrokError

from vibe_core.data import CarbonOffsetInfo, SeasonalFieldInformation
from vibe_dev.testing.op_tester import OpTester


@pytest.fixture
def baseline_information():
    field_info = [
        {
            "id": "25e96fa0-9cf8-4b31-ac9e-24e30c37aeaf",
            "time_range": [
                datetime(year=2020, month=2, day=15),
                datetime(year=2023, month=9, day=15),
            ],
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-87.414918, 37.463251],
                        [-87.399025, 37.470226],
                        [-87.393499, 37.472569],
                        [-87.39827, 37.479898],
                        [-87.405993, 37.478046],
                        [-87.407538, 37.47761],
                        [-87.408122, 37.477501],
                        [-87.408636, 37.477092],
                        [-87.409048, 37.476602],
                        [-87.414918, 37.463251],
                    ]
                ],
            },
            "assets": [],
            "crop_name": "Alfalfa",
            "crop_type": "annual",
            "properties": {
                "pre_1980": "Lowland Non-Irrigated (Pre 1980s)",
                "crp_type": "None",
                "crp_start": "",
                "crp_end": "",
                "year_1980_2000": "Irrigated: Continuous Hay",
                "year_1980_2000_tillage": "Intensive Tillage",
            },
            "fertilizers": [],
            "harvests": [
                {
                    "is_grain": True,
                    "start_date": "2000-09-05T00:00:00Z",
                    "end_date": "2000-09-05T00:00:00Z",
                    "crop_yield": 39.0,
                    "stray_stover_hay_removal": "0",
                },
            ],
            "tillages": [
                {
                    "start_date": "2000-01-01T00:00:00Z",
                    "end_date": "2000-01-01T00:00:00Z",
                    "implement": "Reduced Tillage",
                }
            ],
            "organic_amendments": [],
        }
    ]

    fi = [SeasonalFieldInformation(**item) for item in field_info]
    return fi


@pytest.fixture
def scenario_information():
    field_info = [
        {
            "id": "0e16be1a-eb0f-4b55-a69c-4fa79af8f406",
            "time_range": [
                datetime(year=2023, month=2, day=15),
                datetime(year=2025, month=9, day=15),
            ],
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-87.414918, 37.463251],
                        [-87.399025, 37.470226],
                        [-87.393499, 37.472569],
                        [-87.39827, 37.479898],
                        [-87.405993, 37.478046],
                        [-87.407538, 37.47761],
                        [-87.408122, 37.477501],
                        [-87.408636, 37.477092],
                        [-87.409048, 37.476602],
                        [-87.414918, 37.463251],
                    ]
                ],
            },
            "assets": [],
            "crop_name": "Barley",
            "crop_type": "annual",
            "properties": {
                "pre_1980": "Lowland Non-Irrigated (Pre 1980s)",
                "crp_type": "None",
                "crp_start": "",
                "crp_end": "",
                "year_1980_2000": "Irrigated: Continuous Hay",
                "year_1980_2000_tillage": "Intensive Tillage",
            },
            "fertilizers": [],
            "harvests": [
                {
                    "is_grain": True,
                    "start_date": "2023-11-11T00:00:00Z",
                    "end_date": "2023-11-11T00:00:00Z",
                    "crop_yield": 30.0,
                    "stray_stover_hay_removal": "0",
                }
            ],
            "tillages": [
                {
                    "start_date": "2023-01-01T00:00:00Z",
                    "end_date": "2023-01-01T00:00:00Z",
                    "implement": "Zero Soil Disturbance",
                }
            ],
            "organic_amendments": [],
        }
    ]

    fi = [SeasonalFieldInformation(**item) for item in field_info]
    return fi


@pytest.fixture
def fake_comet_error():
    return {
        "Errors": {
            "ModelRun": {
                "@name": "sdk_int1",
                "Error": {
                    "@index": "0",
                    "@message": "You entered 200 in tag OMADPercentN for "
                    "CropYear: 2000 and CropScenario: Current "
                    ".Percent Nitrogen needs to between 0 and 100",
                },
            }
        }
    }


@pytest.fixture
def fake_comet_response():
    return {
        "Day": {
            "@cometEmailId": "fake-email",
            "@CFARMVersion": "appengine cometfarm v0-10  build 3.2.8472.37261 (03/13/2023)",
            "Cropland": {
                "ModelRun": {
                    "@name": "sdk_int1",
                    "Scenario": [
                        {
                            "@name": "scenario: 17/03/2023 16:00:01",
                            "Carbon": {
                                "SoilCarbon": "1234.4321",
                                "BiomassBurningCarbon": "0",
                                "SoilCarbonStock2000": "1234.4321",
                                "SoilCarbonStockBegin": "1234.4321",
                                "SoilCarbonStockEnd": "1234.4321",
                            },
                            "CO2": {
                                "LimingCO2": "0",
                                "UreaFertilizationCO2": "0",
                                "DrainedOrganicSoilsCO2": "0",
                            },
                            "N2O": {
                                "SoilN2O": "1234.4321",
                                "SoilN2O_Direct": "1234.4321",
                                "SoilN2O_Indirect_Volatilization": "1234.4321",
                                "SoilN2O_Indirect_Leaching": "1234.4321",
                                "WetlandRiceCultivationN2O": "0",
                                "BiomassBurningN2O": "0",
                                "DrainedOrganicSoilsN2O": "0",
                            },
                            "CH4": {
                                "SoilCH4": "0",
                                "WetlandRiceCultivationCH4": "0",
                                "BiomassBurningCH4": "0",
                            },
                        }
                    ],
                }
            },
        }
    }


@patch("http.server.HTTPServer.server_bind")
@patch("vibe_lib.comet_farm.comet_server.CometHTTPServer.start_ngrok")
@patch("vibe_lib.comet_farm.comet_server.CometHTTPServer.start")
@patch("vibe_lib.comet_farm.comet_server.CometHTTPServer.shutdown")
@patch("vibe_lib.comet_farm.comet_requester.CometRequester.get_comet_raw_output")
@patch("vibe_lib.comet_farm.comet_requester.CometRequester.parse_comet_response")
def test_whatif_request(
    parse_comet_response: Mock,
    _: Mock,
    __: Mock,
    ___: Mock,
    ____: Mock,
    _____: Mock,
    baseline_information: List[SeasonalFieldInformation],
    scenario_information: List[SeasonalFieldInformation],
    fake_comet_response: str,
):
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "whatif_comet_local_op.yaml"
    )
    parse_comet_response.return_value = fake_comet_response
    parameters = {
        "comet_support_email": "fake_email",
        "ngrok_token": "fake_ngrok",
    }

    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)

    output_data = op_.run(
        # pyright misidentifies types here
        baseline_seasonal_fields=baseline_information,  # type: ignore
        scenario_seasonal_fields=scenario_information,  # type: ignore
    )

    assert "carbon_output" in output_data
    assert isinstance(output_data["carbon_output"], CarbonOffsetInfo)
    assert "Mg Co2e/year" in output_data["carbon_output"].carbon


@patch("http.server.HTTPServer.server_bind")
@patch("vibe_lib.comet_farm.comet_server.CometHTTPServer.start_ngrok")
@patch("vibe_lib.comet_farm.comet_server.CometHTTPServer.start")
@patch("vibe_lib.comet_farm.comet_requester.CometRequester.get_comet_raw_output")
@patch("vibe_lib.comet_farm.comet_requester.CometRequester.parse_comet_response")
def test_whatif_request_comet_error(
    parse_comet_response: Mock,
    _: Mock,
    __: Mock,
    ___: Mock,
    ____: Mock,
    baseline_information: List[SeasonalFieldInformation],
    scenario_information: List[SeasonalFieldInformation],
    fake_comet_error: str,
):
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "whatif_comet_local_op.yaml"
    )
    parse_comet_response.return_value = fake_comet_error
    parameters = {
        "comet_support_email": "fake_email",
        "ngrok_token": "fake_ngrok",
    }

    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)

    with pytest.raises(RuntimeError):
        op_.run(
            # pyright misidentifies types here
            baseline_seasonal_fields=baseline_information,  # type: ignore
            scenario_seasonal_fields=scenario_information,  # type: ignore
        )


@patch("pyngrok.ngrok.set_auth_token")
def test_whatif_start_ngrok_error(
    set_auth_token: Mock,
    baseline_information: List[SeasonalFieldInformation],
    scenario_information: List[SeasonalFieldInformation],
):
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "whatif_comet_local_op.yaml"
    )
    set_auth_token.side_effect = PyngrokError("Fake Error")
    parameters = {
        "comet_support_email": "fake_email",
        "ngrok_token": "fake_ngrok",
    }

    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)

    with pytest.raises(Exception):
        op_.run(
            # pyright misidentifies types here
            baseline_seasonal_fields=baseline_information,  # type: ignore
            scenario_seasonal_fields=scenario_information,  # type: ignore
        )
