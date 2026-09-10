# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
import yaml
from pyngrok.exception import PyngrokError
from whatif_comet_local import SeasonalFieldConverter

from vibe_core.data import CarbonOffsetInfo, SeasonalFieldInformation
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.comet_farm.comet_server import CometHTTPServer, CometServerParameters


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
            "organic_amendments": [
                {
                    "start_date": "2000-03-01T00:00:00Z",
                    "end_date": "2000-03-01T00:00:00Z",
                    "organic_amendment_type": "Soybean Meal",
                    "organic_amendment_amount": 1.0,
                    "organic_amendment_percent_nitrogen": 7.0,
                    "organic_amendment_carbon_nitrogen_ratio": 4.0,
                }
            ],
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
                            "@name": "Baseline",
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
    fake_comet_response: Dict[str, Any],
):
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "whatif_comet_local_op.yaml"
    )
    scenario = deepcopy(fake_comet_response["Day"]["Cropland"]["ModelRun"]["Scenario"][0])
    scenario["@name"] = "Scenario: 17/03/2023 16:00:01"
    scenario["Carbon"]["SoilCarbon"] = "4321"
    fake_comet_response["Day"]["Cropland"]["ModelRun"]["Scenario"].insert(0, scenario)
    parse_comet_response.return_value = fake_comet_response
    parameters = {
        "comet_support_email": "fake_email",
        "comet_api_key": "fake_api_key",
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
    assert output_data["carbon_output"].carbon == "4321 Mg Co2e/year"


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
        "comet_api_key": "fake_api_key",
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
        "comet_api_key": "fake_api_key",
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


def test_comet_v25_request(
    baseline_information: List[SeasonalFieldInformation],
    scenario_information: List[SeasonalFieldInformation],
):
    converter = SeasonalFieldConverter()
    root = ET.fromstring(
        converter.build_comet_request(baseline_information, scenario_information)
    )

    assert root.tag == "CometFarm"
    assert root.find("Project/ActivityYears") is not None
    cropland = root.find("Project/Cropland")
    assert cropland is not None
    assert cropland.findtext("Pre-1980") == "Cropland Lowland Non-Irrigated (Pre 1980s)"
    assert cropland.findtext("CRPStartYear") == "0"
    assert cropland.findtext("CRPEndYear") == "0"
    assert cropland.findtext("Year1980-2000_Tillage") == "Full Intensive Till (III)"
    scenarios = cropland.findall("CropScenario")
    assert [scenario.attrib["Name"] for scenario in scenarios[:1]] == ["Current"]
    assert len(scenarios) == 2
    assert scenarios[0].findtext("CropYear/Crop/CropType") == "CROPS"
    assert scenarios[0].findtext("CropYear/Crop/HarvestList/HarvestEvent/Grain") == "True"
    assert (
        scenarios[0].findtext("CropYear/Crop/TillageList/TillageEvent/TillageType")
        == "Reduced Till (V)"
    )
    assert (
        scenarios[1].findtext("CropYear/Crop/TillageList/TillageEvent/TillageType")
        == "No Till (III)"
    )
    assert scenarios[0].find("CropYear/Crop/BioCharApplicationList") is not None
    omad = scenarios[0].find("CropYear/Crop/OMADApplicationList/OMADApplicationEvent")
    assert omad is not None
    assert omad.findtext("OMADType") == "Soybean Meal"
    assert omad.findtext("OMADApplicationMethod") == "Surface"


def test_comet_v25_bumps_operation_version():
    with open(
        os.path.join(os.path.dirname(__file__), "whatif_comet_local_op.yaml")
    ) as config:
        assert yaml.safe_load(config)["version"] == 3


@patch("vibe_lib.comet_farm.comet_server.requests.request")
def test_submit_job_uses_api_key_without_logging_it(
    request: Mock, caplog: pytest.LogCaptureFixture
):
    response = Mock()
    request.return_value = response
    server = Mock()
    server.logger = logging.getLogger("test.comet")
    server.comet_request = CometServerParameters(
        url="https://comet-farm.com/apimain/uploadapiqueue",
        webhook="https://callback.example",
        supportEmail="user@example.com",
        apiKey="secret-api-key",
        ngrokToken="fake-ngrok-token",
    )

    with caplog.at_level(logging.INFO):
        CometHTTPServer.submit_job(server, "<CometFarm />", "request-id")

    response.raise_for_status.assert_called_once()
    _, url = request.call_args.args
    assert url == "https://comet-farm.com/apimain/uploadapiqueue"
    payload = request.call_args.kwargs["data"]
    assert payload == {
        "email": "user@example.com",
        "url": "https://callback.example/request-id",
        "apikey": "secret-api-key",
    }
    files = request.call_args.kwargs["files"]
    assert files["Files"][0] == "file.xml"
    assert files["Files"][1].getvalue() == "<CometFarm />"
    assert "secret-api-key" not in caplog.text
