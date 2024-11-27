# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import json
import os
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, Mock, patch

import geopandas as gpd
import pytest
from shapely import geometry as shpg

from vibe_core.admag_client import ADMAgClient
from vibe_core.data import (
    ADMAgPrescription,
    ADMAgPrescriptionInput,
    ADMAgSeasonalFieldInput,
    AssetVibe,
)
from vibe_dev.mock_utils import Request
from vibe_dev.testing.op_tester import OpTester

HERE = os.path.dirname(os.path.abspath(__file__))
ADMAG_SEASONAL_FIELD_OP = os.path.join(HERE, "admag_seasonal_field.yaml")


@pytest.fixture
@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
def admag_client(get_token: MagicMock):
    return ADMAgClient(
        base_url="fake_url",
        api_version="fake_admag_version",
        client_id="fake_client_id",
        client_secret="fake_client_secret",
        authority="fake_authority",
        default_scope="fake_scope",
    )


@pytest.fixture
def fake_get_response_without_next_link() -> Dict[str, Any]:
    return {
        "value": [
            {
                "fake_key": "fake_value",
            },
        ],
    }


@pytest.fixture
def fake_get_response_with_next_link() -> Dict[str, Any]:
    return {
        "value": [
            {
                "fake_key": "fake_value",
            },
        ],
        "nextLink": "http://fake-url",
    }


@pytest.fixture
def fake_input_data() -> ADMAgSeasonalFieldInput:
    return ADMAgSeasonalFieldInput(
        party_id="fake-party-id",
        seasonal_field_id="fake-seasonal-field-id",
    )


@pytest.fixture
def fake_prescription_input_data() -> ADMAgPrescriptionInput:
    return ADMAgPrescriptionInput(
        party_id="fake-party-id",
        prescription_id="fake-prescription-id",
    )


@patch.object(ADMAgClient, "_request")
def test_admag_client_get_limit_requests(
    _request: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    admag_client: ADMAgClient,
    fake_get_response_with_next_link: Dict[str, Any],
    fake_get_response_without_next_link: Dict[str, Any],
):
    fake_response_different_link = fake_get_response_with_next_link.copy()
    fake_response_different_link.update({"nextLink": "different_fake_link"})
    fake_response_another_link = fake_get_response_with_next_link.copy()
    fake_response_another_link.update({"nextLink": "another_fake_link"})

    monkeypatch.setattr(ADMAgClient, "NEXT_PAGES_LIMIT", 1)
    _request.side_effect = [
        fake_get_response_with_next_link,
        fake_response_different_link,
        fake_get_response_without_next_link,
    ]

    with pytest.raises(RuntimeError):
        admag_client._get("fake_url")


@patch.object(ADMAgClient, "_request")
def test_admag_client_get_repeated_link(
    _request: MagicMock,
    admag_client: ADMAgClient,
    fake_get_response_with_next_link: Dict[str, Any],
    fake_get_response_without_next_link: Dict[str, Any],
):
    _request.side_effect = [
        fake_get_response_with_next_link,
        fake_get_response_with_next_link,
        fake_get_response_without_next_link,
    ]

    with pytest.raises(RuntimeError):
        admag_client._get("fake_url")


@patch.object(ADMAgClient, "_request")
def test_admag_client_get_follow_link(
    _request: MagicMock,
    admag_client: ADMAgClient,
    fake_get_response_with_next_link: Dict[str, Any],
    fake_get_response_without_next_link: Dict[str, Any],
):
    fake_response_different_link = fake_get_response_with_next_link.copy()
    fake_response_different_link.update({"nextLink": "different_fake_link"})
    _request.side_effect = [
        fake_get_response_with_next_link,
        fake_response_different_link,
        fake_get_response_without_next_link,
    ]

    result = admag_client._get("fake_url")
    assert len(result["value"]) == 3


def test_admag_client_creation(admag_client: ADMAgClient):
    assert admag_client.header() == {
        "Authorization": "Bearer my_fake_token",
        "Content-Type": "application/merge-patch+json",
    }


@pytest.fixture
def seasonal_field_info(vibe_geometry_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "partyId": "fake-party-id",
        "farmId": "fake-farm-id",
        "fieldId": "fake-field-id",
        "seasonId": "fake-season-id",
        "cropId": "fake-crop-id",
        "id": "fake-seasonal-field-id",
        "eTag": "fake-etag",
        "status": "Active",
        "createdDateTime": "2001-01-01T00:00:00Z",
        "modifiedDateTime": "2001-01-01T00:00:00Z",
        "name": "fake-seasonal-field-name",
        "description": "fake-description",
        "geometry": vibe_geometry_dict,
        "properties": {
            "plantingDateTime": "2001-01-01T00:00:00Z",
        },
    }


@patch("vibe_core.admag_client.ADMAgClient._get")
def test_get_seasonal_field(
    _get: MagicMock, seasonal_field_info: Dict[str, Any], admag_client: ADMAgClient
):
    _get.return_value = seasonal_field_info
    seasonal_field_result = admag_client.get_seasonal_field(
        party_id="fake-party-id",
        seasonal_field_id="fake-seasonal-field-id",
    )
    assert seasonal_field_result
    assert "name" in seasonal_field_result
    assert "description" in seasonal_field_result
    assert "geometry" in seasonal_field_result


@pytest.fixture
def season_info() -> Dict[str, Any]:
    return {
        "startDateTime": "2001-01-01T00:00:00Z",
        "endDateTime": "2001-12-31T00:00:00Z",
        "year": 2001,
        "id": "fake-season-id",
        "eTag": "fake-etag",
        "status": "Active",
        "createdDateTime": "2001-01-01T00:00:00Z",
        "modifiedDateTime": "2001-01-01T00:00:00Z",
        "name": "fake-season-name",
    }


@patch("vibe_core.admag_client.ADMAgClient._get")
def test_get_season(_get: MagicMock, season_info: Dict[str, Any], admag_client: ADMAgClient):
    _get.return_value = season_info
    season_result = admag_client.get_season(
        season_id="fake-season-id",
    )
    assert season_result
    assert "startDateTime" in season_result
    assert "endDateTime" in season_result
    assert "year" in season_result


@pytest.fixture
def field_info(vibe_geometry_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "partyId": "fake-party-id",
        "farmId": "fake-farm-id",
        "geometry": vibe_geometry_dict,
        "eTag": "fake-etag",
        "id": "fake-field-id",
        "status": "Active",
        "createdDateTime": "2001-01-01T00:00:00Z",
        "modifiedDateTime": "2001-01-01T00:00:00Z",
        "name": "fake-field-name",
        "description": "Fake description",
        "properties": {
            "pre_1980": "Lowland Non-Irrigate...Pre 1980s)",
            "crp_type": "None",
            "crp_start": "",
            "crp_end": "",
            "year_1980_2000": "Irrigated: Continuous Hay",
            "year_1980_2000_tillage": "Intensive Tillage",
        },
    }


@pytest.fixture
def prescription_geom_input() -> List[ADMAgPrescription]:
    prescription = {
        "partyId": "ae880a1b-4597-46d7-83ac-bfc6a1ae4116-16",
        "prescriptionMapId": "831989c4-c15a-4fc5-837b-4c0289d53010",
        "productCode": "1635",
        "productName": "Nutrient",
        "type": "Nutrient",
        "measurements": {
            "N": {"value": 47.1},
            "P": {"value": 34.99769206227461},
            "pH": {"value": 4.978131831743143},
            "C": {"value": 0.046408031802193},
        },
        "id": "880094d0-1c48-4d7c-b0d3-f7477a937473",
        "eTag": "24009696-0000-0100-0000-65fb20540000",
        "status": "Active",
        "createdDateTime": "2024-03-20T17:43:48Z",
        "modifiedDateTime": "2024-03-20T17:43:48Z",
        "source": "IOT device",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-117.03642546099948, 47.044663835752566],
                    [-117.05642546099949, 47.044663835752566],
                    [-117.05642546099949, 47.02466383575257],
                    [-117.03642546099948, 47.02466383575257],
                    [-117.03642546099948, 47.044663835752566],
                ]
            ],
        },
        "name": "Nitrogen Nutrient",
        "description": "",
        "createdBy": "f8c6c349-b484-4863-af76-d10eee669306",
        "modifiedBy": "f8c6c349-b484-4863-af76-d10eee669306",
    }

    return [ADMAgPrescription(**prescription)]


@patch("vibe_core.admag_client.ADMAgClient._get")
def test_get_field(_get: MagicMock, field_info: Dict[str, Any], admag_client: ADMAgClient):
    _get.return_value = field_info
    field_result = admag_client.get_field(
        party_id="fake-party-id",
        field_id="fake-field-id",
    )
    assert field_result
    assert "properties" in field_result
    properties = field_result["properties"]
    assert "pre_1980" in properties
    assert "crp_type" in properties
    assert "crp_start" in properties
    assert "crp_end" in properties
    assert "year_1980_2000" in properties
    assert "year_1980_2000_tillage" in properties


@pytest.fixture
def harvest_result(vibe_geometry_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": [
            {
                "geometry": vibe_geometry_dict,
                "attachmentsLink": "https://fake-attachment.bla",
                "createdDateTime": "2021-12-10T00:18:33Z",
                "eTag": "5500c45e-0000-0100-0000-61b29cd90000",
                "partyId": "fake-party-id",
                "id": "fake-harvest-id",
                "modifiedDateTime": "2021-12-10T00:18:33Z",
                "operationEndDateTime": "2001-09-05T00:00:00Z",
                "operationStartDateTime": "2001-09-05T00:00:00Z",
                "properties": {"gfsrt": "True", "strawStoverHayRemoval": "0"},
                "source": "Farming",
                "status": "Active",
                "totalYield": {"unit": "tons", "value": 39.0},
            },
        ]
    }


@pytest.fixture
def planting_result(vibe_geometry_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": [
            {
                "partyId": "fake-party-id",
                "id": "fake-id",
                "source": "Manual",
                "name": "Planting data for North Farm",
                "description": "some description",
                "status": "Active",
                "operationStartDateTime": "2021-02-25T16:57:04Z",
                "operationEndDateTime": "2021-02-27T10:13:06Z",
                "operationModifiedDateTime": "2021-02-28T10:14:12Z",
                "avgPlantingRate": {"unit": "seedsperacre", "value": 30},
                "area": {"unit": "acre", "value": 30},
                "totalMaterial": {"unit": "seeds", "value": 758814},
                "avgMaterial": {"unit": "seedsperacre", "value": 25293},
                "plantingProductDetails": [
                    {
                        "productName": "VAR1",
                        "area": {"unit": "acre", "value": 20},
                        "totalMaterial": {"unit": "seeds", "value": 389214},
                        "avgMaterial": {"unit": "seedsperacre", "value": 19460},
                    }
                ],
                "properties": {"Region": "Europe", "CountyCode": 123},
                "createdDateTime": "2022-05-11T07:00:10.2750191Z",
                "modifiedDateTime": "2022-05-11T07:00:10.2750191Z",
                "eTag": "cb00a3ac-0000-0100-0000-601d21ec0000",
            },
        ]
    }


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("vibe_core.admag_client.ADMAgClient._post")
def test_get_harvest_info(
    _post: MagicMock,
    get_token: MagicMock,
    harvest_result: Dict[str, Any],
    admag_client: ADMAgClient,
    vibe_geometry_dict: Dict[str, Any],
):
    _post.return_value = harvest_result
    harvest_result = admag_client.get_harvest_info(
        party_id="fake-party-id",
        intersects_with_geometry=vibe_geometry_dict,
        min_start_operation="2001-01-01T00:00:00Z",
        max_end_operation="2001-01-01T00:00:00Z",
        associated_resource={"type": "SeasonalField", "id": "fake-seasonal-field-id"},
    )
    assert "value" in harvest_result
    harvest_list = harvest_result["value"]
    assert len(harvest_result) > 0
    harvest_entry = harvest_list[0]
    assert "operationStartDateTime" in harvest_entry
    assert "operationEndDateTime" in harvest_entry
    assert "properties" in harvest_entry
    harvest_properties = harvest_entry["properties"]
    assert "gfsrt" in harvest_properties
    assert "strawStoverHayRemoval" in harvest_properties
    assert "totalYield" in harvest_entry
    harvest_yield = harvest_entry["totalYield"]
    assert "value" in harvest_yield


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("vibe_core.admag_client.ADMAgClient.get_field")
@patch("vibe_core.admag_client.ADMAgClient.get_seasonal_field")
@patch("vibe_core.admag_client.ADMAgClient.get_season")
@patch("vibe_core.admag_client.ADMAgClient.get_harvest_info")
@patch("vibe_core.admag_client.ADMAgClient.get_fertilizer_info")
@patch("vibe_core.admag_client.ADMAgClient.get_tillage_info")
@patch("vibe_core.admag_client.ADMAgClient.get_organic_amendments_info")
def test_admag_incomplete_fertilizer(
    get_organic_amendments_info: MagicMock,
    get_tillage_info: MagicMock,
    get_fertilizer_info: MagicMock,
    get_harvest_info: MagicMock,
    get_season: MagicMock,
    get_seasonal_field: MagicMock,
    get_field: MagicMock,
    get_token: MagicMock,
    seasonal_field_info: Dict[str, Any],
    field_info: Dict[str, Any],
    season_info: Dict[str, Any],
    harvest_result: Dict[str, Any],
    fertilizer_result: Dict[str, Any],
    tillage_result: Dict[str, Any],
    omad_result: Dict[str, Any],
    fake_input_data: ADMAgSeasonalFieldInput,
):
    get_seasonal_field.return_value = seasonal_field_info
    get_field.return_value = field_info
    get_season.return_value = season_info
    get_harvest_info.return_value = harvest_result
    get_tillage_info.return_value = tillage_result
    get_organic_amendments_info.return_value = omad_result

    fertilizer_missing_total_N = copy.deepcopy(fertilizer_result)
    fertilizer_missing_total_N["value"][0]["properties"].pop("totalNitrogen")
    get_fertilizer_info.return_value = fertilizer_missing_total_N

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)

    fertilizer_missing_eep = copy.deepcopy(fertilizer_result)
    fertilizer_missing_eep["value"][0]["properties"].pop("eep")
    get_fertilizer_info.return_value = fertilizer_missing_eep

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)

    fertilizer_wrong_eep = copy.deepcopy(fertilizer_result)
    fertilizer_wrong_eep["value"][0]["properties"]["eep"] = "fake-eep"
    get_fertilizer_info.return_value = fertilizer_wrong_eep

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("vibe_core.admag_client.ADMAgClient.get_field")
@patch("vibe_core.admag_client.ADMAgClient.get_seasonal_field")
@patch("vibe_core.admag_client.ADMAgClient.get_season")
@patch("vibe_core.admag_client.ADMAgClient.get_harvest_info")
@patch("vibe_core.admag_client.ADMAgClient.get_fertilizer_info")
@patch("vibe_core.admag_client.ADMAgClient.get_tillage_info")
@patch("vibe_core.admag_client.ADMAgClient.get_organic_amendments_info")
def test_admag_incomplete_harvest(
    get_organic_amendments_info: MagicMock,
    get_tillage_info: MagicMock,
    get_fertilizer_info: MagicMock,
    get_harvest_info: MagicMock,
    get_season: MagicMock,
    get_seasonal_field: MagicMock,
    get_field: MagicMock,
    _: MagicMock,
    seasonal_field_info: Dict[str, Any],
    field_info: Dict[str, Any],
    season_info: Dict[str, Any],
    harvest_result: Dict[str, Any],
    fertilizer_result: Dict[str, Any],
    tillage_result: Dict[str, Any],
    omad_result: Dict[str, Any],
    fake_input_data: ADMAgSeasonalFieldInput,
):
    get_seasonal_field.return_value = seasonal_field_info
    get_field.return_value = field_info
    get_season.return_value = season_info
    get_fertilizer_info.return_value = fertilizer_result
    get_tillage_info.return_value = tillage_result
    get_organic_amendments_info.return_value = omad_result

    # Don't remove code, it may required for different crop
    # harvest_missing_gfsrt = copy.deepcopy(harvest_result)
    # harvest_missing_gfsrt["value"][0]["properties"].pop("gfsrt")

    # get_harvest_info.return_value = harvest_missing_gfsrt

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)

    # Don't remove code, it may required for different crop
    # harvest_missing_straw_stover_hay_removal = copy.deepcopy(harvest_result)
    # harvest_missing_straw_stover_hay_removal["value"][0]["properties"].pop(
    #   "strawStoverHayRemoval"
    # )
    # get_harvest_info.return_value = harvest_missing_straw_stover_hay_removal

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("vibe_core.admag_client.ADMAgClient.get_field")
@patch("vibe_core.admag_client.ADMAgClient.get_seasonal_field")
@patch("vibe_core.admag_client.ADMAgClient.get_season")
@patch("vibe_core.admag_client.ADMAgClient.get_harvest_info")
@patch("vibe_core.admag_client.ADMAgClient.get_fertilizer_info")
@patch("vibe_core.admag_client.ADMAgClient.get_tillage_info")
@patch("vibe_core.admag_client.ADMAgClient.get_organic_amendments_info")
def test_admag_incomplete_organic_amendments(
    get_organic_amendments_info: MagicMock,
    get_tillage_info: MagicMock,
    get_fertilizer_info: MagicMock,
    get_harvest_info: MagicMock,
    get_season: MagicMock,
    get_seasonal_field: MagicMock,
    get_field: MagicMock,
    _: MagicMock,
    seasonal_field_info: Dict[str, Any],
    field_info: Dict[str, Any],
    season_info: Dict[str, Any],
    harvest_result: Dict[str, Any],
    fertilizer_result: Dict[str, Any],
    tillage_result: Dict[str, Any],
    omad_result: Dict[str, Any],
    fake_input_data: ADMAgSeasonalFieldInput,
):
    get_seasonal_field.return_value = seasonal_field_info
    get_field.return_value = field_info
    get_season.return_value = season_info
    get_harvest_info.return_value = harvest_result
    get_fertilizer_info.return_value = fertilizer_result
    get_tillage_info.return_value = tillage_result

    organic_amendments_missing_type = copy.deepcopy(omad_result)
    organic_amendments_missing_type["value"][0]["properties"].pop("type")
    get_organic_amendments_info.return_value = organic_amendments_missing_type

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)

    organic_amendments_missing_amount = copy.deepcopy(omad_result)
    organic_amendments_missing_amount["value"][0]["properties"].pop("amount")
    get_organic_amendments_info.return_value = organic_amendments_missing_amount

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)

    organic_amendments_missing_percentN = copy.deepcopy(omad_result)
    organic_amendments_missing_percentN["value"][0]["properties"].pop("percentN")
    get_organic_amendments_info.return_value = organic_amendments_missing_percentN

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)

    organic_amendments_missing_CNratio = copy.deepcopy(omad_result)
    organic_amendments_missing_CNratio["value"][0]["properties"].pop("CNratio")
    get_organic_amendments_info.return_value = organic_amendments_missing_CNratio

    with pytest.raises(ValueError):
        OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)


@pytest.fixture
def fertilizer_result() -> Dict[str, Any]:
    return {
        "value": [
            {
                "totalMaterial": {"unit": "tons/ac", "value": 5.0},
                "operationStartDateTime": "2000-01-01T00:00:00Z",
                "operationEndDateTime": "2000-01-01T00:00:00Z",
                "attachmentsLink": "http://fake-url.com/attachments",
                "partyId": "fake-party-id",
                "id": "fake-fertilizer-id",
                "eTag": "fake-etag",
                "createdDateTime": "2021-12-10T00:03:37Z",
                "modifiedDateTime": "2021-12-10T00:03:37Z",
                "source": "Fertilizer",
                "name": "Ammonium Nitrate (34-0-0)",
                "properties": {
                    "eep": "None",
                    "totalNitrogen": 4.0,
                    "method": "Surface Band / Sidedress",
                },
            }
        ],
        "nextLink": "https://fake-next-link.com",
    }


@patch("vibe_core.admag_client.ADMAgClient._post")
def test_get_fertilizer_info(
    _post: MagicMock,
    fertilizer_result: Dict[str, Any],
    admag_client: ADMAgClient,
    vibe_geometry_dict: Dict[str, Any],
):
    _post.return_value = fertilizer_result
    fertilizer_result = admag_client.get_fertilizer_info(
        party_id="fake-party-id",
        intersects_with_geometry=vibe_geometry_dict,
        min_start_operation="2001-01-01T00:00:00Z",
        max_end_operation="2001-01-01T00:00:00Z",
        associated_resource={"type": "SeasonalField", "id": "fake-seasonal_field-id"},
    )
    assert "value" in fertilizer_result
    fertilizer_list = fertilizer_result["value"]
    assert len(fertilizer_result) > 0
    fertilizer_entry = fertilizer_list[0]
    assert "operationStartDateTime" in fertilizer_entry
    assert "operationEndDateTime" in fertilizer_entry
    assert "name" in fertilizer_entry
    fertilizer_properties = fertilizer_entry["properties"]
    assert "totalNitrogen" in fertilizer_properties
    assert "eep" in fertilizer_properties


@pytest.fixture
def tillage_result(vibe_geometry_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": [
            {
                "geometry": vibe_geometry_dict,
                "attachmentsLink": "fake-attachment-link",
                "createdDateTime": "2021-12-10T00:18:33Z",
                "eTag": "fake-etag",
                "partyId": "fake-party-id",
                "id": "fake-tillage-id",
                "modifiedDateTime": "2021-12-10T00:18:33Z",
                "name": "Fake Tillage",
                "operationEndDateTime": "2001-01-01T00:00:00Z",
                "operationStartDateTime": "2001-01-01T00:00:00Z",
                "source": "fake-source",
                "status": "Active",
            },
        ]
    }


@patch("vibe_core.admag_client.ADMAgClient._post")
def test_get_tillage_info(
    _post: MagicMock,
    tillage_result: Dict[str, Any],
    admag_client: ADMAgClient,
    vibe_geometry_dict: Dict[str, Any],
):
    _post.return_value = tillage_result
    tillage_result = admag_client.get_tillage_info(
        party_id="fake-party-id",
        intersects_with_geometry=vibe_geometry_dict,
        min_start_operation="2001-01-01T00:00:00Z",
        max_end_operation="2001-01-01T00:00:00Z",
        associated_resource={"type": "SeasonalField", "id": "fake-seasonal_field-id"},
    )
    assert "value" in tillage_result
    tillage_list = tillage_result["value"]
    assert len(tillage_result) > 0
    tillage_entry = tillage_list[0]
    assert "operationStartDateTime" in tillage_entry
    assert "operationEndDateTime" in tillage_entry
    assert "name" in tillage_entry


@pytest.fixture
def omad_result(vibe_geometry_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": [
            {
                "geometry": vibe_geometry_dict,
                "attachmentsLink": "fake-attachment-link",
                "createdDateTime": "2021-12-10T00:18:33Z",
                "eTag": "fake-etag",
                "partyId": "fake-party-id",
                "id": "fake-tillage-id",
                "modifiedDateTime": "2021-12-10T00:18:33Z",
                "name": "Fake Tillage",
                "operationEndDateTime": "2001-01-01T00:00:00Z",
                "operationStartDateTime": "2001-01-01T00:00:00Z",
                "source": "fake-source",
                "status": "Active",
                "properties": {
                    "type": "fake-omad-tyoe",
                    "amount": "100",
                    "percentN": "200",
                    "CNratio": "0.05",
                },
            },
        ]
    }


@patch("vibe_core.admag_client.ADMAgClient._post")
def test_get_organic_amendments_info(
    _post: MagicMock,
    omad_result: Dict[str, Any],
    admag_client: ADMAgClient,
    vibe_geometry_dict: Dict[str, Any],
):
    _post.return_value = omad_result
    omad_result = admag_client.get_organic_amendments_info(
        party_id="fake-party-id",
        intersects_with_geometry=vibe_geometry_dict,
        min_start_operation="2001-01-01T00:00:00Z",
        max_end_operation="2001-01-01T00:00:00Z",
        associated_resource={"type": "SeasonalField", "id": "fake-seasonal_field-id"},
    )
    assert "value" in omad_result
    omad_list = omad_result["value"]
    assert len(omad_result) > 0
    omad_entry = omad_list[0]
    assert "operationStartDateTime" in omad_entry
    assert "operationEndDateTime" in omad_entry
    assert "properties" in omad_entry
    omad_properties = omad_entry["properties"]
    assert "type" in omad_properties
    assert "amount" in omad_properties
    assert "percentN" in omad_properties
    assert "CNratio" in omad_properties


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("vibe_core.admag_client.ADMAgClient.get_field")
@patch("vibe_core.admag_client.ADMAgClient.get_seasonal_field")
@patch("vibe_core.admag_client.ADMAgClient.get_season")
@patch("vibe_core.admag_client.ADMAgClient.get_harvest_info")
@patch("vibe_core.admag_client.ADMAgClient.get_fertilizer_info")
@patch("vibe_core.admag_client.ADMAgClient.get_tillage_info")
@patch("vibe_core.admag_client.ADMAgClient.get_organic_amendments_info")
@patch("vibe_core.admag_client.ADMAgClient.get_planting_info")
def test_admag_op(
    get_planting_info: MagicMock,
    get_organic_amendments_info: MagicMock,
    get_tillage_info: MagicMock,
    get_fertilizer_info: MagicMock,
    get_harvest_info: MagicMock,
    get_season: MagicMock,
    get_seasonal_field: MagicMock,
    get_field: MagicMock,
    get_token: MagicMock,
    seasonal_field_info: Dict[str, Any],
    field_info: Dict[str, Any],
    season_info: Dict[str, Any],
    harvest_result: Dict[str, Any],
    fertilizer_result: Dict[str, Any],
    tillage_result: Dict[str, Any],
    omad_result: Dict[str, Any],
    planting_result: Dict[str, Any],
    fake_input_data: ADMAgSeasonalFieldInput,
):
    get_seasonal_field.return_value = seasonal_field_info
    get_field.return_value = field_info
    get_season.return_value = season_info
    get_harvest_info.return_value = harvest_result
    get_fertilizer_info.return_value = fertilizer_result
    get_tillage_info.return_value = tillage_result
    get_organic_amendments_info.return_value = omad_result
    get_planting_info.return_value = planting_result

    output_data = OpTester(ADMAG_SEASONAL_FIELD_OP).run(admag_input=fake_input_data)
    assert output_data


@pytest.fixture
def vibe_geometry_dict() -> Dict[str, Any]:
    farm_boundary = {
        "type": "FeatureCollection",
        "name": "small_block_new_new",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [
                            [
                                [-117.046717186923388, 47.036308491044693],
                                [-117.04260145498948, 47.036329968998508],
                                [-117.042643698734992, 47.034569687054848],
                                [-117.046686589954575, 47.034558181995273],
                                [-117.046717186923388, 47.036308491044693],
                            ]
                        ]
                    ],
                },
            }
        ],
    }
    data_frame = gpd.read_file(json.dumps(farm_boundary), driver="GeoJSON")

    if not data_frame.empty:
        geometry = shpg.mapping(data_frame["geometry"][0])  # type: ignore
        return geometry
    else:
        raise Exception("No geometry found in farm boundary")


@pytest.fixture
def admag_prescriptions() -> Request:
    data = {
        "value": [
            {
                "partyId": "ae880a1b-4597-46d7-83ac-bfc6a1ae4116-16",
                "prescriptionMapId": "831989c4-c15a-4fc5-837b-4c0289d53010",
                "productCode": "1635",
                "productName": "Nutrient",
                "type": "Nutrient",
                "measurements": {
                    "N": {"value": 47.1},
                    "P": {"value": 34.99769206227461},
                    "pH": {"value": 4.978131831743143},
                    "C": {"value": 0.046408031802193},
                },
                "id": "880094d0-1c48-4d7c-b0d3-f7477a937473",
                "eTag": "24009696-0000-0100-0000-65fb20540000",
                "status": "Active",
                "createdDateTime": "2024-03-20T17:43:48Z",
                "modifiedDateTime": "2024-03-20T17:43:48Z",
                "source": "IOT device",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-117.03642546099948, 47.044663835752566],
                            [-117.05642546099949, 47.044663835752566],
                            [-117.05642546099949, 47.02466383575257],
                            [-117.03642546099948, 47.02466383575257],
                            [-117.03642546099948, 47.044663835752566],
                        ]
                    ],
                },
                "name": "Nitrogen Nutrient",
                "description": "",
                "createdBy": "f8c6c349-b484-4863-af76-d10eee669306",
                "modifiedBy": "f8c6c349-b484-4863-af76-d10eee669306",
            }
        ]
    }
    data = Request(**{"text": json.dumps(data)})
    return data


@pytest.fixture
def admag_prescriptions_dict() -> Request:
    data = {
        "partyId": "ae880a1b-4597-46d7-83ac-bfc6a1ae4116-16",
        "prescriptionMapId": "831989c4-c15a-4fc5-837b-4c0289d53010",
        "productCode": "1635",
        "productName": "Nutrient",
        "type": "Nutrient",
        "measurements": {
            "N": {"value": 47.1},
            "P": {"value": 34.99769206227461},
            "pH": {"value": 4.978131831743143},
            "C": {"value": 0.046408031802193},
        },
        "id": "880094d0-1c48-4d7c-b0d3-f7477a937473",
        "eTag": "24009696-0000-0100-0000-65fb20540000",
        "status": "Active",
        "createdDateTime": "2024-03-20T17:43:48Z",
        "modifiedDateTime": "2024-03-20T17:43:48Z",
        "source": "IOT device",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-117.03642546099948, 47.044663835752566],
                    [-117.05642546099949, 47.044663835752566],
                    [-117.05642546099949, 47.02466383575257],
                    [-117.03642546099948, 47.02466383575257],
                    [-117.03642546099948, 47.044663835752566],
                ]
            ],
        },
        "name": "Nitrogen Nutrient",
        "description": "",
        "createdBy": "f8c6c349-b484-4863-af76-d10eee669306",
        "modifiedBy": "f8c6c349-b484-4863-af76-d10eee669306",
    }

    data = Request(**{"text": json.dumps(data)})
    return data


@pytest.fixture
def admag_get_field_info() -> Request:
    data = {
        "fieldId": "63c94ae9-b0b6-46b7-8e65-311b9b44191f",
        "cropId": "ae600a8a-3011-4d7c-8146-1f039ba619d0",
        "seasonId": "ae600a8a-3011-4d7c-8146-1f039ba619d0",
        "createdDateTime": "2021-03-21T01:37:06Z",
        "modifiedDateTime": "2021-03-21T01:37:06Z",
        "seasonal_field_id": "",
    }

    data = Request(**{"text": json.dumps(data)})
    return data


@pytest.fixture
def admag_get_prescription_map_id() -> Request:
    data = {
        "value": [
            {
                "partyId": "ae880a1b-4597-46d7-83ac-bfc6a1ae4116-16",
                "type": "Soil Nutrient Map",
                "seasonId": "ae600a8a-3011-4d7c-8146-1f039ba619d0-16",
                "cropId": "d4c8427b-4540-4c05-82f6-27c771e48b7c",
                "fieldId": "04b1d9f6-7444-4df5-b468-9a4e4c96314e-16",
                "id": "831989c4-c15a-4fc5-837b-4c0289d53050",
                "eTag": "8400e17b-0000-0100-0000-660075240000",
                "status": "Active",
                "createdDateTime": "2024-03-21T14:48:27Z",
                "modifiedDateTime": "2024-03-24T18:47:00Z",
                "source": "IOT devices",
                "name": "Prescription test Map",
                "description": "Farmbeats Agriculture research",
                "createdBy": "f8c6c349-b484-4863-af76-d10eee669306",
                "modifiedBy": "255a13c4-c1e0-4ac9-9e60-5139b3f8e0a3",
                "properties": {"seasonal_field_id": "fake-seasonal-field-id"},
            }
        ]
    }
    data = Request(**{"text": json.dumps(data)})
    return data


@pytest.fixture
def admag_seasonal_field_info(seasonal_field_info: Dict[str, Any]) -> Request:
    data = Request(**{"text": json.dumps(seasonal_field_info)})
    return data


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("requests.Session.request")
def test_prescriptions(
    session_mock: Mock,
    _: MagicMock,
    admag_prescriptions: str,
    admag_seasonal_field_info: str,
    fake_input_data: ADMAgSeasonalFieldInput,
    prescription_geom_input: List[ADMAgPrescription],
):
    session_mock.side_effect = [
        admag_seasonal_field_info,
        admag_prescriptions,
    ]
    parameters = {
        "base_url": "base_url",
        "client_id": "client_id",
        "client_secret": "client_secret",
        "authority": "authority",
        "default_scope": "default_scope",
    }
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prescriptions.yaml")
    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)
    output_data = op_.run(
        admag_input=fake_input_data,
        prescriptions_with_geom_input=prescription_geom_input,  # type: ignore
    )
    assets = cast(List[AssetVibe], output_data["response"].assets)  # type: ignore
    assert len(assets[0].path_or_url) > 0


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("requests.Session.request")
def test_list_prescriptions(
    session_mock: Mock,
    _: MagicMock,
    admag_prescriptions: str,
    admag_get_prescription_map_id: str,
    admag_seasonal_field_info: str,
    fake_input_data: ADMAgSeasonalFieldInput,
):
    session_mock.side_effect = [
        admag_seasonal_field_info,
        admag_get_prescription_map_id,
        admag_prescriptions,
    ]
    parameters = {
        "base_url": "base_url",
        "client_id": "client_id",
        "client_secret": "client_secret",
        "authority": "authority",
        "default_scope": "default_scope",
    }
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "list_prescriptions.yaml"
    )
    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)
    output_data = op_.run(admag_input=fake_input_data)
    assert "prescriptions" in output_data


@patch("vibe_core.admag_client.ADMAgClient.get_token", return_value="my_fake_token")
@patch("requests.Session.request")
def test_get_prescriptions(
    session_mock: Mock,
    _: MagicMock,
    admag_prescriptions_dict: str,
    fake_prescription_input_data: ADMAgPrescriptionInput,
):
    session_mock.side_effect = [
        admag_prescriptions_dict,
    ]
    parameters = {
        "base_url": "base_url",
        "client_id": "client_id",
        "client_secret": "client_secret",
        "authority": "authority",
        "default_scope": "default_scope",
    }
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_prescription.yaml")
    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)
    output_data = op_.run(prescription_without_geom_input=fake_prescription_input_data)
    assert "prescription_with_geom" in output_data
