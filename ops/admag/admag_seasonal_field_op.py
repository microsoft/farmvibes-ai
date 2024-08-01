# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from typing import Any, Dict, Tuple

from vibe_core.admag_client import ADMAgClient
from vibe_core.data import (
    ADMAgSeasonalFieldInput,
    FertilizerInformation,
    HarvestInformation,
    OrganicAmendmentInformation,
    SeasonalFieldInformation,
    TillageInformation,
    gen_guid,
)

API_VERSION = "2023-11-01-preview"


class ADMAgConnector:
    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        authority: str,
        default_scope: str,
    ):
        self.admag_client = ADMAgClient(
            base_url=base_url,
            api_version=API_VERSION,
            client_id=client_id,
            client_secret=client_secret,
            authority=authority,
            default_scope=default_scope,
        )
        self.date_fmt = "%Y-%m-%dT%H:%M:%S%z"

    def get_field_entities(
        self, admag_input: ADMAgSeasonalFieldInput
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        seasonal_field_info: Dict[str, Any] = self.admag_client.get_seasonal_field(
            admag_input.party_id, admag_input.seasonal_field_id
        )

        field_info = self.admag_client.get_field(
            admag_input.party_id, seasonal_field_info["fieldId"]
        )

        season_info: Dict[str, Any] = self.admag_client.get_season(seasonal_field_info["seasonId"])

        return (
            seasonal_field_info,
            field_info,
            season_info,
        )

    def get_harvests(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        def check_harvest_properties(harvest: Dict[str, Any]) -> Dict[str, Any]:
            if "gfsrt" not in harvest["properties"]:
                raise ValueError(
                    "Harvest does not have gfsrt property. "
                    f"Please check harvest properties with id={harvest['id']} in Admag. "
                    "havest['properties']['gfsrt'] = True, means the crop is grain."
                )

            if "strawStoverHayRemoval" not in harvest["properties"]:
                raise ValueError(
                    "Harvest does not have strawStoverHayRemoval property "
                    f"for entity with id={harvest['id']}. "
                    "Please check harvest properties in Admag. "
                    "strawStoverHayremoval is percentage of straw, "
                    "stover, and hay removed at harvest."
                )

            return harvest

        harvest_result = self.admag_client.get_harvest_info(
            party_id,
            intersects_with_geometry,
            min_start_operation,
            max_end_operation,
            associated_resource,
        )

        [check_harvest_properties(harvest) for harvest in harvest_result["value"]]

        return [
            HarvestInformation(
                is_grain=harvest["properties"]["gfsrt"] == "True",
                start_date=harvest["operationStartDateTime"],
                end_date=harvest["operationEndDateTime"],
                crop_yield=harvest["totalYield"]["value"],
                stray_stover_hay_removal=harvest["properties"]["strawStoverHayRemoval"],
            )
            for harvest in harvest_result["value"]
        ]

    def get_latest_harvest(
        self,
        operation_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        harvest_result = self.admag_client.get_harvest_info(**operation_params)
        if "value" in harvest_result and len(harvest_result["value"]) == 0:
            raise ValueError(f"No harvest found with parameters: {operation_params}")
        latest_harvest = max(harvest_result["value"], key=lambda x: x["operationEndDateTime"])
        return latest_harvest

    def get_fertilizers(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        def check_fertilizer_properties(fertilizer: Dict[str, Any]):
            if "totalNitrogen" not in fertilizer["properties"]:
                raise ValueError(
                    "Fertilizer does not have totalNitrogen property. "
                    f"Please check ADMAg application with id={fertilizer['id']}. "
                    "totalNitrogen is the total amount of nitrogen applied (lbs N/acre)."
                )

            if "eep" not in fertilizer["properties"]:
                raise ValueError(
                    "Fertilizer does not have eep property. "
                    f"Please check ADMAg application with id={fertilizer['id']}. "
                    "eep is the enhanced efficiency phosphorus."
                )

            possible_eeps = ["None", "Slow Release", "Nitrification Inhibitor"]
            if fertilizer["properties"]["eep"] not in possible_eeps:
                raise ValueError(
                    f"eep property of ADMAg application with id={fertilizer['id']} "
                    "is not one of the allowed values. "
                    f"Allowed values are {possible_eeps}"
                )

        fertilizer_result = self.admag_client.get_fertilizer_info(
            party_id,
            intersects_with_geometry,
            min_start_operation,
            max_end_operation,
            associated_resource,
        )

        [check_fertilizer_properties(fertilizer) for fertilizer in fertilizer_result["value"]]

        return [
            FertilizerInformation(
                start_date=fertilizer["operationStartDateTime"],
                end_date=fertilizer["operationEndDateTime"],
                application_type=fertilizer["name"],
                total_nitrogen=fertilizer["properties"]["totalNitrogen"],
                enhanced_efficiency_phosphorus=fertilizer["properties"]["eep"],
            )
            for fertilizer in fertilizer_result["value"]
        ]

    def get_first_planting(
        self,
        operation_params: Dict[str, Any],
    ):
        operation_result = self.admag_client.get_planting_info(**operation_params)

        if "value" in operation_result and len(operation_result["value"]) == 0:
            raise ValueError(f"No planting found with parameters: {operation_params}")
        obj_start = min(operation_result["value"], key=lambda x: x["operationStartDateTime"])
        return obj_start["operationStartDateTime"]

    def get_tillages(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        tillage_result = self.admag_client.get_tillage_info(
            party_id,
            intersects_with_geometry,
            min_start_operation,
            max_end_operation,
            associated_resource,
        )

        return [
            TillageInformation(
                implement=tilage["name"],
                start_date=tilage["operationStartDateTime"],
                end_date=tilage["operationEndDateTime"],
            )
            for tilage in tillage_result["value"]
        ]

    def get_organic_amendments(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        def check_organic_amendment_properties(organic_amendments: Dict[str, Any]):
            if "type" not in organic_amendments["properties"]:
                raise ValueError(
                    "Organic amendment does not have type property. "
                    f"Please check ADMAg application with id={organic_amendments['id']}. "
                    "Type is the type of organic amendment. Check Comet-Farm API documentation "
                    "for the list of allowed values."
                )

            if "amount" not in organic_amendments["properties"]:
                raise ValueError(
                    "Organic amendment does not have amount property. "
                    f"Please check ADMAg application with id={organic_amendments['id']}. "
                    "Amount is the amount of organic amendment applied (tons/acre)."
                )

            if "percentN" not in organic_amendments["properties"]:
                raise ValueError(
                    "Organic amendment does not have percentN property. "
                    f"Please check ADMAg application with id={organic_amendments['id']}. "
                    "percentN is the percent nitrogen in the organic amendment."
                )

            if "CNratio" not in organic_amendments["properties"]:
                raise ValueError(
                    "Organic amendment does not have CNratio property. "
                    f"Please check ADMAg application with id={organic_amendments['id']}. "
                    "CNratio is the carbon nitrogen ratio of the organic amendment."
                )

        omad_result = self.admag_client.get_organic_amendments_info(
            party_id,
            intersects_with_geometry,
            min_start_operation,
            max_end_operation,
            associated_resource,
        )

        [
            check_organic_amendment_properties(organic_amendment)
            for organic_amendment in omad_result["value"]
        ]

        return [
            OrganicAmendmentInformation(
                start_date=omad["operationStartDateTime"],
                end_date=omad["operationEndDateTime"],
                organic_amendment_type=omad["properties"]["type"],
                organic_amendment_amount=omad["properties"]["amount"],
                organic_amendment_percent_nitrogen=omad["properties"]["percentN"],
                organic_amendment_carbon_nitrogen_ratio=omad["properties"]["CNratio"],
            )
            for omad in omad_result["value"]
        ]

    def get_season_field_data(
        self,
        party_id: str,
        seasonal_field_info: Dict[str, Any],
        season_info: Dict[str, Any],
        field_info: Dict[str, Any],
    ) -> SeasonalFieldInformation:
        associated_resource = {"type": "SeasonalField", "id": seasonal_field_info["id"]}

        operation_params = {
            "party_id": party_id,
            "intersects_with_geometry": seasonal_field_info["geometry"],
            "min_start_operation": season_info["startDateTime"],
            "max_end_operation": season_info["endDateTime"],
            "associated_resource": associated_resource,
        }

        latest_harvest = self.get_latest_harvest(operation_params)

        planting_start_time = self.get_first_planting(operation_params)

        return SeasonalFieldInformation(
            id=gen_guid(),
            time_range=(
                datetime.strptime(planting_start_time, self.date_fmt),
                datetime.strptime(latest_harvest["operationEndDateTime"], self.date_fmt),
            ),
            geometry=seasonal_field_info["geometry"],
            assets=[],
            crop_name=seasonal_field_info["name"],
            crop_type=seasonal_field_info["description"],
            fertilizers=self.get_fertilizers(**operation_params),
            harvests=self.get_harvests(**operation_params),
            tillages=self.get_tillages(**operation_params),
            organic_amendments=self.get_organic_amendments(**operation_params),
            properties=field_info["properties"],
        )

    def __call__(self):
        def get_admag_seasonal_field(
            admag_input: ADMAgSeasonalFieldInput,
        ) -> Dict[str, SeasonalFieldInformation]:
            seasonal_field_info, field_info, season_info = self.get_field_entities(admag_input)
            seasonal_field = self.get_season_field_data(
                admag_input.party_id, seasonal_field_info, season_info, field_info
            )

            return {"seasonal_field": seasonal_field}

        return get_admag_seasonal_field
