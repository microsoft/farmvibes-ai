from typing import Any, Dict, List, Tuple

from vibe_core.admag_client import ADMAgClient
from vibe_core.data import ADMAgPrescriptionInput, ADMAgSeasonalFieldInput

API_VERSION = "2023-11-01-preview"


class CallbackBuilder:
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

    def get_prescriptions(
        self, party_id: str, field_info: Dict[str, str], intersect_geometry: Dict[str, Any]
    ) -> List[ADMAgPrescriptionInput]:
        response = self.admag_client.get_prescription_map_id(
            party_id=party_id,
            field_id=field_info["fieldId"],
            crop_id=field_info["cropId"],
        )

        prescription_map_id = None
        for p_map in response["value"]:
            if "properties" in p_map and "seasonal_field_id" in p_map["properties"]:
                if p_map["properties"]["seasonal_field_id"] == field_info["seasonal_field_id"]:
                    prescription_map_id = p_map["id"]
                    break

        if not prescription_map_id:
            raise ValueError("Prescription map not found")

        response = self.admag_client.get_prescriptions(
            party_id, prescription_map_id, geometry=intersect_geometry
        )

        prescriptions = []

        for value in response["value"]:
            prescriptions.append(
                ADMAgPrescriptionInput(
                    prescription_id=value["id"],
                    party_id=value["partyId"],
                )
            )

        return prescriptions

    def get_field_info(
        self, party_id: str, seasonal_field_id: str
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        response = self.admag_client.get_seasonal_field(party_id, seasonal_field_id)
        field_info = {
            "fieldId": response["fieldId"],
            "cropId": response["cropId"],
            "seasonId": response["seasonId"],
            "createdDateTime": response["createdDateTime"],
            "modifiedDateTime": response["modifiedDateTime"],
            "seasonal_field_id": seasonal_field_id,
        }
        geometry = response["geometry"]
        return field_info, geometry

    def prescriptions(self, user_input: ADMAgSeasonalFieldInput) -> List[ADMAgPrescriptionInput]:
        field_info, geometry = self.get_field_info(
            user_input.party_id, user_input.seasonal_field_id
        )

        list_prescriptions = self.get_prescriptions(
            user_input.party_id, field_info, intersect_geometry=geometry
        )
        return list_prescriptions

    def __call__(self):
        def prescriptions_init(
            admag_input: ADMAgSeasonalFieldInput,
        ) -> Dict[str, List[ADMAgPrescriptionInput]]:
            out_prescriptions = self.prescriptions(admag_input)
            return {"prescriptions": out_prescriptions}

        return prescriptions_init
