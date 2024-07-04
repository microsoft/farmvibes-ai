from typing import Dict

from vibe_core.admag_client import ADMAgClient
from vibe_core.data import ADMAgPrescription, ADMAgPrescriptionInput

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

    def prescriptions(self, user_input: ADMAgPrescriptionInput) -> ADMAgPrescription:
        response = self.admag_client.get_prescription(
            user_input.party_id, user_input.prescription_id
        )

        prescription = ADMAgPrescription(**response)

        return prescription

    def __call__(self):
        def prescriptions_init(
            prescription_without_geom_input: ADMAgPrescriptionInput,
        ) -> Dict[str, ADMAgPrescription]:
            out_prescriptions = self.prescriptions(prescription_without_geom_input)
            return {"prescription_with_geom": out_prescriptions}

        return prescriptions_init
