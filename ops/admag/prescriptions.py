# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import shape

from vibe_core.admag_client import ADMAgClient
from vibe_core.data import (
    ADMAgPrescription,
    ADMAgSeasonalFieldInput,
    AssetVibe,
    GeometryCollection,
    gen_guid,
    gen_hash_id,
)

API_VERSION = "2023-11-01-preview"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class CallbackBuilder:
    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        authority: str,
        default_scope: str,
    ):
        self.temp_dir = TemporaryDirectory()

        self.admag_client = ADMAgClient(
            base_url=base_url,
            api_version=API_VERSION,
            client_id=client_id,
            client_secret=client_secret,
            authority=authority,
            default_scope=default_scope,
        )

    def get_prescriptions(self, prescriptions: List[ADMAgPrescription]) -> AssetVibe:
        if not prescriptions:
            raise ValueError("No prescriptions found")

        measures = [item.measurements for item in prescriptions]
        geometry = [shape(item.geometry) for item in prescriptions]
        df = pd.DataFrame(measures)

        for column in df.columns:
            df[column] = df[column].apply(lambda x: x["value"])  # type: ignore

        df["geometry"] = geometry

        df = GeoDataFrame(data=df, geometry="geometry")  # type: ignore
        out_path = f"{self.temp_dir.name}/prescription.geojson"
        df.to_file(out_path, driver="GeoJSON")
        asset_vibe = AssetVibe(reference=out_path, type="application/json", id=gen_guid())
        return asset_vibe

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
        }
        geometry = response["geometry"]
        return field_info, geometry

    def prescriptions(
        self, user_input: ADMAgSeasonalFieldInput, prescriptions: List[ADMAgPrescription]
    ) -> GeometryCollection:
        field_info, geometry = self.get_field_info(
            user_input.party_id, user_input.seasonal_field_id
        )
        asset_vibe = self.get_prescriptions(prescriptions)

        time_range = (
            datetime.strptime(prescriptions[0].createdDateTime, DATE_FORMAT),
            datetime.strptime(prescriptions[0].modifiedDateTime, DATE_FORMAT),
        )
        return GeometryCollection(
            id=gen_hash_id("heatmap_nutrients", geometry, time_range),
            time_range=time_range,
            geometry=geometry,
            assets=[asset_vibe],
        )

    def __call__(self):
        def prescriptions_init(
            admag_input: ADMAgSeasonalFieldInput,
            prescriptions_with_geom_input: List[ADMAgPrescription],
        ) -> Dict[str, GeometryCollection]:
            out_prescriptions = self.prescriptions(admag_input, prescriptions_with_geom_input)
            return {"response": out_prescriptions}

        return prescriptions_init

    def __del__(self):
        if self.temp_dir:
            self.temp_dir.cleanup()
