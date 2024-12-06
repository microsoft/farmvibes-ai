# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List

from pyngrok import ngrok
from pyproj import Geod
from shapely.geometry import shape

from vibe_core.data import (
    CarbonOffsetInfo,
    FertilizerInformation,
    HarvestInformation,
    OrganicAmendmentInformation,
    SeasonalFieldInformation,
    TillageInformation,
    gen_guid,
)
from vibe_lib.comet_farm.comet_requester import CometRequester, CometServerParameters
from vibe_lib.comet_farm.comet_server import HTTP_SERVER_HOST, HTTP_SERVER_PORT

WEBHOOK_URL = f"http://{HTTP_SERVER_HOST}:{HTTP_SERVER_PORT}"


class SeasonalFieldConverter:
    def get_location(self, geojson: Dict[str, Any]):
        """
        calculate area and center point of polygon
        """
        s = shape(geojson)

        location = (s.centroid.x, s.centroid.y)  # type: ignore

        geod = Geod("+a=6378137 +f=0.0033528106647475126")
        area_in_acres = geod.geometry_area_perimeter(s)[0] * 0.000247105

        return (area_in_acres, location)

    def format_datetime(self, date: str) -> str:
        date_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
        return date_obj.strftime("%m/%d/%Y")

    def _add_historical(self, historical_data: Dict[str, Any], cropland: ET.Element):
        ET.SubElement(cropland, "Pre-1980").text = historical_data["pre_1980"]
        ET.SubElement(cropland, "CRP").text = historical_data["crp_type"]
        ET.SubElement(cropland, "CRPStartYear").text = historical_data["crp_start"]
        ET.SubElement(cropland, "CRPEndYear").text = historical_data["crp_end"]
        ET.SubElement(cropland, "CRPType").text = historical_data["crp_type"]
        ET.SubElement(cropland, "Year1980-2000").text = historical_data["year_1980_2000"]
        ET.SubElement(cropland, "Year1980-2000_Tillage").text = historical_data[
            "year_1980_2000_tillage"
        ]

    def _add_harvest_information(self, harvest_data: HarvestInformation, harvest_list: ET.Element):
        if isinstance(harvest_data, dict):
            harvest_data = HarvestInformation(**harvest_data)
        harvest = ET.SubElement(harvest_list, "HarvestEvent")

        ET.SubElement(harvest, "HarvestDate").text = self.format_datetime(harvest_data.end_date)
        ET.SubElement(harvest, "Grain").text = "Yes" if harvest_data.is_grain else "No"
        ET.SubElement(harvest, "yield").text = str(harvest_data.crop_yield)
        ET.SubElement(harvest, "StrawStoverHayRemoval").text = str(
            harvest_data.stray_stover_hay_removal
        )

    def _add_tillage_information(self, tillage_data: TillageInformation, tillage_list: ET.Element):
        if isinstance(tillage_data, dict):
            tillage_data = TillageInformation(**tillage_data)
        tillage = ET.SubElement(tillage_list, "TillageEvent")
        ET.SubElement(tillage, "TillageDate").text = self.format_datetime(tillage_data.end_date)
        ET.SubElement(tillage, "TillageType").text = tillage_data.implement

    def _add_fertilization_information(
        self, fertilizer_data: FertilizerInformation, fertilization_list: ET.Element
    ):
        if isinstance(fertilizer_data, dict):
            fertilizer_data = FertilizerInformation(**fertilizer_data)
        fertilizer = ET.SubElement(fertilization_list, "NApplicationEvent")
        fertilizer_date = self.format_datetime(fertilizer_data.end_date)
        ET.SubElement(fertilizer, "NApplicationDate").text = fertilizer_date
        ET.SubElement(fertilizer, "NApplicationType").text = fertilizer_data.application_type
        ET.SubElement(fertilizer, "NApplicationAmount").text = str(fertilizer_data.total_nitrogen)
        ET.SubElement(fertilizer, "NApplicationMethod").text = "Surface Band / Sidedress"
        ET.SubElement(fertilizer, "EEP").text = fertilizer_data.enhanced_efficiency_phosphorus

    def _add_organic_amendmentes_information(
        self, omad_data: OrganicAmendmentInformation, omad_list: ET.Element
    ):
        if isinstance(omad_data, dict):
            # Same restriction of previous method
            omad_data = OrganicAmendmentInformation(**omad_data)
        omadevent = ET.SubElement(omad_list, "OMADApplicationEvent")
        ET.SubElement(omadevent, "OMADApplicationDate").text = self.format_datetime(
            omad_data.end_date
        )
        ET.SubElement(omadevent, "OMADType").text = omad_data.organic_amendment_type
        ET.SubElement(omadevent, "OMADAmount").text = str(omad_data.organic_amendment_amount)
        ET.SubElement(omadevent, "OMADPercentN").text = str(
            omad_data.organic_amendment_percent_nitrogen
        )
        ET.SubElement(omadevent, "OMADCNRatio").text = str(
            omad_data.organic_amendment_carbon_nitrogen_ratio
        )

    def _add_seasonal_field(
        self, seasonal_field: SeasonalFieldInformation, year: ET.Element, crop_number: int
    ):
        crop = ET.SubElement(year, "Crop")
        # According to COMET documentation crop numbers
        # can be only 1, 2 or -1 if cover
        crop_number = crop_number + 1
        crop_number = min(crop_number, 2)
        crop.attrib["CropNumber"] = (
            "-1" if "cover" in seasonal_field.crop_type.lower() else str(crop_number)
        )
        ET.SubElement(crop, "CropName").text = seasonal_field.crop_name
        # We assume SeasonalField.time_range = (plantingDate, lastHarvestDate)
        ET.SubElement(crop, "PlantingDate").text = seasonal_field.time_range[0].strftime("%m/%d/%Y")
        ET.SubElement(crop, "ContinueFromPreviousYear").text = "N"

        harvest_list = ET.SubElement(crop, "HarvestList")
        [
            self._add_harvest_information(harvest_data, harvest_list)
            for harvest_data in seasonal_field.harvests
        ]
        ET.SubElement(crop, "GrazingList")

        tillage_list = ET.SubElement(crop, "TillageList")
        [
            self._add_tillage_information(tillage_data, tillage_list)
            for tillage_data in seasonal_field.tillages
        ]

        fertilizer_list = ET.SubElement(crop, "NApplicationList")
        [
            self._add_fertilization_information(fertilizer_data, fertilizer_list)
            for fertilizer_data in seasonal_field.fertilizers
        ]

        omad_application_list = ET.SubElement(crop, "OMADApplicationList")
        [
            self._add_organic_amendmentes_information(omad_data, omad_application_list)
            for omad_data in seasonal_field.organic_amendments
        ]

        ET.SubElement(crop, "IrrigationList")

        pass

    def _add_scenario(self, seasonal_fields: List[SeasonalFieldInformation], scenario: ET.Element):
        min_year = min(seasonal_fields, key=lambda x: x.time_range[0].year).time_range[0].year
        max_year = max(seasonal_fields, key=lambda x: x.time_range[0].year).time_range[0].year

        for crop_year in list(range(min_year, max_year + 1)):
            if any(s.time_range[0].year == crop_year for s in seasonal_fields):
                year_element = ET.SubElement(scenario, "CropYear")
                year_element.attrib["Year"] = str(crop_year)
                for crop_number, seasonal_field in enumerate(
                    filter(lambda s: s.time_range[0].year == crop_year, seasonal_fields)
                ):
                    self._add_seasonal_field(seasonal_field, year_element, crop_number)

    def build_comet_request(
        self,
        support_email: str,
        baseline_seasonal_fields: List[SeasonalFieldInformation],
        scenario_seasonal_fields: List[SeasonalFieldInformation],
    ) -> str:
        root = ET.fromstring("<Day />")
        tree = ET.ElementTree(root)
        root.attrib["cometEmailId"] = support_email

        cropland = ET.SubElement(root, "Cropland")
        cropland.attrib["name"] = "sdk_int1"

        # Baseline field
        baseline_field = baseline_seasonal_fields[0]

        # cropland elements
        farm_location = self.get_location(baseline_field.geometry)

        geom = ET.SubElement(cropland, "GEOM")
        geom.attrib["SRID"] = "4326"
        geom.attrib["AREA"] = str(farm_location[0])
        geom.text = f"POINT({farm_location[1][0]} {farm_location[1][1]})"

        self._add_historical(baseline_field.properties, cropland)

        scenario = ET.SubElement(cropland, "CropScenario")
        scenario.attrib["Name"] = "Current"
        self._add_scenario(seasonal_fields=baseline_seasonal_fields, scenario=scenario)

        scenario = ET.SubElement(cropland, "CropScenario")
        scenario.attrib["Name"] = "scenario: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self._add_scenario(seasonal_fields=scenario_seasonal_fields, scenario=scenario)

        return ET.tostring(tree.getroot(), encoding="unicode")


class CallbackBuilder:
    def __init__(self, comet_url: str, comet_support_email: str, ngrok_token: str, api_key: str):
        self.cometRequest = CometServerParameters(
            url=comet_url,
            webhook=WEBHOOK_URL,
            supportEmail=comet_support_email,
            ngrokToken=ngrok_token,
            apiKey=api_key
        )

        self.comet_requester = CometRequester(self.cometRequest)

        self.start_date = datetime.now(timezone.utc)
        self.end_date = datetime.now(timezone.utc)

    def get_carbon_offset(
        self,
        baseline_seasonal_fields: List[SeasonalFieldInformation],
        scenario_seasonal_fields: List[SeasonalFieldInformation],
    ) -> Dict[str, CarbonOffsetInfo]:
        converter = SeasonalFieldConverter()
        xml_str = converter.build_comet_request(
            self.cometRequest.supportEmail, baseline_seasonal_fields, scenario_seasonal_fields
        )

        comet_response = self.comet_requester.run_comet_request(xml_str)

        obj_carbon = CarbonOffsetInfo(
            id=gen_guid(),
            geometry=scenario_seasonal_fields[-1].geometry,
            time_range=(
                baseline_seasonal_fields[0].time_range[0],
                scenario_seasonal_fields[-1].time_range[1],
            ),
            assets=[],
            carbon=comet_response,
        )

        return {"carbon_output": obj_carbon}

    def __call__(self):
        return self.get_carbon_offset

    def __del__(self):
        try:
            ngrok.kill()
        except Exception:
            pass
