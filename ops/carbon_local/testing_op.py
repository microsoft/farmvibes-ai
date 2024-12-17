from datetime import datetime
from datetime import datetime
from typing import Any, Dict
from shapely import geometry as shpg
import json
import os

from vibe_core.data import (
    FertilizerInformation,
    HarvestInformation,
    OrganicAmendmentInformation,
    SeasonalFieldInformation,
    TillageInformation,
    DataVibe,
)

from vibe_dev.testing.op_tester import OpTester

CARBON_YAML = "F:/Comet-Changes/farmvibes-ai/ops/carbon_local/whatif_comet_local_op.yaml"
def seasonal_field_from_dict(data: Dict[str, Any]) -> SeasonalFieldInformation:
    harvests = [HarvestInformation(**h) for h in data["harvests"]]
    tillages = [TillageInformation(**t) for t in data["tillages"]]
    organic_amendments = [OrganicAmendmentInformation(**o) for o in data["organic_amendments"]]
    fertilizers = [FertilizerInformation(**f) for f in data["fertilizers"]]


    return SeasonalFieldInformation(
        id=data["id"],
        time_range=(
            datetime.fromisoformat(data["time_range"][0]),
            datetime.fromisoformat(data["time_range"][1]),
        ),
        geometry=data["geometry"],
        assets=data["assets"],
        crop_name=data["crop_name"],
        crop_type=data["crop_type"],
        properties=data["properties"],
        fertilizers=fertilizers,
        harvests=harvests,
        tillages=tillages,
        organic_amendments=organic_amendments,
    )

def carbon_sequesteration():
    current_path = os.path.abspath(__file__)
    while True:
        if os.path.basename(current_path) == "farmvibes-ai":
            repo_root = current_path
            break
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:  # Root of the filesystem reached
            raise RuntimeError("Repository root 'farmvibes-ai' not found")
        current_path = parent_path
    op_tester = OpTester(CARBON_YAML)
    baseline_seasonal_fields_path = os.path.join(repo_root, "notebooks", "carbon", "baseline_seasonal_fields.json")
    scenario_seasonal_fields_path = os.path.join(repo_root, "notebooks", "carbon", "scenario_seasonal_fields.json")
    baseline_fields_file = baseline_seasonal_fields_path
    with open(baseline_fields_file) as json_file:
        baseline_seasonal_fields = [
            seasonal_field_from_dict(seasonal_field_dict)
            for seasonal_field_dict in json.load(json_file)
        ]
    scenario_fields_file = scenario_seasonal_fields_path
    with open(scenario_fields_file) as json_file:
        scenario_seasonal_fields = [
            seasonal_field_from_dict(seasonal_field_dict)
            for seasonal_field_dict in json.load(json_file)

        ]

    run = op_tester.run(baseline_seasonal_fields=baseline_seasonal_fields,scenario_seasonal_fields=scenario_seasonal_fields)
    #run.monitor()
    #carbon_offset_info = run.output["carbon_output"][0]
    #print(f"Estimated carbon offset is {carbon_offset_info.carbon}")
    print(run)

if __name__ == "__main__":
    carbon_sequesteration()