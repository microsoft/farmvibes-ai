# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from hydra_zen import MISSING, builds, instantiate
from shapely.geometry import Polygon, mapping

from vibe_agent.storage import Storage
from vibe_common.input_handlers import gen_stac_item_from_bounds
from vibe_common.schemas import CacheInfo, ItemDict, OpRunId

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_PATH = os.path.join(HERE, "..", "..", "..", "workflows", "helloworld.yaml")


class DipatchedStorage(Storage):
    def __init__(self, original: Storage):
        self.original = original

    def retrieve_output_from_input_if_exists(self, *args: Any):
        ret = self.original.retrieve_output_from_input_if_exists(*args)
        assert ret is not None
        return ret

    async def retrieve_output_from_input_if_exists_async(
        self, cache_info: CacheInfo, **kwargs: Any
    ) -> Optional[ItemDict]:
        ret = await self.original.retrieve_output_from_input_if_exists_async(cache_info, **kwargs)
        assert ret is not None
        return ret

    def store(self, *args: Any):
        return self.original.store(*args)

    def __getattr__(self, name: str):
        return getattr(self.original, name)

    def remove(self, op_run_id: OpRunId):
        self.original.remove(op_run_id)


PatchedStorageConfig = builds(
    DipatchedStorage,
    original=MISSING,
    zen_dataclass={
        "module": "tests.workflows_integration.test_helloworld_integration",
        "cls_name": "PatchedStorageConfig",
    },
)


@pytest.fixture
def helloworld_input() -> Dict[str, Any]:
    polygon_coords = [
        (-88.062073563448919, 37.081397673802059),
        (-88.026349330507315, 37.085463858128762),
        (-88.026349330507315, 37.085463858128762),
        (-88.012445388773259, 37.069230099135126),
        (-88.035931592028305, 37.048441375086092),
        (-88.068120429075847, 37.058833638440767),
        (-88.062073563448919, 37.081397673802059),
    ]
    polygon: Dict[str, Any] = mapping(Polygon(polygon_coords))  # type: ignore
    start_date = datetime(year=2021, month=2, day=1, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=2, day=11, tzinfo=timezone.utc)

    return gen_stac_item_from_bounds(polygon, start_date, end_date)


# TODO: add "remote" to the list of storage_spec
@pytest.mark.parametrize("storage_spec", ["local"], indirect=True)
@pytest.mark.anyio
async def test_helloworld_workflow(
    storage_spec: Any,
    helloworld_input: List[Dict[str, Any]],
    workflow_test_helper,  # type: ignore
):
    runner = workflow_test_helper.gen_workflow(WORKFLOW_PATH, storage_spec)
    result = await runner.run({k: helloworld_input for k in runner.workflow.inputs_spec})

    workflow_test_helper.verify_workflow_result(WORKFLOW_PATH, result)


# TODO: add "remote" to the list of storage_spec
@pytest.mark.parametrize("storage_spec", ["local"], indirect=True)
@pytest.mark.anyio
async def test_helloworld_cache(
    storage_spec: Any,
    helloworld_input: List[Dict[str, Any]],
    workflow_test_helper,  # type: ignore
    tmpdir: str,
):
    runner = workflow_test_helper.gen_workflow(WORKFLOW_PATH, storage_spec)

    result_first_run = await runner.run({k: helloworld_input for k in runner.workflow.inputs_spec})
    workflow_test_helper.verify_workflow_result(WORKFLOW_PATH, result_first_run)

    runner = workflow_test_helper.gen_workflow(
        WORKFLOW_PATH, PatchedStorageConfig(original=instantiate(storage_spec))
    )
    result_second_run = await runner.run({k: helloworld_input for k in runner.workflow.inputs_spec})

    workflow_test_helper.verify_workflow_result(WORKFLOW_PATH, result_second_run)

    # Need to improve this test to be agnostic to the order of elements in the list
    assert result_first_run.keys() == result_second_run.keys()
    for k in result_first_run.keys():
        out1 = result_first_run[k]
        out2 = result_second_run[k]
        assert len(out1) == len(out2)
        assert out1["id"] == out2["id"]
        assert out1["assets"].keys() == out2["assets"].keys()
