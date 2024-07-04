from typing import Any, List
from uuid import UUID, uuid4

import pytest

from vibe_common.messaging import OpIOType
from vibe_core.data.utils import StacConverter, is_container_type, serialize_stac
from vibe_dev.testing.fake_workflows_fixtures import (  # noqa
    fake_ops_dir,
    fake_workflows_dir,
    get_fake_workflow_path,
)
from vibe_dev.testing.workflow_fixtures import THE_DATAVIBE
from vibe_server.workflow.runner.runner import WorkflowRunner
from vibe_server.workflow.runner.task_io_handler import WorkflowIOHandler
from vibe_server.workflow.workflow import GraphNodeType, Workflow


class MockWorkflowRunner(WorkflowRunner):
    def __init__(self, fail_list: List[str], *args: Any, **kwargs: Any):
        self.fail_list = fail_list
        super().__init__(*args, **kwargs)

    async def _run_op_impl(
        self, op: GraphNodeType, input: OpIOType, run_id: UUID, _: int
    ) -> OpIOType:
        for fail in self.fail_list:
            if op.name.startswith(fail):
                raise RuntimeError(f"Failed op {op} because it was in the fail list")
        converter = StacConverter()
        return {
            k: serialize_stac(
                converter.to_stac_item(
                    [THE_DATAVIBE] if is_container_type(v) else THE_DATAVIBE  # type: ignore
                )
            )
            for k, v in op.spec.output_spec.items()
        }


@pytest.mark.anyio
async def test_one_failure_in_sink_fails_workflow(
    fake_ops_dir: str,  # noqa
    fake_workflows_dir: str,  # noqa
):
    workflow = Workflow.build(
        get_fake_workflow_path("custom_indices_structure"),
        fake_ops_dir,
        fake_workflows_dir,
    )

    data = StacConverter().to_stac_item([THE_DATAVIBE])
    wf_input: OpIOType = {"user_input": serialize_stac(data)}

    runner = MockWorkflowRunner(
        fail_list=["ndvi"],
        workflow=workflow,
        io_mapper=WorkflowIOHandler(workflow),
    )

    with pytest.raises(RuntimeError):
        await runner.run(wf_input, uuid4())
