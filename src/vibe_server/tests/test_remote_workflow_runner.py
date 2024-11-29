# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import traceback
from asyncio.queues import Queue
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, cast
from unittest.mock import AsyncMock, patch

import pydantic
import pytest
from shapely.geometry import Polygon, mapping

from vibe_common.input_handlers import gen_stac_item_from_bounds
from vibe_common.messaging import (
    ErrorContent,
    ExecuteReplyContent,
    ExecuteRequestContent,
    MessageHeader,
    MessageType,
    OpStatusType,
    WorkMessage,
    build_work_message,
)
from vibe_common.schemas import CacheInfo, EntryPointDict, OperationSpec
from vibe_core.data import TypeDictVibe
from vibe_core.data.core_types import OpIOType
from vibe_core.data.utils import is_vibe_list
from vibe_core.datamodel import TaskDescription
from vibe_dev.testing.fake_workflows_fixtures import get_fake_workflow_path
from vibe_server.workflow.runner.remote_runner import (
    MessageRouter,
    RemoteWorkflowRunner,
    WorkMessageBuilder,
)
from vibe_server.workflow.runner.task_io_handler import WorkflowIOHandler
from vibe_server.workflow.workflow import Workflow

HERE = os.path.dirname(os.path.abspath(__file__))


class FakeMessage(str):
    def __init__(self, s: str):
        self.parent_id = ""
        self.msg = s

    def __str__(self):
        return self.msg


@pytest.fixture
def time_range() -> Tuple[datetime, datetime]:
    return (
        datetime(year=2021, month=2, day=1, tzinfo=timezone.utc),
        datetime(year=2021, month=2, day=11, tzinfo=timezone.utc),
    )


@pytest.fixture
def input_polygon() -> Polygon:
    polygon_coords = [
        (-88.062073563448919, 37.081397673802059),
        (-88.026349330507315, 37.085463858128762),
        (-88.026349330507315, 37.085463858128762),
        (-88.012445388773259, 37.069230099135126),
        (-88.035931592028305, 37.048441375086092),
        (-88.068120429075847, 37.058833638440767),
        (-88.062073563448919, 37.081397673802059),
    ]

    return Polygon(polygon_coords)


@pytest.fixture
def helloworld_input(input_polygon: Polygon, time_range: Tuple[datetime, datetime]):
    return gen_stac_item_from_bounds(mapping(input_polygon), time_range[0], time_range[1])


def test_work_message_builder_fails(workflow_execution_message: WorkMessage):
    if hasattr(pydantic, "error_wrappers"):
        ValidationError = pydantic.error_wrappers.ValidationError  # type: ignore
    else:
        ValidationError = pydantic.ValidationError  # type: ignore
    with pytest.raises(ValidationError):
        WorkMessageBuilder.build_execute_request(
            workflow_execution_message.header.run_id,
            "",
            None,  # type: ignore
            {},
        )


def test_work_message_builder_succeeds_with_op_spec(workflow_execution_message: WorkMessage):
    message = WorkMessageBuilder.build_execute_request(
        workflow_execution_message.header.run_id,
        "",
        OperationSpec(
            name="fake",
            root_folder="/tmp",
            inputs_spec=TypeDictVibe({}),
            output_spec=TypeDictVibe({}),
            entrypoint=EntryPointDict(file="op.py", callback_builder="whatever"),
            description=TaskDescription(),
        ),
        {},
    )
    assert cast(ExecuteRequestContent, message.content).operation_spec


@pytest.mark.anyio
async def test_message_router_put():
    inqueue = Queue()
    handler = MessageRouter(inqueue)
    item = FakeMessage("some really cool item")
    await inqueue.put(item)
    assert await handler.get("") == item


@pytest.mark.anyio
async def test_message_router_len():
    inqueue = Queue()
    handler = MessageRouter(inqueue)
    assert len(handler) == 0
    for i in range(10):
        await inqueue.put(FakeMessage(f"{i}"))
    assert len(handler) == 10
    handler.should_stop = True


def build_reply(
    parent_header: MessageHeader, op: Optional[OperationSpec] = None, failure: bool = False
) -> WorkMessage:
    if op is None:
        output = {}
    else:
        output = {
            k: ([{"a": 1}] if is_vibe_list(op.output_spec[k]) else {"a": 1}) for k in op.output_spec
        }
    if failure:
        try:
            1 / 0  # type: ignore
        except ZeroDivisionError:
            ename, evalue, tb = sys.exc_info()
        content = ErrorContent(
            status=OpStatusType.failed,
            ename=str(ename),  # type: ignore
            evalue=str(evalue),  # type: ignore
            traceback=traceback.format_tb(tb),  # type: ignore
        )
    else:
        content = ExecuteReplyContent(
            cache_info=CacheInfo("test_op", "1.0", {}, {}),
            status=OpStatusType.done,
            output=output,  # type: ignore
        )
    header = MessageHeader(
        type=MessageType.error if failure else MessageType.execute_reply,
        run_id=parent_header.run_id,
        parent_id=parent_header.id,
    )
    return build_work_message(header=header, content=content)


async def workflow_callback(change, **kwargs):  # type: ignore
    print(change, kwargs)  # type: ignore


@patch("vibe_server.workflow.runner.remote_runner.send_async")
@pytest.mark.anyio
async def test_remote_workflow_runner_runs(
    send_async: AsyncMock,
    fake_ops_dir: str,
    fake_workflows_dir: str,
    helloworld_input: OpIOType,
    workflow_execution_message: WorkMessage,
):
    inqueue: "Queue[WorkMessage]" = Queue()
    handler = MessageRouter(inqueue)
    workflow = Workflow.build(get_fake_workflow_path("str_input"), fake_ops_dir, fake_workflows_dir)
    io_mapper = WorkflowIOHandler(workflow)
    runner = RemoteWorkflowRunner(
        handler,
        workflow,
        workflow_execution_message.id,
        pubsubname="",
        source="",
        topic="",
        io_mapper=io_mapper,
        update_state_callback=workflow_callback,
    )

    async def patched_send(item: WorkMessage, *args: Any) -> None:
        reply = build_reply(
            parent_header=item.header, op=cast(ExecuteRequestContent, item.content).operation_spec
        )
        await inqueue.put(reply)

    send_async.side_effect = patched_send

    await runner.run(
        {k: helloworld_input for k in runner.workflow.inputs_spec},
        workflow_execution_message.header.run_id,
    )


@patch("vibe_server.workflow.runner.remote_runner.send_async")
@pytest.mark.anyio
async def test_remote_workflow_runner_fails(
    send_async: AsyncMock,
    fake_ops_dir: str,
    fake_workflows_dir: str,
    helloworld_input: OpIOType,
    workflow_execution_message: WorkMessage,
):
    inqueue: "Queue[WorkMessage]" = Queue()
    handler = MessageRouter(inqueue)
    workflow = Workflow.build(get_fake_workflow_path("str_input"), fake_ops_dir, fake_workflows_dir)
    io_mapper = WorkflowIOHandler(workflow)
    runner = RemoteWorkflowRunner(
        handler,
        workflow,
        workflow_execution_message.id,
        pubsubname="",
        source="",
        topic="",
        io_mapper=io_mapper,
        update_state_callback=workflow_callback,
    )

    async def patched_send(item: WorkMessage, *args: Any) -> None:
        reply = build_reply(item.header, None, True)
        await inqueue.put(reply)

    send_async.side_effect = patched_send

    with pytest.raises(RuntimeError):
        await runner.run(
            {k: helloworld_input for k in runner.workflow.inputs_spec},
            workflow_execution_message.header.run_id,
        )
