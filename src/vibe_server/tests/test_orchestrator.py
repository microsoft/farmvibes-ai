# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from asyncio.queues import Queue
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, cast
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID
from uuid import uuid4 as uuid

import pytest
from cloudevents.sdk.event import v1

from vibe_common.constants import STATUS_PUBSUB_TOPIC, WORKFLOW_REQUEST_PUBSUB_TOPIC
from vibe_common.dropdapr import TopicEventResponseStatus
from vibe_common.messaging import (
    ErrorContent,
    ExecuteReplyContent,
    MessageHeader,
    MessageType,
    OpStatusType,
    WorkflowExecutionContent,
    WorkflowExecutionMessage,
    WorkMessage,
    WorkMessageBuilder,
    build_work_message,
    encode,
    gen_traceparent,
)
from vibe_common.schemas import CacheInfo
from vibe_common.statestore import StateStore
from vibe_core.data.core_types import OpIOType
from vibe_core.data.json_converter import dump_to_json
from vibe_core.data.utils import StacConverter, is_container_type, serialize_stac
from vibe_core.datamodel import RunConfig, RunDetails, RunStatus, SpatioTemporalJson
from vibe_dev.testing.fake_workflows_fixtures import get_fake_workflow_path  # noqa
from vibe_dev.testing.workflow_fixtures import THE_DATAVIBE
from vibe_server.orchestrator import Orchestrator, WorkflowRunManager
from vibe_server.workflow.runner import WorkflowChange
from vibe_server.workflow.runner.remote_runner import RemoteWorkflowRunner
from vibe_server.workflow.spec_parser import WorkflowParser
from vibe_server.workflow.workflow import GraphNodeType, Workflow


def make_test_message(
    workflow_name: str,
    params: Optional[Dict[str, Any]],
    fake_ops_dir: str,  # noqa
    fake_workflows_dir: str,  # noqa
) -> WorkflowExecutionMessage:
    header = MessageHeader(
        type=MessageType.workflow_execution_request,
        run_id=uuid(),
    )
    workflow_dict = asdict(
        WorkflowParser.parse(
            get_fake_workflow_path(workflow_name),
            ops_dir=fake_ops_dir,
            workflows_dir=fake_workflows_dir,
            parameters_override=params,
        )
    )
    content = WorkflowExecutionContent(
        input={},
        workflow=workflow_dict,
        parameters=params,
    )
    return cast(WorkflowExecutionMessage, build_work_message(header, content))


@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_orchestrator_add_output(store: Mock, retrieve: Mock, run_config: Dict[str, Any]):
    retrieve.side_effect = lambda _: run_config
    output = cast(OpIOType, {"some-op": {"data": "fake"}})
    statestore = StateStore()
    await WorkflowRunManager.add_output_to_run(run_config["id"], output, statestore)
    run_config["output"] = encode(dump_to_json(output))
    store.assert_called_with(run_config["id"], RunConfig(**run_config))


@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_orchestrator_fail_workflow(store: Mock, retrieve: Mock, run_config: Dict[str, Any]):
    retrieve.side_effect = lambda _: run_config
    orchestrator = Orchestrator()
    reason = "fake reason"
    await orchestrator.fail_workflow(run_config["id"], reason)
    run_config["details"]["status"] = RunStatus.failed
    run_config["details"]["reason"] = reason
    assert store.mock_calls[0][1][1].details.status == RunStatus.failed
    assert store.mock_calls[0][1][1].details.reason == reason


def to_cloud_event(msg: WorkMessage) -> v1.Event:
    ce = v1.Event()
    msgdict = msg.to_cloud_event("test")
    for key in msgdict:
        if hasattr(ce, key):
            try:
                setattr(ce, key, msgdict[key])
            except Exception:
                pass
    ce.data = ce.data.encode("ascii")  # type: ignore
    return ce


def test_run_config_fails_on_invalid_inputs():
    rc = RunConfig(
        name="name",
        workflow="fake",
        parameters=None,
        user_input=SpatioTemporalJson(
            datetime.now(),
            datetime.now(),
            {},
        ),
        id=uuid(),
        details=RunDetails(status=RunStatus.pending, start_time=None, end_time=None, reason=None),
        task_details={},
        spatio_temporal_json=None,
    )
    for value in float("nan"), float("inf"), float("-inf"):
        with pytest.raises(ValueError):
            rc.set_output({"a": value})  # type: ignore


@pytest.mark.anyio
async def test_orchestrator_update_response():
    reply_content = ExecuteReplyContent(
        cache_info=CacheInfo("test_op", "1.0", {}, {}), status=OpStatusType.done, output={}
    )
    header = MessageHeader(type=MessageType.execute_reply, run_id=uuid())
    reply = build_work_message(header=header, content=reply_content)
    orchestrator = Orchestrator()
    orchestrator.inqueues[str(header.run_id)] = Queue()
    topic_reply = await orchestrator.handle_update_workflow_status(
        STATUS_PUBSUB_TOPIC, to_cloud_event(reply)
    )
    assert topic_reply.status == TopicEventResponseStatus.success["status"]


@pytest.mark.anyio
async def test_orchestrator_update_error_response():
    reply_content = ErrorContent(status=OpStatusType.failed, ename="", evalue="", traceback=[])
    header = MessageHeader(type=MessageType.error, run_id=uuid())
    reply = build_work_message(header=header, content=reply_content)
    orchestrator = Orchestrator()
    orchestrator.inqueues[str(header.run_id)] = Queue()
    topic_reply = await orchestrator.handle_update_workflow_status(
        STATUS_PUBSUB_TOPIC, to_cloud_event(reply)
    )
    assert topic_reply.status == TopicEventResponseStatus.success["status"]


@pytest.mark.anyio
async def test_orchestrator_update_response_fails_as_message_not_in_queue():
    orchestrator = Orchestrator()
    ack_reply = WorkMessageBuilder.build_ack_reply(gen_traceparent(uuid()))
    topic_reply = await orchestrator.handle_update_workflow_status(
        STATUS_PUBSUB_TOPIC, to_cloud_event(ack_reply)
    )
    assert topic_reply.status == TopicEventResponseStatus.drop["status"]


@pytest.mark.anyio
async def test_orchestrator_update_response_fails_with_invalid_message(
    workflow_execution_message: WorkMessage,
):
    orchestrator = Orchestrator()
    topic_reply = await orchestrator.handle_update_workflow_status(
        STATUS_PUBSUB_TOPIC, to_cloud_event(workflow_execution_message)
    )
    assert topic_reply.status == TopicEventResponseStatus.drop["status"]


@pytest.mark.anyio
async def test_orchestrator_workflow_submission_rejects():
    request = WorkMessageBuilder.build_error(gen_traceparent(uuid()), "", "", [])
    orchestrator = Orchestrator()
    topic_reply = await orchestrator.handle_manage_workflow_event(
        WORKFLOW_REQUEST_PUBSUB_TOPIC, to_cloud_event(request)
    )
    assert topic_reply.status == TopicEventResponseStatus.drop["status"]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.anyio
async def test_orchestrator_workflow_submission_accepts():
    spec = WorkflowParser._load_workflow(get_fake_workflow_path("item_gather"))
    request = WorkMessageBuilder.build_workflow_request(uuid(), spec, {}, {})
    orchestrator = Orchestrator()
    reply = await orchestrator.handle_manage_workflow_event(
        WORKFLOW_REQUEST_PUBSUB_TOPIC, to_cloud_event(request)
    )
    assert reply.status == TopicEventResponseStatus.success["status"]


@patch("vibe_common.statestore.StateStore.retrieve_bulk")
@patch("vibe_common.statestore.StateStore.retrieve")
@pytest.mark.anyio
async def test_orchestrator_startup_sees_no_runs(retrieve: Mock, retrieve_bulk: Mock):
    retrieve.return_value = []
    retrieve_bulk.return_value = []
    orchestrator = Orchestrator()
    assert await orchestrator.get_unfinished_workflows() == []
    retrieve_bulk.assert_called_once_with([])


@patch("vibe_common.statestore.StateStore.retrieve")
@pytest.mark.anyio
async def test_orchestrator_startup_dapr_not_stared(retrieve: Mock):
    retrieve.side_effect = Exception("Random error when retrieving runs")
    with pytest.raises(RuntimeError):
        orchestrator = Orchestrator()
        await orchestrator._resume_workflows()


@patch("vibe_common.statestore.StateStore.retrieve_bulk")
@patch("vibe_common.statestore.StateStore.retrieve")
@pytest.mark.anyio
async def test_orchestrator_startup_sees_no_unfinished_runs(
    retrieve: Mock, retrieve_bulk: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = [run_config["id"]]
    run_config["details"]["status"] = RunStatus.done
    retrieve_bulk.return_value = [run_config]
    orchestrator = Orchestrator()
    assert await orchestrator.get_unfinished_workflows() == []
    retrieve_bulk.assert_called_once_with([run_config["id"]])


@patch("vibe_common.statestore.StateStore.retrieve_bulk")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@patch("vibe_server.workflow.runner.task_io_handler.WorkflowIOHandler.map_output")
@patch("vibe_server.workflow.runner.task_io_handler.TaskIOHandler.retrieve_sinks")
@patch("vibe_server.workflow.runner.remote_runner.RemoteWorkflowRunner._run_ops")
@pytest.mark.anyio
async def test_orchestrator_startup_sees_unfinished_runs(
    _run_ops: AsyncMock,
    retrieve_sinks: Mock,
    map_output: Mock,
    store: Mock,
    retrieve: Mock,
    retrieve_bulk: Mock,
    run_config: Dict[str, Any],
    fake_ops_dir: str,
    fake_workflows_dir: str,
):
    first = True

    def retrieve_fun(_: str):
        nonlocal first
        if first:
            first = False
            return run_config["id"]
        return run_config

    _run_ops.return_value = None
    retrieve_sinks.return_value = None
    map_output.return_value = None
    retrieve.side_effect = retrieve_fun
    retrieve_bulk.return_value = [run_config, run_config, run_config]
    build_return_value = Workflow.build(
        get_fake_workflow_path("single_and_parallel"), fake_ops_dir, fake_workflows_dir
    )

    with patch("vibe_server.workflow.workflow.Workflow.build", return_value=build_return_value):
        orchestrator = Orchestrator()
        await orchestrator._resume_workflows()
        retrieve_bulk.assert_called_once_with(run_config["id"])
        _run_ops.assert_called()


@patch("vibe_server.orchestrator.WorkflowStateUpdate.__call__")
@pytest.mark.anyio
async def test_orchestrator_cancel_run(
    state_update: Mock,
    fake_ops_dir: str,  # noqa
    fake_workflows_dir: str,  # noqa
):
    workflow = Workflow.build(
        get_fake_workflow_path("str_input"),
        fake_ops_dir,
        fake_workflows_dir,
    )

    message = WorkMessageBuilder.build_workflow_request(
        uuid(),
        asdict(workflow.workflow_spec),
        None,
        {k: [{}] for k in workflow.inputs_spec},
    )

    cancellation = WorkMessageBuilder.build_workflow_cancellation(message.run_id)
    orchestrator = Orchestrator(ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)
    await orchestrator.manage_workflow(message)
    assert len(orchestrator._workflow_management_tasks.values()) == 1
    wf = list(orchestrator._workflow_management_tasks.values())[0]

    await orchestrator.manage_workflow(cancellation)
    await wf.task
    assert wf.is_cancelled
    assert wf.runner
    assert wf.runner.is_cancelled
    state_update.assert_any_call(WorkflowChange.WORKFLOW_CANCELLED)


@pytest.mark.parametrize("params", [None, {"new": "from_message"}])
@pytest.mark.anyio
async def test_build_workflow_with_params(
    fake_ops_dir: str,  # noqa
    fake_workflows_dir: str,  # noqa
    params: Optional[Dict[str, Any]],
):
    msg = make_test_message("resolve_params", params, fake_ops_dir, fake_workflows_dir)
    manager = WorkflowRunManager(
        None,  # type: ignore
        msg,
        1,  # type: ignore
        "",
        "",
        "",
        fake_ops_dir,
        fake_workflows_dir,
    )
    workflow, _ = manager.build_workflow({"input": None})  # type: ignore
    expected = workflow.workflow_spec.default_parameters["new"] if params is None else params["new"]
    assert workflow.workflow_spec.parameters["new"] == expected


@pytest.mark.parametrize(
    "wf_params", [("resolve_params", {"made_up": 1}), ("list_list", {"any": "!"})]
)
@patch("vibe_server.orchestrator.update_workflow")
@pytest.mark.anyio
async def test_build_workflow_invalid_params_update_status(
    update: Mock,
    wf_params: Tuple[str, Dict[str, Any]],
    fake_ops_dir: str,  # noqa
    fake_workflows_dir: str,  # noqa
):
    msg = make_test_message(
        wf_params[0], {}, fake_ops_dir=fake_ops_dir, fake_workflows_dir=fake_workflows_dir
    )
    msg.content.parameters = wf_params[1]
    manager = WorkflowRunManager(
        {},
        msg,
        1,  # type: ignore
        "",
        "",
        "",
        fake_ops_dir,
        fake_workflows_dir,  # type: ignore
    )
    with pytest.raises(ValueError):
        await manager.task
    update.assert_called_once()
    run_id, _, status, _ = update.call_args[0]
    assert run_id == str(msg.header.run_id)
    assert status == RunStatus.failed


@patch.object(RemoteWorkflowRunner, "_build_and_process_request", autospec=True)
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_run_workflow_that_will_fail(
    store: Mock,
    retrieve: Mock,
    bpr: Mock,
    fake_ops_dir: str,  # noqa
    fake_workflows_dir: str,  # noqa
    run_config: Dict[str, Any],
):
    converter = StacConverter()

    workflow = Workflow.build(
        get_fake_workflow_path("custom_indices_structure"),
        fake_ops_dir,
        fake_workflows_dir,
    )

    message = WorkMessageBuilder.build_workflow_request(
        uuid(),
        asdict(workflow.workflow_spec),
        None,
        {k: serialize_stac(converter.to_stac_item([THE_DATAVIBE])) for k in workflow.inputs_spec},
    )

    def mock_build_and_process_request(
        self: Any, op: GraphNodeType, input: OpIOType, run_id: UUID, subtask_idx: int
    ) -> OpIOType:
        self._handle_ack_message(op.name, subtask_idx)
        if op.name.startswith("ndvi"):
            raise RuntimeError("Received unsupported message error. Aborting execution.")
        return {
            k: serialize_stac(
                converter.to_stac_item(
                    # This should work just fine, as `DataVibe` inherits from `BaseVibe`,
                    # but pyright doesn't like it. I think the issue pyright is having
                    # is because we use `__init_subclass__` in a dataclass, and it is
                    # getting confused
                    [THE_DATAVIBE] if is_container_type(v) else THE_DATAVIBE  # type: ignore
                )
            )
            for k, v in op.spec.output_spec.items()
        }

    def store_side_effect(key: str, obj: Any, _: Optional[str] = None):  # type: ignore
        nonlocal run_config
        run_config = obj

    def retrieve_side_effect(key: str, _: Optional[str] = None):  # type: ignore
        return run_config

    store.side_effect = store_side_effect
    retrieve.side_effect = retrieve_side_effect
    bpr.side_effect = mock_build_and_process_request

    orchestrator = Orchestrator(ops_dir=fake_ops_dir, workflows_dir=fake_workflows_dir)

    with pytest.raises(RuntimeError):
        await orchestrator.manage_workflow(message)
        wf = list(orchestrator._workflow_management_tasks.values())[0]
        await wf.task

        assert run_config["details"]["status"] == RunStatus.failed
