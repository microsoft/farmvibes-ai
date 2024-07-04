import json
import sys
import traceback as tb
from typing import Callable, List, Type, cast
from unittest.mock import patch

import pytest
from dapr.conf import settings

from vibe_common.constants import PUBSUB_URL_TEMPLATE
from vibe_common.messaging import (
    ErrorContent,
    ExecuteReplyContent,
    ExecuteRequestContent,
    MessageHeader,
    MessageType,
    OperationSpec,
    OpStatusType,
    WorkMessage,
    build_work_message,
    decode,
    encode,
    operation_spec_serializer,
    run_id_from_traceparent,
    send,
)
from vibe_common.schemas import CacheInfo
from vibe_core.data import TypeDictVibe
from vibe_dev.testing.workflow_fixtures import SimpleStrDataType


@pytest.fixture
def message_header(traceparent: str) -> MessageHeader:
    header = MessageHeader(
        type=MessageType.execute_request, run_id=run_id_from_traceparent(traceparent)
    )
    return header


@pytest.fixture
def traceparent(workflow_execution_message: WorkMessage) -> str:
    return workflow_execution_message.id


@pytest.fixture
def execute_request_content(
    simple_op_spec: OperationSpec, SimpleStrData: Type[SimpleStrDataType]
) -> ExecuteRequestContent:
    data = SimpleStrData("some fake data")
    content = ExecuteRequestContent(
        input=TypeDictVibe({"user_input": {"data": data}}),  # type: ignore
        operation_spec=simple_op_spec,
    )
    return content


def test_workflow_message_construction(workflow_execution_message: ExecuteRequestContent):
    assert workflow_execution_message


def test_execute_request_message_construction(
    message_header: MessageHeader, traceparent: str, execute_request_content: ExecuteRequestContent
):
    build_work_message(
        header=message_header,
        content=execute_request_content,
        traceparent=traceparent,  # type: ignore
    )


def test_execute_reply_message_construction(message_header: MessageHeader, traceparent: str):
    content = ExecuteReplyContent(
        cache_info=CacheInfo("test_op", "1.0", {}, {}), status=OpStatusType.done, output={}
    )
    message_header.type = MessageType.execute_reply
    build_work_message(header=message_header, content=content, traceparent=traceparent)


def test_error_message_construction(message_header: MessageHeader, traceparent: str):
    try:
        1 / 0  # type: ignore
    except ZeroDivisionError:
        e, value, traceback = sys.exc_info()
    content = ErrorContent(
        status=OpStatusType.failed,
        ename=e.__name__,  # type: ignore
        evalue=str(e),  # type: ignore
        traceback=tb.format_tb(traceback),  # type: ignore
    )
    message_header.type = MessageType.error
    build_work_message(header=message_header, content=content, traceparent=traceparent)


@patch("requests.post")
def test_send_work_message(post: Callable[..., None], workflow_execution_message: WorkMessage):
    send(workflow_execution_message, "test", "fake", "fake")
    post.assert_called_with(
        PUBSUB_URL_TEMPLATE.format(
            cast(str, settings.DAPR_RUNTIME_HOST),
            cast(str, settings.DAPR_HTTP_PORT),
            "fake",
            "fake",
        ),
        json=workflow_execution_message.to_cloud_event("test"),
        headers={
            "Content-Type": "application/cloudevents+json",
            "traceparent": workflow_execution_message.id,
        },
    )


def test_operation_spec_serializer(execute_request_content: ExecuteRequestContent):
    spec = execute_request_content.operation_spec
    assert spec is not None
    out = operation_spec_serializer(spec)
    type_mapper = {
        "plain_input": "SimpleStrDataType",
        "list_input": "List[SimpleStrDataType]",
        "terravibes_input": "DataVibe",
        "terravibes_list": "List[DataVibe]",
    }
    for k, v in type_mapper.items():
        assert out["inputs_spec"][k] == v
    spec.inputs_spec["nested_list_input"] = List[List[SimpleStrDataType]]  # type: ignore
    with pytest.raises(ValueError):
        operation_spec_serializer(spec)


def test_encoder_decoder():
    messages = [
        "1, 2, 3, 4",
        "🤩😱🤷‍🤔🍎😜♾️🍔🤭😒😵‍",
        json.dumps(
            {
                "+♾️": float("+inf"),
                "-♾️": float("-inf"),
                "🦇👨": [float("nan") for _ in range(20)],
            }
        ),
    ]

    for message in messages:
        assert message == decode(encode(message))


def test_refuse_to_encode_message_with_invalid_values(workflow_execution_message: WorkMessage):
    invalid_values = (float("nan"), float("inf"), float("-inf"))

    for value in invalid_values:
        content = cast(ExecuteRequestContent, workflow_execution_message.content)
        content.input["plain_input"]["data"] = [{"a": value}]  # type: ignore
        with pytest.raises(ValueError):
            workflow_execution_message.to_cloud_event("test")
