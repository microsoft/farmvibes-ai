import json
import logging
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from enum import auto
from random import getrandbits
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Set,
    Type,
    Union,
    cast,
    get_args,
    get_type_hints,
    overload,
)
from uuid import UUID

import aiohttp
import requests
from cloudevents.sdk.event import v1
from dapr.clients.grpc._response import TopicEventResponse
from dapr.conf import settings
from fastapi_utils.enums import StrEnum
from pydantic import BaseModel as PyBaseModel
from pydantic import Field, ValidationError, validator
from pystac.item import Item

import vibe_common.telemetry as telemetry
from vibe_core.data.core_types import OpIOType
from vibe_core.data.utils import get_base_type, is_container_type, serialize_stac
from vibe_core.datamodel import decode, encode
from vibe_core.utils import get_input_ids

from .constants import (
    CACHE_PUBSUB_TOPIC,
    CONTROL_PUBSUB_TOPIC,
    PUBSUB_URL_TEMPLATE,
    STATUS_PUBSUB_TOPIC,
    TRACEPARENT_FLAGS,
    TRACEPARENT_STRING,
    WORKFLOW_REQUEST_PUBSUB_TOPIC,
)
from .dropdapr import TopicEventResponse as HttpTopicEventResponse
from .schemas import CacheInfo, OperationSpec

CLOUDEVENTS_JSON: Final[str] = "application/cloudevents+json"
OCTET_STREAM: Final[str] = "application/octet-stream"
MAXIMUM_MESSAGE_SIZE: Final[int] = 256 * 1024

MessageContent = Union[
    "AckContent",
    "CacheInfoExecuteRequestContent",
    "ExecuteRequestContent",
    "ExecuteReplyContent",
    "ErrorContent",
    "WorkflowExecutionContent",
    "EvictedReplyContent",
    "WorkflowCancellationContent",
    "WorkflowDeletionContent",
]
ValidVersion = Literal["1.0"]


class OpStatusType(StrEnum):
    done = auto()
    failed = auto()


class MessageType(StrEnum):
    ack = auto()
    cache_info_execute_request = auto()
    error = auto()
    execute_request = auto()
    execute_reply = auto()
    evicted_reply = auto()
    workflow_execution_request = auto()
    workflow_cancellation_request = auto()
    workflow_deletion_request = auto()


class BaseModel(PyBaseModel):
    class Config:
        json_encoders = {Item: serialize_stac}


class MessageHeader(BaseModel):
    type: MessageType
    run_id: UUID
    id: str = ""
    parent_id: str = ""
    current_trace_parent: str = ""
    version: ValidVersion = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)

    @validator("id", always=True)
    def set_id(cls, value: str, values: Dict[str, Any]):
        return value or gen_traceparent(values["run_id"])


class ExecuteRequestContent(BaseModel):
    input: OpIOType
    operation_spec: OperationSpec

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"(operation_spec={self.operation_spec}, "
            f"input={get_input_ids(self.input)})"
        )


class CacheInfoExecuteRequestContent(ExecuteRequestContent):
    cache_info: CacheInfo

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"(operation_spec={self.operation_spec}, "
            f"input={get_input_ids(self.input)}, "
            f"cache_info={self.cache_info})"
        )


class ExecuteReplyContent(BaseModel):
    cache_info: CacheInfo
    status: OpStatusType
    output: OpIOType


class AckContent(BaseModel):
    pass


class EvictedReplyContent(BaseModel):
    pass


class ErrorContent(BaseModel):
    status: OpStatusType
    ename: str
    evalue: str
    traceback: List[str]


class WorkflowExecutionContent(BaseModel):
    input: OpIOType
    workflow: Dict[str, Any]
    parameters: Optional[Dict[str, Any]]

    def __str__(self):
        return (
            f"{self.__class__.__name__}(workflow={self.workflow}, parameters={self.parameters}, "
            f"input={get_input_ids(self.input)})"
        )


class WorkflowCancellationContent(BaseModel):
    pass


class WorkflowDeletionContent(BaseModel):
    pass


class BaseMessage(BaseModel):
    header: MessageHeader
    content: MessageContent
    _supported_channels: Set[str]

    class Config:
        # VibeType is not JSON serializable, so we need to convert
        # it to string, and convert it back when we receive the
        # message
        json_encoders = {OperationSpec: lambda x: operation_spec_serializer(x)}  # type: ignore

    def is_valid_for_channel(self, channel: str):
        return channel in self._supported_channels

    @property
    def id(self):
        return self.header.id

    @property
    def parent_id(self):
        return self.header.parent_id

    @property
    def run_id(self):
        return self.header.run_id

    @property
    def current_trace_parent(self):
        return self.header.current_trace_parent

    def update_current_trace_parent(self):
        self.header.current_trace_parent = telemetry.get_current_trace_parent()

    @validator("content")
    def validate_content(cls, value: MessageContent, values: Dict[str, MessageHeader]):
        type: MessageType = values["header"].type
        if not isinstance(value, MESSAGE_TYPE_TO_CONTENT_TYPE[type]):
            raise ValueError(
                f"Message of type {type} doesn't specify content of correct type "
                f"({MESSAGE_TYPE_TO_CONTENT_TYPE[type]})"
            )

        if isinstance(value, ExecuteRequestContent) and value.operation_spec is None:
            raise ValueError("Operation execution content requires an operation_spec")
        return value

    def to_cloud_event(self, source: str) -> Dict[str, Any]:
        """Converts this message to a CloudEvents 1.0 dict representation.

        Params:
            source: str
                From the spec: The "source" is the context in which the
                occurrence happened. We should use the name of the TerraVibes
                component that created this message.

        For details, please see the specification at
        https://github.com/cloudevents/spec/blob/v1.0/spec.md
        """

        return {
            "specversion": "1.0",
            "datacontenttype": CLOUDEVENTS_JSON,
            "type": f"ai.terravibes.work.{self.header.type}",
            "source": source,
            "data": encode(self.json(allow_nan=False)),
            "time": datetime.now().isoformat(timespec="seconds") + "Z",  # RFC3339 time
            "subject": f"{self.header.type}-{self.header.id}",
            "id": self.id,
            "traceparent": self.id,
            "traceid": self.id,
        }


class CacheInfoExecuteRequestMessage(BaseMessage):
    _supported_channels: Set[str] = {CONTROL_PUBSUB_TOPIC}
    content: ExecuteRequestContent


class ExecuteRequestMessage(BaseMessage):
    _supported_channels: Set[str] = {CACHE_PUBSUB_TOPIC}
    content: ExecuteRequestContent


class ExecuteReplyMessage(BaseMessage):
    _supported_channels: Set[str] = {STATUS_PUBSUB_TOPIC}
    content: ExecuteReplyContent


class EvictedReplyMessage(BaseMessage):
    _supported_channels: Set[str] = {STATUS_PUBSUB_TOPIC}
    content: EvictedReplyContent


class ErrorMessage(BaseMessage):
    _supported_channels: Set[str] = {STATUS_PUBSUB_TOPIC}
    content: ErrorContent


class WorkflowDeletionMessage(BaseMessage):
    _supported_channels: Set[str] = {WORKFLOW_REQUEST_PUBSUB_TOPIC}
    content: WorkflowDeletionContent


class WorkflowExecutionMessage(BaseMessage):
    _supported_channels: Set[str] = {WORKFLOW_REQUEST_PUBSUB_TOPIC}
    content: WorkflowExecutionContent


class WorkflowCancellationMessage(BaseMessage):
    _supported_channels: Set[str] = {WORKFLOW_REQUEST_PUBSUB_TOPIC}
    content: WorkflowCancellationContent


class AckMessage(BaseMessage):
    _supported_channels: Set[str] = {STATUS_PUBSUB_TOPIC}
    content: AckContent


WorkMessage = Union[
    AckMessage,
    CacheInfoExecuteRequestMessage,
    ExecuteRequestMessage,
    ExecuteReplyMessage,
    EvictedReplyMessage,
    ErrorMessage,
    WorkflowExecutionMessage,
    WorkflowCancellationMessage,
    WorkflowDeletionMessage,
]


class WorkMessageBuilder:
    @staticmethod
    def build_execute_request(
        run_id: UUID,
        traceparent: str,
        op_spec: OperationSpec,
        input: OpIOType,
    ) -> WorkMessage:
        header = MessageHeader(
            type=MessageType.execute_request,
            run_id=run_id,
            parent_id=traceparent,
        )
        content = ExecuteRequestContent(input=input, operation_spec=op_spec)
        return ExecuteRequestMessage(header=header, content=content)

    @staticmethod
    def add_cache_info_to_execute_request(
        execute_request_message: ExecuteRequestMessage, cache_info: CacheInfo
    ) -> WorkMessage:
        header = execute_request_message.header
        header.type = MessageType.cache_info_execute_request
        content = CacheInfoExecuteRequestContent(
            input=execute_request_message.content.input,
            operation_spec=execute_request_message.content.operation_spec,
            cache_info=cache_info,
        )
        return CacheInfoExecuteRequestMessage(header=header, content=content)

    @staticmethod
    def build_workflow_request(
        run_id: UUID,
        workflow: Dict[str, Any],
        parameters: Optional[Dict[str, Any]],
        input: OpIOType,
    ) -> WorkMessage:
        header = MessageHeader(type=MessageType.workflow_execution_request, run_id=run_id)
        content = WorkflowExecutionContent(input=input, workflow=workflow, parameters=parameters)
        return WorkflowExecutionMessage(header=header, content=content)

    @staticmethod
    def build_workflow_cancellation(run_id: UUID) -> WorkMessage:
        header = MessageHeader(type=MessageType.workflow_cancellation_request, run_id=run_id)
        content = WorkflowCancellationContent()
        return WorkflowCancellationMessage(header=header, content=content)

    @staticmethod
    def build_workflow_deletion(run_id: UUID) -> WorkMessage:
        header = MessageHeader(type=MessageType.workflow_deletion_request, run_id=run_id)
        content = WorkflowDeletionContent()
        return WorkflowDeletionMessage(header=header, content=content)

    @staticmethod
    def build_execute_reply(
        traceparent: str, cache_info: CacheInfo, output: OpIOType
    ) -> WorkMessage:
        run_id = run_id_from_traceparent(traceparent)
        header = MessageHeader(type=MessageType.execute_reply, run_id=run_id, parent_id=traceparent)
        content = ExecuteReplyContent(
            cache_info=cache_info, status=OpStatusType.done, output=output
        )
        return ExecuteReplyMessage(header=header, content=content)

    @staticmethod
    def build_error(traceparent: str, ename: str, evalue: str, traceback: List[str]) -> WorkMessage:
        run_id = run_id_from_traceparent(traceparent)
        header = MessageHeader(type=MessageType.error, run_id=run_id, parent_id=traceparent)
        content = ErrorContent(
            status=OpStatusType.failed, ename=ename, evalue=evalue, traceback=traceback
        )
        return ErrorMessage(header=header, content=content)

    @staticmethod
    def build_evicted_reply(traceparent: str) -> WorkMessage:
        run_id = run_id_from_traceparent(traceparent)
        header = MessageHeader(type=MessageType.evicted_reply, run_id=run_id, parent_id=traceparent)
        content = EvictedReplyContent()
        return EvictedReplyMessage(header=header, content=content)

    @staticmethod
    def build_ack_reply(traceparent: str) -> WorkMessage:
        run_id = run_id_from_traceparent(traceparent)
        header = MessageHeader(type=MessageType.ack, run_id=run_id, parent_id=traceparent)
        content = AckContent()
        return AckMessage(header=header, content=content)


MESSAGE_TYPE_TO_CONTENT_TYPE: Dict[MessageType, Type[MessageContent]] = {
    MessageType.ack: AckContent,
    MessageType.cache_info_execute_request: CacheInfoExecuteRequestContent,
    MessageType.error: ErrorContent,
    MessageType.evicted_reply: EvictedReplyContent,
    MessageType.execute_reply: ExecuteReplyContent,
    MessageType.execute_request: ExecuteRequestContent,
    MessageType.workflow_execution_request: WorkflowExecutionContent,
    MessageType.workflow_cancellation_request: WorkflowCancellationContent,
    MessageType.workflow_deletion_request: WorkflowDeletionContent,
}


def build_work_message(
    header: MessageHeader, content: MessageContent, traceparent: Optional[str] = None
) -> WorkMessage:
    error = None
    for cls in get_args(WorkMessage):
        try:
            ret = cls(header=header, content=content)
            if traceparent is not None:
                ret.header.parent_id = traceparent
            return ret
        except ValidationError as e:
            error = e
    assert error is not None
    raise error


def extract_event_data(event: v1.Event) -> Dict[str, Any]:
    logger = logging.getLogger(f"{__name__}.extract_event_data")
    if not isinstance(event.data, (bytes, str)):
        logger.error("Received data is not a byte stream nor a string.")
        raise ValueError("Unable to decode event data {event.data}")
    try:
        # dapr tries to encode our already-encoded string
        data = json.loads(decode(json.loads(event.data)))
    except json.decoder.JSONDecodeError:
        data = json.loads(
            decode(event.data if isinstance(event.data, str) else event.data.decode())
        )

    return data


def event_to_work_message(event: v1.Event) -> WorkMessage:
    data = extract_event_data(event)
    header = MessageHeader(**data["header"])
    content = MESSAGE_TYPE_TO_CONTENT_TYPE[header.type](**data["content"])
    return build_work_message(header, content)


def extract_message_header_from_event(event: v1.Event) -> MessageHeader:
    extracted_data = extract_event_data(event)
    return MessageHeader(**extracted_data["header"])


def send(message: WorkMessage, source: str, pubsubname: str, topic: str) -> bool:
    message.update_current_trace_parent()
    logger = logging.getLogger(f"{__name__}.send")
    try:
        logger.debug(
            f"Sending message with header {message.header} from "
            f"{source} to pubsub {pubsubname}, topic {topic}"
        )
        response = requests.post(
            PUBSUB_URL_TEMPLATE.format(
                cast(str, settings.DAPR_RUNTIME_HOST),
                cast(str, settings.DAPR_HTTP_PORT),
                pubsubname,
                topic,
            ),
            json=message.to_cloud_event(source),
            headers={
                "Content-Type": CLOUDEVENTS_JSON,
                "traceparent": message.id,
            },
        )
        request_body_length = (
            len(response.request.body) if response.request and response.request.body else 0
        )
        logger.debug(
            f"Last request to pubsub {pubsubname} topic {topic} had "
            f"status code {response.status_code} and body length {request_body_length} bytes"
        )
        if request_body_length > MAXIMUM_MESSAGE_SIZE:
            logger.warning(
                f"Last request to pubsub {pubsubname} topic {topic} exceeded "
                f"maximum safe message size of {MAXIMUM_MESSAGE_SIZE} bytes. "
                f"The message might have been dropped by the message broker."
            )
        return response.ok
    except Exception:
        logger.exception(
            f"Failed to send payload {message} from {source} to pubsub {pubsubname}, topic {topic}"
        )
        raise


async def send_async(message: WorkMessage, source: str, pubsubname: str, topic: str) -> bool:
    message.update_current_trace_parent()
    logger = logging.getLogger(f"{__name__}.send_async")
    try:
        logger.debug(
            f"Sending async message with header {message.header} from "
            f"{source} to pubsub {pubsubname}, topic {topic}"
        )
        async with aiohttp.ClientSession() as session:
            payload = message.to_cloud_event(source)
            async with await session.post(
                PUBSUB_URL_TEMPLATE.format(
                    cast(str, settings.DAPR_RUNTIME_HOST),
                    cast(str, settings.DAPR_HTTP_PORT),
                    pubsubname,
                    topic,
                ),
                json=payload,
                headers={
                    "Content-Type": CLOUDEVENTS_JSON,
                    "traceparent": message.id,
                },
            ) as response:
                request_body_length = json.dumps(payload).encode("utf-8").__len__()
                logger.debug(
                    f"Last request to pubsub {pubsubname} topic {topic} had "
                    f"status code {response.status} and body length {request_body_length} bytes"
                )
                if request_body_length > MAXIMUM_MESSAGE_SIZE:
                    logger.warning(
                        f"Last request to pubsub {pubsubname} topic {topic} exceeded "
                        f"maximum safe message size of {MAXIMUM_MESSAGE_SIZE} bytes. "
                        f"The message might have been dropped by the message broker."
                    )
                return response.ok
    except Exception:
        logger.exception(
            f"Failed to send payload {message} from {source} to pubsub {pubsubname}, topic {topic}"
        )
        raise


def operation_spec_serializer(spec: OperationSpec) -> Dict[str, Any]:
    opdict = asdict(spec)
    for field in "inputs_spec output_spec".split():
        if field not in opdict:
            continue
        for k, v in opdict[field].items():
            if is_container_type(v):
                base = get_base_type(v)
                v = f"List[{base.__name__}]"
            else:
                v = get_base_type(v).__name__
            opdict[field][k] = str(v)
    return opdict


def gen_traceparent(run_id: UUID) -> str:
    """Generates a unique identifier that can be used as W3C traceparent header.

    See https://www.w3.org/TR/trace-context/#examples-of-http-traceparent-headers for examples.
    """
    trace_id = int(run_id.hex, 16)
    parent_id = getrandbits(64)

    return TRACEPARENT_STRING.format(
        trace_id=trace_id, parent_id=parent_id, trace_flags=TRACEPARENT_FLAGS
    )


def run_id_from_traceparent(traceparent: str) -> UUID:
    """Given the contents of a TerraVibes traceparent header, extracts a run_id from it."""

    return UUID(traceparent.split("-")[1])


@overload
def accept_or_fail_event(
    event: v1.Event,
    success_callback: Callable[[WorkMessage], HttpTopicEventResponse],
    failure_callback: Callable[[v1.Event, Exception, List[str]], HttpTopicEventResponse],
) -> HttpTopicEventResponse: ...


@overload
def accept_or_fail_event(
    event: v1.Event,
    success_callback: Callable[[WorkMessage], TopicEventResponse],
    failure_callback: Callable[[v1.Event, Exception, List[str]], TopicEventResponse],
) -> TopicEventResponse: ...


def accept_or_fail_event(
    event: v1.Event,
    success_callback: Callable[[WorkMessage], Union[HttpTopicEventResponse, TopicEventResponse]],
    failure_callback: Callable[
        [v1.Event, Exception, List[str]], Union[HttpTopicEventResponse, TopicEventResponse]
    ],
):
    logger = logging.getLogger(f"{__name__}.accept_or_fail_event")
    try:
        message = event_to_work_message(event)
        logger.info(f"Received message: header={message.header}")
        return success_callback(message)
    except Exception as e:
        _, _, exc_traceback = sys.exc_info()
        logger.exception(f"Failed to process event with id {event.id}")
        try:
            return failure_callback(event, e, traceback.format_tb(exc_traceback))
        except Exception:
            logger.error(f"Unable to parse traceparent. Discarding event with id {event.id}")

            ResponseType = get_type_hints(success_callback).get("return", HttpTopicEventResponse)
            return ResponseType("drop")


@overload
async def accept_or_fail_event_async(
    event: v1.Event,
    success_callback: Callable[[WorkMessage], Awaitable[HttpTopicEventResponse]],
    failure_callback: Callable[[v1.Event, Exception, List[str]], Awaitable[HttpTopicEventResponse]],
) -> HttpTopicEventResponse: ...


@overload
async def accept_or_fail_event_async(
    event: v1.Event,
    success_callback: Callable[[WorkMessage], Awaitable[TopicEventResponse]],
    failure_callback: Callable[[v1.Event, Exception, List[str]], Awaitable[TopicEventResponse]],
) -> TopicEventResponse: ...


async def accept_or_fail_event_async(
    event: v1.Event,
    success_callback: Callable[
        [WorkMessage], Awaitable[Union[HttpTopicEventResponse, TopicEventResponse]]
    ],
    failure_callback: Callable[
        [v1.Event, Exception, List[str]],
        Awaitable[Union[HttpTopicEventResponse, TopicEventResponse]],
    ],
):
    logger = logging.getLogger(f"{__name__}.accept_or_fail_event_async")
    try:
        message = event_to_work_message(event)
        logger.info(f"Received message: header={message.header}")
        return await success_callback(message)
    except Exception as e:
        _, _, exc_traceback = sys.exc_info()
        logger.exception(f"Failed to process event with id {event.id}")
        try:
            return await failure_callback(event, e, traceback.format_tb(exc_traceback))
        except Exception:
            logger.error(f"Unable to parse traceparent. Discarding event with id {event.id}")

            ResponseType = get_type_hints(success_callback).get("return", HttpTopicEventResponse)
            return ResponseType("drop")
