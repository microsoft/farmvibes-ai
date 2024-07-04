import asyncio
import asyncio.queues
import logging
from collections import defaultdict
from typing import Any, Dict, List, NoReturn, Optional, TypeVar, cast
from uuid import UUID

from vibe_common.messaging import (
    ErrorContent,
    ExecuteReplyContent,
    ExecuteRequestMessage,
    MessageType,
    OperationSpec,
    WorkMessage,
    WorkMessageBuilder,
    send_async,
)
from vibe_common.telemetry import add_span_attributes, add_trace
from vibe_core.data.core_types import OpIOType

from ..workflow import GraphNodeType, Workflow
from .runner import (
    CancelledOpError,
    NoOpStateChange,
    WorkflowCallback,
    WorkflowChange,
    WorkflowRunner,
)

SLEEP_S = 0.2
RAISE_STR = "raise"
T = TypeVar("T")


class MessageRouter:
    def __init__(self, inqueue: "asyncio.queues.Queue[WorkMessage]"):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.message_map: Dict[str, "asyncio.queues.Queue[WorkMessage]"] = defaultdict(
            asyncio.queues.Queue
        )
        self.inqueue = inqueue
        self.should_stop = False
        self.task = asyncio.create_task(self.route_messages())

        def done_callback(task: Any) -> None:
            self.task = None
            try:
                maybe_exception = task.exception()
                if maybe_exception is not None:
                    self.logger.warning(
                        f"MessageRouter task {task} encountered an exception: {maybe_exception}"
                    )
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass

        self.task.add_done_callback(done_callback)

    async def route_messages(self):
        while not self.should_stop:
            try:
                msg = await asyncio.wait_for(self.inqueue.get(), timeout=SLEEP_S)
                self.message_map[msg.parent_id].put_nowait(msg)
                self.inqueue.task_done()
            except asyncio.TimeoutError:
                pass

    async def get(self, request_id: str, block: bool = True) -> WorkMessage:
        if block:
            msg = await self.message_map[request_id].get()
        else:
            msg = self.message_map[request_id].get_nowait()
        return msg

    def task_done(self, request_id: str) -> None:
        try:
            self.message_map[request_id].task_done()
        except ValueError:
            self.logger.exception(
                "task_done() called more times than there were items in the queue. "
                "This indicates a correctness issue and should be fixed. I'm ignoring "
                "it for now, though."
            )

    def clear(self) -> None:
        for queue in self.message_map.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    pass

    def __len__(self) -> int:
        return sum([q.qsize() for q in self.message_map.values()]) + self.inqueue.qsize()

    def __del__(self):
        if self.task and not self.task.done():
            self.task.cancel()
            self.task = None


class RemoteWorkflowRunner(WorkflowRunner):
    def __init__(
        self,
        message_router: "MessageRouter",
        workflow: Workflow,
        traceid: str,
        update_state_callback: WorkflowCallback = NoOpStateChange,
        pubsubname: Optional[str] = None,
        source: Optional[str] = None,
        topic: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            workflow=workflow,
            update_state_callback=update_state_callback,
            **kwargs,
        )
        self.topic = topic
        self.source = source
        self.pubsubname = pubsubname

        self.message_router = message_router
        self.traceid = traceid
        self.id_queue_map: Dict[str, "asyncio.queues.Queue[WorkMessage]"] = {}

    def _handle_failure(self, request: ExecuteRequestMessage, reply: WorkMessage) -> NoReturn:
        content = cast(ErrorContent, reply.content)
        root_idx = content.evalue.rfind(RAISE_STR)
        root_idx = root_idx + len(RAISE_STR) if root_idx != -1 else 0
        evalue = content.evalue[root_idx:]
        error = f"{content.ename}: {evalue}"
        self.logger.info(
            f"Operation {reply.id} failed with error {error}. (run id {reply.run_id})."
            f"Traceback: {content.traceback}"
        )
        raise RuntimeError(
            f"Failed to run op {request.content.operation_spec.name} in workflow run id "
            f"{reply.run_id} for input with message id {request.id}. Error description: {error}."
        )

    async def _handle_ack_message(self, op_name: str, subtask_idx: int) -> None:
        await self._report_state_change(
            WorkflowChange.SUBTASK_RUNNING, task=op_name, subtask_idx=subtask_idx
        )

    def _process_reply(self, request: WorkMessage, reply: WorkMessage) -> OpIOType:
        assert (
            reply.header.type != MessageType.execute_request
        ), f"Received invalid message {reply.id}"
        assert (
            reply.header.parent_id
        ), f"Received invalid reply {reply.id} with empty parent_id. (run id {reply.run_id})"
        if reply.header.type == MessageType.error:
            self._handle_failure(cast(ExecuteRequestMessage, request), reply)
        else:
            content = cast(ExecuteReplyContent, reply.content)
            self.logger.debug(
                f"Received execute reply for run id {reply.run_id} "
                f"(op name {content.cache_info.name}, op hash {content.cache_info.hash})."
            )
            return content.output

    async def _build_and_process_request(
        self, op: GraphNodeType, input: OpIOType, run_id: UUID, subtask_idx: int
    ) -> OpIOType:
        op_spec: OperationSpec = op.spec
        request: ExecuteRequestMessage = cast(
            ExecuteRequestMessage,
            WorkMessageBuilder.build_execute_request(
                run_id,
                self.traceid,
                op_spec,
                input,
            ),
        )

        failure_msg: str = (
            f"Failed to run op {op_spec.name} (subtask {subtask_idx})"
            f"with execution request id {request.id}, run id {run_id}."
        )
        if all([e is not None for e in (self.source, self.pubsubname, self.topic)]):
            await send_async(request, self.source, self.pubsubname, self.topic)  # type: ignore

        while True:
            if self.is_cancelled:
                raise CancelledOpError()

            try:
                reply = await self._wait_for_reply(request)
            except CancelledOpError:
                raise
            except Exception as e:
                raise RuntimeError(failure_msg) from e

            if reply.header.type == MessageType.ack:
                await self._handle_ack_message(op.name, subtask_idx)
                continue
            elif reply.header.type in (MessageType.execute_reply, MessageType.error):
                try:
                    return self._process_reply(request, reply)
                finally:
                    self.message_router.task_done(request.id)
            else:
                raise RuntimeError(f"Received unsupported message {reply}. Aborting execution.")

    async def _wait_for_reply(self, request: ExecuteRequestMessage) -> WorkMessage:
        while True:
            try:
                return await self.message_router.get(request.id, block=False)
            except asyncio.QueueEmpty:
                await asyncio.sleep(SLEEP_S)
                if self.is_cancelled:
                    raise CancelledOpError()

    @add_trace
    async def _run_op_impl(
        self, op: GraphNodeType, input: OpIOType, run_id: UUID, subtask_idx: int
    ) -> OpIOType:
        try:
            add_span_attributes({"op_name": op.spec.name})
            return await self._build_and_process_request(op, input, run_id, subtask_idx)
        except CancelledOpError:
            self.logger.debug(
                f"Did not try to run operation {op.name} for parent event {self.traceid}"
                " because the workflow was cancelled"
            )
            raise

    @add_trace
    async def _run_ops(self, ops: List[GraphNodeType], run_id: UUID):
        add_span_attributes({"workflow_id": str(run_id)})
        await super()._run_ops(ops, run_id)
        if len(self.message_router):
            self.logger.warning(
                f"Finishing workflow level {ops} execution with messages still in queue "
                f"(run id: {run_id})."
            )
        self.message_router.clear()

    def __del__(self):
        self.message_router.should_stop = True
