import asyncio
import asyncio.queues
import logging
from argparse import ArgumentParser
from copy import copy
from dataclasses import asdict
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from uuid import UUID

import debugpy
from cloudevents.sdk.event import v1
from dapr.conf import settings
from opentelemetry import trace

from vibe_common.constants import (
    CACHE_PUBSUB_TOPIC,
    CONTROL_STATUS_PUBSUB,
    DEFAULT_OPS_DIR,
    RUNS_KEY,
    STATUS_PUBSUB_TOPIC,
    WORKFLOW_REQUEST_PUBSUB_TOPIC,
)
from vibe_common.dapr import dapr_ready
from vibe_common.dropdapr import App, TopicEventResponse
from vibe_common.messaging import (
    OpIOType,
    WorkflowCancellationMessage,
    WorkflowDeletionMessage,
    WorkflowExecutionMessage,
    WorkMessage,
    WorkMessageBuilder,
    accept_or_fail_event_async,
    extract_message_header_from_event,
    run_id_from_traceparent,
)
from vibe_common.statestore import StateStore, TransactionOperation
from vibe_common.telemetry import add_trace, setup_telemetry, update_telemetry_context
from vibe_core.datamodel import RunConfig, RunDetails, RunStatus
from vibe_core.logconfig import LOG_BACKUP_COUNT, MAX_LOG_FILE_BYTES, configure_logging

from .workflow import workflow_from_input
from .workflow.input_handler import build_args_for_workflow, patch_workflow_sources
from .workflow.runner.remote_runner import MessageRouter, RemoteWorkflowRunner
from .workflow.runner.runner import WorkflowCallback, WorkflowChange, WorkflowRunner
from .workflow.runner.task_io_handler import WorkflowIOHandler
from .workflow.spec_parser import WorkflowParser
from .workflow.workflow import Workflow, get_workflow_dir

Updates = Tuple[bool, List[str]]


class WorkflowStateUpdate(WorkflowCallback):
    """Keeps track of the state of a workflow and its tasks.
    The state is stored in the statestore and updated based on the events received
    from the workflow runner. The workflow and task states are updated in the statestore with
    different keys in order to avoid upserting a large amount of data with every update.

    In general, the state of a task is defined based on the status of its subtasks.
    A task is marked as a status when at least one of its subtasks is marked as that status
    in the following other of priority:
    1. failed.
    2. running.
    3. queued.
    4. pending.
    5. done.
    Whenever an update to a subtask happens, we propagate it up and update statuses as necessary.
    The analogous is defined for workflow w.r.t tasks.

    Cancellation and failure events are also propagated down.
    This means that when a workflow is cancelled, all tasks are updated and cancelled as well
    (unless already done). The analogous happens for tasks and subtasks.
    For failures, we propagate the cancelled state down and the failed state up.
    """

    user_request_reason = "Cancellation requested by user"
    workflow_failure_reason = "Cancelled due to failure during workflow execution"

    def __init__(self, workflowRunId: UUID):
        self.run_id = workflowRunId
        self.wf_cache: Dict[str, Any] = {}
        self.task_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.statestore = StateStore()
        self.update_lock = asyncio.Lock()
        # Cache "empty" RunDetails because creating it triggers the big bad bug
        self.pending_run = asdict(RunDetails())
        self.wf_change_to_update = {
            WorkflowChange.WORKFLOW_STARTED: self.create_workflow,
            WorkflowChange.WORKFLOW_FINISHED: self.complete_workflow,
            WorkflowChange.WORKFLOW_CANCELLED: self.cancel_workflow,
            WorkflowChange.WORKFLOW_FAILED: self.fail_workflow,
            WorkflowChange.TASK_STARTED: self.create_subtasks,
            WorkflowChange.SUBTASK_QUEUED: self.queue_subtask,
            WorkflowChange.SUBTASK_RUNNING: self.execute_subtask,
            WorkflowChange.SUBTASK_FINISHED: self.complete_subtask,
            WorkflowChange.SUBTASK_FAILED: self.fail_subtask,
            WorkflowChange.SUBTASK_PENDING: self.pend_subtask,
        }
        self._cache_init = False

    async def _init_cache(self):
        # TODO: We could also load task cache here in case we want to resume a workflow
        cache = await self.statestore.retrieve(str(self.run_id))
        self.wf_cache["details"] = cache["details"]
        self._cache_init = True

    def create_workflow(self, tasks: List[str]) -> Updates:
        # Workflow start time is set when we start running the graph
        self.wf_cache["details"]["start_time"] = datetime.now()
        self.wf_cache["tasks"] = tasks
        for t in tasks:
            self.task_cache[t] = copy(self.pending_run)
        return True, tasks

    def complete_workflow(self) -> Updates:
        return self._update_finish_change(None, None, cancelled=False, reason=""), []

    def cancel_workflow(self) -> Updates:
        fun = partial(self._update_finish_change, cancelled=True, reason=self.user_request_reason)
        return self._propagate_down(fun)

    def fail_workflow(self, reason: str) -> Updates:
        wf_updated = self._update_failure_change(None, None, reason=reason)
        if not wf_updated:
            # We won't cancel the workflow because it is already finished
            return False, []
        fun = partial(
            self._update_finish_change,
            cancelled=True,
            reason=self.workflow_failure_reason,
        )
        _, updated_tasks = self._propagate_down(fun)
        return wf_updated, updated_tasks

    def create_subtasks(self, task: str, num_subtasks: int) -> Updates:
        cache, name = self._get_cache(task, None)
        cache["subtasks"] = [copy(self.pending_run) for _ in range(num_subtasks)]
        self.logger.info(f"Created {num_subtasks} subtasks for {name}. (run id: {self.run_id})")
        return False, [task]

    def queue_subtask(self, task: str, subtask_idx: int) -> Updates:
        return self._propagate_up(self._update_queued_change, task, subtask_idx)

    def execute_subtask(self, task: str, subtask_idx: int) -> Updates:
        return self._propagate_up(self._update_start_change, task, subtask_idx)

    def complete_subtask(self, task: str, subtask_idx: int) -> Updates:
        fun = partial(self._update_finish_change, cancelled=False, reason="")
        return self._propagate_up(fun, task, subtask_idx)

    def fail_subtask(self, task: str, subtask_idx: int, reason: str) -> Updates:
        fail_fun = partial(self._update_failure_change, reason=reason)
        subtask_updated = fail_fun(task, subtask_idx, reason=reason)
        task_updated = fail_fun(task, None, reason=reason)
        wf_updated_up = fail_fun(None, None, reason=reason)
        updated_tasks_up = [task] if (task_updated or subtask_updated) else []
        cancel_fun = partial(
            self._update_finish_change,
            cancelled=True,
            reason=f"Cancelled because task '{task}' (subtask {subtask_idx}) failed",
        )
        wf_updated_down, updated_tasks_down = self._propagate_down(cancel_fun)
        wf_updated = wf_updated_up or wf_updated_down
        updated_tasks = updated_tasks_up + [
            i for i in updated_tasks_down if i not in updated_tasks_up
        ]
        return wf_updated, updated_tasks

    def pend_subtask(self, task: str, subtask_idx: int) -> Updates:
        return self._propagate_up(self._update_pending_change, task, subtask_idx)

    def _combine_children_status(self, children_status: Set[RunStatus]) -> RunStatus:
        for status in (RunStatus.running, RunStatus.queued, RunStatus.pending):
            if status in children_status:
                new_status = status
                break
        else:
            if children_status != {RunStatus.done}:
                raise ValueError(f"Unknown status combination: {children_status}")
            new_status = RunStatus.done
        return new_status

    def _combine_children_time(
        self, children_start: List[Optional[datetime]], children_end: List[Optional[datetime]]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        children_start = [i for i in children_start if i is not None]
        if not children_start:
            start_time = None
        else:
            start_time = min(cast(List[datetime], children_start))
        if any(i is None for i in children_end):
            end_time = None
        else:
            end_time = max(cast(List[datetime], children_end))
        return start_time, end_time

    def _update_task_status(self, task: str) -> bool:
        cache, _ = self._get_cache(task, None)
        if cache["subtasks"] is None:
            raise RuntimeError(f"Tried to update status of task {task} before creating subtasks")
        subtask_status = {i["status"] for i in cache["subtasks"]}
        new_status = self._combine_children_status(subtask_status)
        if new_status != cache["status"]:
            cache["status"] = new_status
            cache["submission_time"], _ = self._combine_children_time(
                [i["submission_time"] for i in cache["subtasks"]],
                [None],
            )
            cache["start_time"], cache["end_time"] = self._combine_children_time(
                [i["start_time"] for i in cache["subtasks"]],
                [i["end_time"] for i in cache["subtasks"]],
            )
            return True
        return False

    def _update_workflow_status(self) -> bool:
        cache, _ = self._get_cache(None, None)
        task_status = {i["status"] for i in self.task_cache.values()}
        new_status = self._combine_children_status(task_status)
        if new_status == RunStatus.done:
            # We don't set it to done here because we still need to store the output
            # We only set to done when complete_workflow is called by the orchestrator
            new_status = RunStatus.running
        if new_status != cache["status"]:
            cache["status"] = new_status
            return True
        return False

    def _propagate_up(
        self, fun: Callable[[Optional[str], Optional[int]], bool], task: str, subtask_idx: int
    ) -> Updates:
        subtask_updated = fun(task, subtask_idx)
        if not subtask_updated:
            return False, []
        task_updated = self._update_task_status(task)
        if not task_updated:
            return False, [task]
        return self._update_workflow_status(), [task]

    def _propagate_down(self, fun: Callable[[Optional[str], Optional[int]], bool]) -> Updates:
        wf_updated = fun(None, None)
        updated_tasks = []
        for task, task_cache in self.task_cache.items():
            task_updated = fun(task, None)
            if task_cache["subtasks"] is not None:
                subtask_updated = any([fun(task, i) for i in range(len(task_cache["subtasks"]))])
            else:
                subtask_updated = False
            if task_updated or subtask_updated:
                updated_tasks.append(task)
        return wf_updated, updated_tasks

    def _get_cache(
        self, task: Optional[str], subtask_idx: Optional[int]
    ) -> Tuple[Dict[str, Any], str]:
        if task is None:
            return self.wf_cache["details"], "workflow"
        if subtask_idx is None:
            return self.task_cache[task], f"task {task}"
        subtasks_cache = self.task_cache[task]["subtasks"]
        if subtasks_cache is None:
            raise ValueError(
                f"Tried to update subtask {subtask_idx} for {task} before creating subtasks"
            )
        return (
            self.task_cache[task]["subtasks"][subtask_idx],
            f"task {task} (subtask {subtask_idx})",
        )

    def _update_pending_change(self, task: Optional[str], subtask_idx: Optional[int]) -> bool:
        cache, name = self._get_cache(task, subtask_idx)
        if RunStatus.finished(cache["status"]):
            return False
        cache["status"] = RunStatus.pending
        self.logger.info(f"Changed {name} status to {RunStatus.pending}. (run id: {self.run_id})")
        return True

    def _update_queued_change(self, task: Optional[str], subtask_idx: Optional[int]) -> bool:
        cache, name = self._get_cache(task, subtask_idx)
        if RunStatus.finished(cache["status"]):
            return False
        if cache["submission_time"] is None:
            cache["submission_time"] = datetime.now()
        cache["status"] = RunStatus.queued
        self.logger.info(f"Changed {name} status to {RunStatus.queued}. (run id: {self.run_id})")
        return True

    def _update_start_change(self, task: Optional[str], subtask_idx: Optional[int]) -> bool:
        cache, name = self._get_cache(task, subtask_idx)
        if RunStatus.finished(cache["status"]) or cache["status"] == RunStatus.running:
            return False
        if cache["start_time"] is None:
            cache["start_time"] = datetime.now()
        cache["status"] = RunStatus.running
        self.logger.info(f"Changed {name} status to {RunStatus.running}. (run id: {self.run_id})")
        return True

    def _update_finish_change(
        self, task: Optional[str], subtask_idx: Optional[int], cancelled: bool, reason: str
    ) -> bool:
        cache, name = self._get_cache(task, subtask_idx)
        if RunStatus.finished(cache["status"]):
            return False
        status = RunStatus.cancelled if cancelled else RunStatus.done
        for missing in ("submission_time", "start_time"):
            if cache[missing] is None:
                cache[missing] = datetime.now()
                if not cancelled:
                    self.logger.warning(
                        f"Marking {name} as finished, "
                        f"but it didn't have a {missing} set. (run id: {self.run_id})"
                    )
        cache["end_time"] = datetime.now()
        cache["status"] = status
        if cancelled:
            cache["reason"] = reason
        self.logger.info(f"Changed {name} status to {status}. (run id: {self.run_id})")
        return True

    def _update_failure_change(
        self, task: Optional[str], subtask_idx: Optional[int], reason: str
    ) -> bool:
        cache, name = self._get_cache(task, subtask_idx)
        if RunStatus.finished(cache["status"]):
            return False
        if cache["start_time"] is None:
            self.logger.error(
                f"Marking {name} as failed, "
                f"but it didn't have a start time set. (run id: {self.run_id})"
            )
            cache["start_time"] = datetime.now()
        cache["end_time"] = datetime.now()
        cache["status"] = RunStatus.failed
        cache["reason"] = reason
        self.logger.info(f"Changed {name} status to {RunStatus.failed}. (run id: {self.run_id})")
        return True

    def update_cache_for(self, change: WorkflowChange, **kwargs: Any) -> Updates:
        update_fun = self.wf_change_to_update[change]
        return update_fun(**kwargs)

    async def commit_cache_for(self, update_workflow: bool, tasks: List[str]) -> None:
        # We are not deserializing run data into a RunConfig object because this breaks *something*
        # We do not deserialize the cache into RunDetails for the same reason
        operations = [
            TransactionOperation(
                key=f"{self.run_id}-{t}", operation="upsert", value=self.task_cache[t]
            )
            for t in tasks
        ]
        if update_workflow:
            wf_data = await self.statestore.retrieve(str(self.run_id))
            wf_data["tasks"] = self.wf_cache["tasks"]
            wf_data["details"] = self.wf_cache["details"]
            operations.append(
                TransactionOperation(key=str(self.run_id), operation="upsert", value=wf_data)
            )

        await self.statestore.transaction(operations)

    async def __call__(self, change: WorkflowChange, **kwargs: Any) -> None:
        async with self.update_lock:
            # Since we parallelize op execution, there might be a race condition
            # on updating the overall status of a given workflow run. Locking
            # here serializes status updates and guarantees we won't overwrite
            # previously-written updates
            if not self._cache_init:
                await self._init_cache()
            update_workflow, tasks_to_update = self.update_cache_for(change, **kwargs)
            if update_workflow or tasks_to_update:
                await self.commit_cache_for(update_workflow, tasks_to_update)


class WorkflowRunManager:
    inqueues: Dict[str, "asyncio.queues.Queue[WorkMessage]"]
    runner: Optional[WorkflowRunner]

    def __init__(
        self,
        inqueues: Dict[str, "asyncio.queues.Queue[WorkMessage]"],
        message: WorkflowExecutionMessage,
        pubsubname: str,
        source: str,
        topic: str,
        ops_dir: str = DEFAULT_OPS_DIR,
        workflows_dir: str = get_workflow_dir(),
        *args: Any,
        **kwargs: Dict[str, Any],
    ):
        self.message = message
        self.inqueues = inqueues
        self.statestore = StateStore()
        self.runner = None
        self.name = str(message.run_id)
        self.is_cancelled = False
        self.ops_dir = ops_dir
        self.workflows_dir = workflows_dir
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.task = asyncio.create_task(self.start_managing())
        self.pubsubname = pubsubname
        self.topic = topic
        self.source = source

        def done_callback(task: Any) -> None:
            self.task = None
            try:
                maybe_exception = task.exception()
                if maybe_exception is not None:
                    self.logger.warning(
                        f"Task {task} for workflow run {self.name} failed "
                        f"with exception {maybe_exception}"
                    )
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass

        self.task.add_done_callback(done_callback)

    def build_workflow(self, input_items: OpIOType):
        content = self.message.content
        spec = WorkflowParser.parse_dict(
            content.workflow,
            ops_dir=self.ops_dir,
            parameters_override=content.parameters,
        )
        workflow = Workflow(spec)
        patch_workflow_sources(input_items, workflow)
        io_mapper = WorkflowIOHandler(workflow)
        return workflow, io_mapper

    async def start_managing(self) -> None:
        content = self.message.content
        input_items = content.input
        run_id = self.message.run_id
        self.inqueues[str(run_id)] = asyncio.queues.Queue()
        try:
            workflow, io_mapper = self.build_workflow(input_items)
        except Exception:
            await update_workflow(
                str(run_id),
                self.statestore,
                RunStatus.failed,
                f"Failed to build workflow {content.workflow}"
                f" with parameters: {content.parameters}",
            )
            raise
        router = MessageRouter(self.inqueues[str(run_id)])
        self.runner = RemoteWorkflowRunner(
            traceid=self.message.id,
            message_router=router,
            workflow=workflow,
            io_mapper=io_mapper,
            update_state_callback=WorkflowStateUpdate(run_id),
            pubsubname=self.pubsubname,
            source=self.source,
            topic=self.topic,
        )
        self.runner.is_cancelled = self.is_cancelled
        output = await self.runner.run(input_items, run_id)
        router.should_stop = True
        if router.task is not None:
            await router.task
        if not self.is_cancelled:
            await self.add_output(output)
            self.logger.debug(
                f"Updated statestore with output for workflow run {self.message.run_id}"
            )
            await self.runner.update_state(WorkflowChange.WORKFLOW_FINISHED)
            self.logger.debug(f"Marked workflow run {self.message.run_id} as done")

    async def add_output(self, output: OpIOType) -> None:
        await self.add_output_to_run(str(self.message.run_id), output, self.statestore)

    @staticmethod
    async def add_output_to_run(run_id: str, output: OpIOType, statestore: StateStore) -> None:
        run_data = await statestore.retrieve(run_id)
        run_config = RunConfig(**run_data)
        run_config.set_output(output)
        await statestore.store(run_id, run_config)

    async def cancel(self):
        self.is_cancelled = True
        if self.runner is not None:
            await self.runner.cancel()


async def update_workflow(
    run_id: str,
    statestore: StateStore,
    new_status: RunStatus,
    reason: Optional[str] = None,
    dont_update: Callable[[RunStatus], bool] = RunStatus.finished,
) -> None:
    run_data = await statestore.retrieve(run_id)
    run_config = RunConfig(**run_data)
    if dont_update(run_config.details.status):
        return
    run_config.details.status = new_status
    run_config.details.reason = reason if reason else ""
    if new_status in {RunStatus.failed}:
        run_config.details.start_time = run_config.details.end_time = datetime.now()
    await statestore.store(run_id, run_config)


class Orchestrator:
    app: App
    inqueues: Dict[str, "asyncio.queues.Queue[WorkMessage]"]
    pubsubname: str
    cache_topic: str
    new_workflow_topic: str
    _workflow_management_tasks: Dict[UUID, WorkflowRunManager]
    ops_dir: str
    workflows_dir: str

    # TODO: We need some way of reloading orchestrator state to make it robust
    # to crashes

    def __init__(
        self,
        pubsubname: str = CONTROL_STATUS_PUBSUB,
        cache_topic: str = CACHE_PUBSUB_TOPIC,
        status_topic: str = STATUS_PUBSUB_TOPIC,
        new_workflow_topic: str = WORKFLOW_REQUEST_PUBSUB_TOPIC,
        port: int = settings.GRPC_APP_PORT,
        ops_dir: str = DEFAULT_OPS_DIR,
        workflows_dir: str = get_workflow_dir(),
    ):
        self.app = App()
        self.port = port
        self.pubsubname = pubsubname
        self.cache_topic = cache_topic
        self.status_topic = status_topic
        self.new_workflow_topic = new_workflow_topic
        self.inqueues = {}
        self.statestore = StateStore()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._workflow_management_tasks: Dict[UUID, WorkflowRunManager] = {}
        self.ops_dir = ops_dir
        self.workflows_dir = workflows_dir

        @self.app.subscribe_async(self.pubsubname, self.status_topic)
        async def update(event: v1.Event):
            await self.handle_update_workflow_status(self.status_topic, event)

        @self.app.subscribe_async(self.pubsubname, self.new_workflow_topic)
        async def manage_workflow(event: v1.Event):
            await self.handle_manage_workflow_event(self.new_workflow_topic, event)

    async def handle_update_workflow_status(self, channel: str, event: v1.Event):
        async def success_callback(message: WorkMessage) -> TopicEventResponse:
            if not message.is_valid_for_channel(channel):
                self.logger.error(
                    f"Received unsupported message {message} for channel {channel}. Dropping it."
                )
                return TopicEventResponse("drop")
            if str(message.run_id) not in self.inqueues:
                self.logger.info(
                    f"Received message {message}, but the run it references"
                    " is not being managed. Dropping it."
                )
                return TopicEventResponse("drop")
            await self.inqueues[str(message.run_id)].put(message)
            return TopicEventResponse("success")

        return await accept_or_fail_event_async(event, success_callback, self._failure_callback)

    async def handle_manage_workflow_event(self, channel: str, event: v1.Event):
        update_telemetry_context(extract_message_header_from_event(event).current_trace_parent)

        @add_trace
        async def success_callback(message: WorkMessage) -> TopicEventResponse:
            try:
                if not message.is_valid_for_channel(channel):
                    self.logger.error(f"Received unsupported message {message}. Dropping it.")
                    return TopicEventResponse("drop")
                await self.manage_workflow(message)
                return TopicEventResponse("success")
            except Exception as e:
                await self.fail_workflow(str(message.run_id), f"{e.__class__.__name__}: {e}")
                self.logger.exception(
                    f"Failed to submit workflow {message.run_id} "
                    f"from event {event.id} for execution"
                )
                return TopicEventResponse("drop")

        with trace.get_tracer(__name__).start_as_current_span("handle_manage_workflow_event"):
            return await accept_or_fail_event_async(event, success_callback, self._failure_callback)

    @add_trace
    async def handle_workflow_execution_message(self, message: WorkflowExecutionMessage):
        wf = WorkflowRunManager(
            self.inqueues,
            message,
            pubsubname=self.pubsubname,
            source="orchestrator",
            topic=self.cache_topic,
            ops_dir=self.ops_dir,
            workflows_dir=self.workflows_dir,
        )
        self._workflow_management_tasks[message.run_id] = wf

        def wf_done_callback(task: "asyncio.Future[Any]") -> None:
            self.logger.info(f"Workflow run {message.run_id} finished. Freeing up space.")
            self.inqueues.pop(str(message.run_id))
            self._workflow_management_tasks.pop(message.run_id)
            try:
                maybe_exception = task.exception()
                if maybe_exception is not None:
                    self.logger.warning(
                        f"Workflow run {message.run_id} failed with exception {maybe_exception}"
                    )
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass

        wf.task.add_done_callback(wf_done_callback)

    async def handle_workflow_cancellation_message(self, message: WorkflowCancellationMessage):
        if message.run_id in self._workflow_management_tasks:
            wf = self._workflow_management_tasks[message.run_id]
            if not wf.task.done():
                await wf.cancel()
            else:
                self.logger.warning(
                    f"Tried to cancel a workflow run from message {message}, "
                    f"but the workflow has already finished. (run id: {message.run_id})"
                )
        else:
            # We don't know this workflow run. Either this completed execution
            # already, or it doesn't exist. Log and ignore.
            self.logger.warning(
                f"Tried to cancel a workflow run from message {message}, "
                f"but the run doesn't exist. (run id: {message.run_id})"
            )

    async def handle_workflow_deletion_message(self, message: WorkflowDeletionMessage):
        # deletion of a workflow run is handled by the data ops service
        pass

    async def manage_workflow(self, message: WorkMessage) -> None:
        message_handler_map = {
            WorkflowExecutionMessage: self.handle_workflow_execution_message,
            WorkflowCancellationMessage: self.handle_workflow_cancellation_message,
            WorkflowDeletionMessage: self.handle_workflow_deletion_message,
        }
        handled = False
        for type in message_handler_map:
            if isinstance(message, type):
                handled = True
                await message_handler_map[type](message)
                break
        if not handled:
            self.logger.error(f"Unable to process message {message}. Ignoring.")

    async def update_workflow_if_not_finished(self, run_id: str, reason: str, status: RunStatus):
        await update_workflow(run_id, self.statestore, status, reason)

    async def fail_workflow(self, run_id: str, reason: str):
        await self.update_workflow_if_not_finished(run_id, reason, RunStatus.failed)

    async def _failure_callback(
        self, event: v1.Event, e: Exception, traceback: List[str]
    ) -> TopicEventResponse:
        run_id = str(run_id_from_traceparent(event.id))
        await self.fail_workflow(
            run_id, f"{e.__class__.__name__}: {str(e)}\n" + "\n".join(traceback)
        )
        self.logger.info(f"Marked workflow {run_id} from event {event.id} failed")
        return TopicEventResponse("drop")

    @dapr_ready
    async def run(self):
        async def shutdown_callback(task: Any):
            try:
                maybe_exception = task.exception()
                if maybe_exception is not None:
                    self.logger.warning(f"Server task failed with exception {maybe_exception}.")
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass

        self.logger.info(f"Starting orchestrator listening on port {self.port}")
        server_task = asyncio.create_task(self.app.run_async(self.port))
        server_task.add_done_callback(shutdown_callback)
        resume_call = self._resume_workflows()
        await asyncio.gather(server_task, resume_call)

    async def get_unfinished_workflows(self) -> List[RunConfig]:
        keys = []
        try:
            keys = await self.statestore.retrieve(RUNS_KEY)
        except KeyError:
            await self.statestore.store(RUNS_KEY, [])

        all_runs = cast(
            List[RunConfig], [RunConfig(**r) for r in await self.statestore.retrieve_bulk(keys)]
        )
        return [r for r in all_runs if not RunStatus.finished(r.details.status)]

    def run_config_to_workflow_message(self, run: RunConfig) -> WorkflowExecutionMessage:
        workflow = workflow_from_input(run.workflow)
        inputs_spec = workflow.inputs_spec
        user_input = build_args_for_workflow(run.user_input, list(inputs_spec))
        message = WorkMessageBuilder.build_workflow_request(
            run.id, asdict(workflow.workflow_spec), run.parameters, user_input
        )

        update_telemetry_context(message.current_trace_parent)
        with trace.get_tracer(__name__).start_as_current_span("re-submit-workflow"):
            return cast(WorkflowExecutionMessage, message)

    async def _resume_workflows(self):
        self.logger.debug("Searching for unfinished workflow runs")
        try:
            runs = await self.get_unfinished_workflows()
        except Exception:
            raise RuntimeError(
                "Failed to fetch list of unfinished workflow runs. Aborting Execution."
            )
        self.logger.debug(f"Found {len(runs)} unfinished workflow run(s)")

        unfinished_tasks = []
        for run in runs:
            self.logger.debug(f"Resuming workflow run {run.id}")
            try:
                message = self.run_config_to_workflow_message(run)
                self.logger.debug(f"Created workflow execution message for run id {run.id}")
                unfinished_tasks.append(
                    asyncio.create_task(self.handle_workflow_execution_message(message))
                )
            except Exception:
                self.logger.exception(f"Failed to resume execution for workflow run {run.id}")
                raise

        await asyncio.gather(*unfinished_tasks)


async def main():
    parser = ArgumentParser(description="TerraVibes 🌎 Orchestrator")
    parser.add_argument(
        "--pubsubname",
        type=str,
        default=CONTROL_STATUS_PUBSUB,
        help="The name of the publish subscribe component to use",
    )
    parser.add_argument(
        "--cache-topic",
        type=str,
        default=CACHE_PUBSUB_TOPIC,
        help="The name of the topic to use to send control messages",
    )
    parser.add_argument(
        "--status-topic",
        type=str,
        default=STATUS_PUBSUB_TOPIC,
        help="The name of the topic to use to receive status messages",
    )
    parser.add_argument(
        "--workflow-topic",
        type=str,
        default=WORKFLOW_REQUEST_PUBSUB_TOPIC,
        help="The name of the topic to use to receive workflow execution requests",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(settings.GRPC_APP_PORT),
        help="The port to use to listen for HTTP requests from dapr",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Whether to enable remote debugging"
    )
    parser.add_argument(
        "--debugger-port",
        type=int,
        default=5678,
        help="The port on which to listen to the debugger",
    )
    parser.add_argument(
        "--otel-service-name",
        type=str,
        help="The name of the service to use for OpenTelemetry collector",
        default="",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="The directory on which to save logs",
        default="",
    )
    parser.add_argument(
        "--max-log-file-bytes",
        type=int,
        help="The maximum number of bytes for a log file",
        default=MAX_LOG_FILE_BYTES,
    )
    parser.add_argument(
        "--log-backup-count",
        type=int,
        help="The number of log files to keep",
        required=False,
        default=LOG_BACKUP_COUNT,
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        help="The default log level to use",
        default="INFO",
    )
    options = parser.parse_args()

    appname = "terravibes-orchestrator"
    configure_logging(
        appname=appname,
        logdir=options.logdir if options.logdir else None,
        max_log_file_bytes=options.max_log_file_bytes,
        log_backup_count=options.log_backup_count,
        logfile=f"{appname}.log",
        default_level=options.loglevel,
    )

    if options.otel_service_name:
        setup_telemetry(appname, options.otel_service_name)

    if options.debug:
        debugpy.listen(options.debugger_port)  # type: ignore
        logging.info(f"Debugger enabled and listening on port {options.debugger_port}")

    orchestrator = Orchestrator(
        pubsubname=options.pubsubname,
        cache_topic=options.cache_topic,
        status_topic=options.status_topic,
        new_workflow_topic=options.workflow_topic,
        port=options.port,
    )
    await orchestrator.run()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
