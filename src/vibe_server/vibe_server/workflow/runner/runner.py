import asyncio
import gc
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import auto
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Protocol, Set, Tuple, cast
from uuid import UUID, uuid4

from fastapi_utils.enums import StrEnum

from vibe_core.data.core_types import OpIOType
from vibe_core.data.utils import is_vibe_list
from vibe_core.utils import ensure_list

from ..workflow import DESTINATION, LABEL, EdgeLabel, EdgeType, GraphNodeType, InputFanOut, Workflow
from .task_io_handler import TaskIOHandler, WorkflowIOHandler


class CancelledOpError(Exception):
    pass


class WorkflowCallback(Protocol):
    async def __call__(self, change: "WorkflowChange", **kwargs: Any) -> None:
        pass


async def NoOpStateChange(change: "WorkflowChange", **kwargs: Any) -> None:
    return None


class WorkflowChange(StrEnum):
    WORKFLOW_STARTED = cast("WorkflowChange", auto())
    WORKFLOW_FINISHED = cast("WorkflowChange", auto())
    WORKFLOW_FAILED = cast("WorkflowChange", auto())
    WORKFLOW_CANCELLED = cast("WorkflowChange", auto())
    TASK_STARTED = cast("WorkflowChange", auto())
    SUBTASK_QUEUED = cast("WorkflowChange", auto())
    SUBTASK_RUNNING = cast("WorkflowChange", auto())
    SUBTASK_FINISHED = cast("WorkflowChange", auto())
    SUBTASK_FAILED = cast("WorkflowChange", auto())
    SUBTASK_PENDING = cast("WorkflowChange", auto())


class OpParallelism:
    parallel_edges: Set[EdgeType] = {EdgeType.parallel, EdgeType.scatter}

    def __init__(
        self,
        in_edges: List[EdgeLabel],
        op: GraphNodeType,
        run_task: Callable[[GraphNodeType, OpIOType, UUID, int], Awaitable[OpIOType]],
        update_state_callback: WorkflowCallback = NoOpStateChange,
    ):
        self.op = op
        self.in_edges = in_edges
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.run_task = run_task
        self.update_state = update_state_callback

    def is_parallel(self, edge: EdgeLabel) -> bool:
        return edge.type in self.parallel_edges

    def fan_in(self, inputs: List[OpIOType]) -> OpIOType:
        if any(self.is_parallel(edge) for edge in self.in_edges):
            # Op is running in parallel so we collate the outputs
            outputs: OpIOType = {k: [] for k in inputs[0]}
            for input in inputs:
                for key, value in outputs.items():
                    cast(List[Dict[str, Any]], value).append(cast(Dict[str, Any], input[key]))
            return outputs
        # Op is single, so we just return the output
        if len(inputs) > 1:
            raise RuntimeError(f"Expected a single input in the list, found {len(inputs)}")
        return inputs[0]

    @staticmethod
    def align(**kwargs: Any) -> Iterable[Tuple[Any, ...]]:
        input_lens = {n: len(arg) for n, arg in kwargs.items() if len(arg) != 1}
        lens = set(input_lens.values())
        if len(lens) > 1:
            error_str = ", ".join(f"'{k}': {v}" for k, v in input_lens.items())
            raise ValueError(f"Unable to pair sequences of different sizes - {error_str}")
        for i in range(1 if len(lens) == 0 else max(lens)):
            yield tuple((arg[i] if len(arg) > 1 else arg[0]) for arg in kwargs.values())

    def fan_out(self, op_input: OpIOType) -> Iterable[Tuple[OpIOType, ...]]:
        parallel = {edge.dstport for edge in self.in_edges if self.is_parallel(edge)}
        try:
            aligned = self.align(
                **{k: ([vv for vv in v] if k in parallel else [v]) for k, v in op_input.items()}
            )
            for input in aligned:
                yield tuple(
                    cast(OpIOType, ensure_list(i))
                    if is_vibe_list(self.op.spec.inputs_spec[name])
                    else i
                    for i, name in zip(input, op_input)
                )
        except ValueError as e:
            raise ValueError(f"Unable to fan-out input for op {self.op.name}: {e}") from e

    async def run(self, op_input: OpIOType, run_id: UUID) -> List[OpIOType]:
        if isinstance(self.op.spec, InputFanOut):
            self.logger.info(f"Bypassing input fan-out node {self.op.name}")
            await self.update_state(WorkflowChange.TASK_STARTED, task=self.op.name, num_subtasks=1)
            await self.update_state(
                WorkflowChange.SUBTASK_FINISHED, task=self.op.name, subtask_idx=0
            )
            return [{self.op.spec.output_port: op_input[self.op.spec.input_port]}]
        inputs: List[OpIOType] = [
            {k: v for k, v in zip(op_input.keys(), input)} for input in self.fan_out(op_input)
        ]
        await self.update_state(
            WorkflowChange.TASK_STARTED, task=self.op.name, num_subtasks=len(inputs)
        )
        self.logger.info(
            f"Will run op {self.op.name} with {len(inputs)} different input(s). "
            f"(run id: {run_id})"
        )

        async def sub_run(args: Tuple[int, OpIOType]) -> OpIOType:
            idx, input = args
            try:
                self.logger.debug(
                    f"Executing task {idx + 1}/{len(inputs)} of op {self.op.name}. "
                    f"(run id: {run_id})"
                )
                await self.update_state(
                    WorkflowChange.SUBTASK_QUEUED, task=self.op.name, subtask_idx=idx
                )
                ret = await self.run_task(self.op, input, run_id, idx)
                self.logger.debug(
                    f"Successfully executed task {idx + 1}/{len(inputs)} of op {self.op.name}. "
                    f"(run id: {run_id})"
                )
                await self.update_state(
                    WorkflowChange.SUBTASK_FINISHED, task=self.op.name, subtask_idx=idx
                )
                return ret
            except Exception as e:
                self.logger.exception(
                    f"Failed to execute task {idx + 1}/{len(inputs)} of op {self.op.name}. "
                    f"(run id: {run_id})"
                )
                await self.update_state(
                    WorkflowChange.SUBTASK_FAILED,
                    task=self.op.name,
                    subtask_idx=idx,
                    reason=f"{e.__class__.__name__}: {e}",
                )
                raise

        results = await asyncio.gather(*[sub_run(args) for args in enumerate(inputs)])
        return results


class WorkflowRunner(ABC):
    workflow: Workflow
    update_state: WorkflowCallback
    logger: logging.Logger
    io_mapper: WorkflowIOHandler
    io_handler: TaskIOHandler
    is_cancelled: bool

    def __init__(
        self,
        workflow: Workflow,
        io_mapper: WorkflowIOHandler,
        update_state_callback: WorkflowCallback = NoOpStateChange,
        **_: Any,
    ):
        self.workflow = workflow
        self.update_state = update_state_callback
        self.io_mapper = io_mapper
        self.is_cancelled = False

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def cancel(self):
        await self._report_state_change(WorkflowChange.WORKFLOW_CANCELLED)
        self.is_cancelled = True

    @abstractmethod
    async def _run_op_impl(
        self, op: GraphNodeType, input: OpIOType, run_id: UUID, subtask_idx: int
    ) -> OpIOType:
        raise NotImplementedError

    async def _run_graph_impl(self, input: OpIOType, run_id: UUID) -> OpIOType:
        self.io_handler.add_sources(input)
        for ops in self.workflow:
            self.logger.info(f"Will run ops {ops} in parallel. (run id: {run_id})")
            await self._run_ops(ops, run_id)
        if not self.is_cancelled:
            return self.io_handler.retrieve_sinks()

        # Workflow was cancelled
        return {}

    async def _run_ops(self, ops: List[GraphNodeType], run_id: UUID):
        try:
            op_parallelism = {}
            tasks: List[Tuple[GraphNodeType, "asyncio.Task[List[OpIOType]]"]] = []
            for op in ops:
                op_parallelism[op.name] = OpParallelism(
                    [e[LABEL] for e in self.workflow.edges if e[DESTINATION] == op],
                    op,
                    self._run_op_impl,
                    update_state_callback=self.update_state,
                )
                task = asyncio.create_task(
                    self._submit_op(op, run_id, op_parallelism[op.name]), name=op.name
                )
                tasks.append((op, task))
            await self._monitor_futures(tasks, run_id, op_parallelism)
            for _, task in tasks:
                if not task.done():
                    task.cancel()
            del tasks
        finally:
            # The garbage collector seems to be a bit lazy, so we need to force it to collect
            # anything that's been leftover from previous executions
            collected = gc.collect()
            self.logger.debug(
                f"Garbage collector collected {collected} objects after running ops {ops} "
                f"in run {run_id}."
            )

    async def _monitor_futures(
        self,
        tasks: List[Tuple[GraphNodeType, "asyncio.Task[List[OpIOType]]"]],
        run_id: UUID,
        op_parallelism: Dict[str, OpParallelism],
    ):
        op_outputs: Dict[GraphNodeType, List[OpIOType]] = defaultdict(list)
        gather = asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        await gather
        for op, task in tasks:
            op_name = task.get_name()
            try:
                assert op_name is not None
                result = await task
                if isinstance(result, Exception):
                    raise result
                op_outputs[op].extend(result)
            except CancelledOpError:
                return
            except Exception as e:
                gather.cancel()
                await self._fail_workflow(e, run_id)
                raise

        for op, op_result in op_outputs.items():
            self.io_handler.add_result(op, op_parallelism[op.name].fan_in(op_result))

    async def _fail_workflow(self, e: Exception, run_id: UUID):
        self.logger.exception(f"Failed to run workflow {self.workflow.name}. (run id: {run_id})")
        await self._report_state_change(WorkflowChange.WORKFLOW_FAILED, reason=str(e))

    @classmethod
    def build(
        cls,
        workflow: Workflow,
        **kwargs: Any,
    ) -> "WorkflowRunner":
        return cls(workflow, **kwargs)

    async def _submit_op(
        self,
        op: GraphNodeType,
        run_id: UUID,
        parallelism: OpParallelism,
    ) -> List[OpIOType]:
        if self.is_cancelled:
            # Exit early, as this run has been cancelled
            return [{}]
        input = self.io_handler.retrieve_input(op)
        try:
            return await parallelism.run(input, run_id)
        except CancelledOpError:
            return [{}]
        except Exception as e:
            await self._fail_workflow(e, run_id)
            raise

    async def _run_graph(self, input: OpIOType, run_id: UUID) -> OpIOType:
        self.logger.debug(f"Starting execution of workflow {self.workflow.name} (run id: {run_id})")
        tasks = [task.name for level in self.workflow for task in level]
        await self._report_state_change(WorkflowChange.WORKFLOW_STARTED, tasks=tasks)
        output = self._run_graph_impl(input, run_id)
        # Mark workflow as cancelled if needed
        # Do not mark workflow as done, as it will be marked as such after the outputs are updated
        # in the statestore
        if self.is_cancelled:
            await self._report_state_change(WorkflowChange.WORKFLOW_CANCELLED)
        self.logger.debug(f"Finished execution of workflow {self.workflow.name} (run id: {run_id})")

        return await output

    async def run(self, input_items: OpIOType, run_id: UUID = uuid4()) -> OpIOType:
        try:
            # Initializing task IO handler for this specific run.
            self.io_handler = TaskIOHandler(self.workflow)
            output = await self._run_graph(self.io_mapper.map_input(input_items), run_id)
            return self.io_mapper.map_output(output) if not self.is_cancelled else {}
        except Exception as e:
            self.logger.exception(f"Failed to run workflow {self.workflow.name} (run id: {run_id})")
            await self._report_state_change(WorkflowChange.WORKFLOW_FAILED, reason=str(e))
            raise
        finally:
            del self.io_handler

    async def _report_state_change(
        self,
        change: WorkflowChange,
        **kwargs: Any,
    ) -> None:
        try:
            await self.update_state(change, **kwargs)
        except Exception:
            logging.exception(
                f"Failed to update workflow/operation state with change {change}. Ignoring."
            )
