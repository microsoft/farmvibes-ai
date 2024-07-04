from typing import cast
from uuid import UUID

from vibe_agent.ops import OperationDependencyResolver, OperationFactoryConfig, OpIOType
from vibe_agent.ops_helper import OpIOConverter
from vibe_agent.worker import Worker
from vibe_common.messaging import (
    CacheInfoExecuteRequestContent,
    ExecuteRequestMessage,
    WorkMessageBuilder,
)
from vibe_common.schemas import CacheInfo
from vibe_server.workflow.runner.runner import (
    NoOpStateChange,
    WorkflowCallback,
    WorkflowChange,
    WorkflowRunner,
)
from vibe_server.workflow.runner.task_io_handler import WorkflowIOHandler
from vibe_server.workflow.workflow import GraphNodeType, Workflow

MAX_OP_EXECUTION_TIME_S = 60 * 60 * 3


class LocalWorkflowRunner(WorkflowRunner):
    timeout_s: float = 1  # in seconds

    def __init__(
        self,
        workflow: Workflow,
        io_mapper: WorkflowIOHandler,
        factory_spec: OperationFactoryConfig,
        update_state_callback: WorkflowCallback = NoOpStateChange,
        max_tries: int = 1,
    ):
        super().__init__(workflow, io_mapper, update_state_callback)
        self.runner = Worker(
            termination_grace_period_s=int(self.timeout_s),
            control_topic="",
            max_tries=max_tries,
            factory_spec=factory_spec,
        )

        self.dependency_resolver = OperationDependencyResolver()

    async def _run_op_impl(
        self, op: GraphNodeType, input: OpIOType, run_id: UUID, subtask_idx: int
    ) -> OpIOType:
        try:
            message = WorkMessageBuilder.build_execute_request(run_id, "", op.spec, input)
            self.runner.current_message = message
            stac = OpIOConverter.deserialize_input(input)
            dependencies = self.dependency_resolver.resolve(op.spec)
            message = WorkMessageBuilder.add_cache_info_to_execute_request(
                cast(ExecuteRequestMessage, message),
                CacheInfo(op.spec.name, op.spec.version, stac, dependencies),
            )
            content = message.content
            assert isinstance(content, CacheInfoExecuteRequestContent)
            await self._report_state_change(
                WorkflowChange.SUBTASK_RUNNING, task=op.name, subtask_idx=subtask_idx
            )
            out = self.runner.run_op_with_retry(content, run_id, MAX_OP_EXECUTION_TIME_S)
            await self._report_state_change(
                WorkflowChange.SUBTASK_FINISHED, task=op.name, subtask_idx=subtask_idx
            )
            return out
        except Exception as e:
            self.logger.exception(f"Failed to run operation {op.name}")
            await self._report_state_change(
                WorkflowChange.SUBTASK_FAILED, task=op.name, subtask_idx=subtask_idx, reason=str(e)
            )
            raise
        finally:
            self.runner.current_message = None
