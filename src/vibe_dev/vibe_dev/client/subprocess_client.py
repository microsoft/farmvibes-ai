# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from typing import Any, List, Tuple, cast

from hydra_zen import instantiate
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_agent.ops import OperationFactoryConfig, OpIOType
from vibe_agent.storage import LocalFileAssetManagerConfig, LocalStorageConfig
from vibe_agent.storage.storage import Storage
from vibe_common.input_handlers import gen_stac_item_from_bounds
from vibe_common.secret_provider import AzureSecretProviderConfig
from vibe_core.client import Client
from vibe_core.data import BaseVibeDict, StacConverter
from vibe_core.data.utils import deserialize_stac
from vibe_core.datamodel import RunStatus, WorkflowRun
from vibe_server.workflow import list_workflows
from vibe_server.workflow.runner.task_io_handler import WorkflowIOHandler
from vibe_server.workflow.workflow import Workflow, load_workflow_by_name

from ..local_runner import LocalWorkflowRunner, WorkflowCallback, WorkflowChange


class SubprocessWorkflowRun(WorkflowRun):
    def __init__(self):
        self._output: BaseVibeDict = {}
        self._status = RunStatus.pending
        self._reason = ""

    def _workflow_callback(self) -> WorkflowCallback:
        async def callback(change: WorkflowChange, **kwargs: Any):
            if change == WorkflowChange.WORKFLOW_STARTED:
                self._status = RunStatus.running
            elif change == WorkflowChange.WORKFLOW_FINISHED:
                self._status = RunStatus.done
            elif change == WorkflowChange.WORKFLOW_FAILED:
                self._status = RunStatus.failed
                self._reason = kwargs["reason"]

        return callback

    @property
    def status(self) -> str:
        if self._status == RunStatus.failed:
            return f"{self._status}: {self._reason}"
        return self._status

    @property
    def output(self) -> BaseVibeDict:
        return self._output


class SubprocessClient(Client):
    """
    LocalWorkflowRunner wrapper that runs the workflow and retrieves results as DataVibe.
    """

    def __init__(
        self,
        factory_spec: OperationFactoryConfig,
        raise_exception: bool,
    ):
        self.factory_spec = factory_spec
        self.converter = StacConverter()
        self.storage: Storage = instantiate(factory_spec.storage)
        self.raise_exception = raise_exception

    def _deserialize_to_datavibe(self, workflow_output: OpIOType) -> BaseVibeDict:
        stac_items = {k: deserialize_stac(v) for k, v in workflow_output.items()}
        retrieved = self.storage.retrieve(stac_items)
        vibe_data = {k: self.converter.from_stac_item(v) for k, v in retrieved.items()}
        return vibe_data

    async def run(
        self, workflow_name: str, geometry: BaseGeometry, time_range: Tuple[datetime, datetime]
    ) -> WorkflowRun:
        output = SubprocessWorkflowRun()
        callback = output._workflow_callback()
        if workflow_name in self.list_workflows():
            # Load workflow by it's name
            workflow = load_workflow_by_name(workflow_name)
        else:
            # Assume it's the path to a YAML file
            workflow = Workflow.build(workflow_name)

        runner = LocalWorkflowRunner.build(
            workflow,
            io_mapper=WorkflowIOHandler(workflow),
            factory_spec=self.factory_spec,
            update_state_callback=callback,
        )

        stac_item_dict = gen_stac_item_from_bounds(
            shpg.mapping(geometry),  # type: ignore
            *time_range,
        )
        input_spec = cast(OpIOType, {k: stac_item_dict for k in runner.workflow.inputs_spec})
        try:
            runner_output = await runner.run(input_spec)
            output._output = self._deserialize_to_datavibe(runner_output)
            await callback(WorkflowChange.WORKFLOW_FINISHED)
        except Exception as e:
            await callback(WorkflowChange.WORKFLOW_FAILED, reason=str(e))
            if self.raise_exception:
                raise
        return output

    def list_workflows(self) -> List[str]:
        return list_workflows()


def get_default_subprocess_client(cache_dir: str) -> SubprocessClient:
    tmp_asset_path = os.path.join(cache_dir, "assets")
    storage_config = LocalStorageConfig(
        local_path=cache_dir, asset_manager=LocalFileAssetManagerConfig(tmp_asset_path)
    )
    factory_spec = OperationFactoryConfig(storage_config, AzureSecretProviderConfig())
    return SubprocessClient(factory_spec, False)
