from typing import List
from unittest import TestCase

import yaml
from azure.identity import AzureCliCredential

from vibe_agent.ops import OperationFactoryConfig, OpIOType
from vibe_agent.storage import StorageConfig
from vibe_common.secret_provider import AzureSecretProviderConfig
from vibe_server.workflow.runner import WorkflowRunner
from vibe_server.workflow.runner.task_io_handler import WorkflowIOHandler
from vibe_server.workflow.workflow import Workflow

from ..local_runner import LocalWorkflowRunner


class WorkflowTestHelper:
    @staticmethod
    def get_groundtruth_for_workflow(workflow_path: str) -> List[str]:
        with open(workflow_path) as yaml_file:
            workflow_def = yaml.safe_load(yaml_file)

        return workflow_def["sinks"]

    @staticmethod
    def verify_workflow_result(workflow_path: str, result: OpIOType):
        case = TestCase()
        expected_output_names = WorkflowTestHelper.get_groundtruth_for_workflow(workflow_path)

        assert len(expected_output_names) == len(result.keys())
        case.assertCountEqual(result.keys(), expected_output_names)
        for value in result.values():
            assert isinstance(value, dict) or isinstance(value, list)
            assert len(result) > 0

    @staticmethod
    def gen_workflow(
        workflow_path: str,
        storage_spec: StorageConfig,
    ) -> WorkflowRunner:
        factory_spec = OperationFactoryConfig(
            storage_spec, AzureSecretProviderConfig(credential=AzureCliCredential())
        )
        workflow = Workflow.build(workflow_path)
        io_mapper = WorkflowIOHandler(workflow)
        runner = LocalWorkflowRunner.build(
            io_mapper=io_mapper,
            factory_spec=factory_spec,
            workflow=workflow,
        )
        runner.runner.is_workflow = lambda *args, **kwargs: False  # type: ignore

        return runner
