from dataclasses import asdict
from datetime import datetime
from os.path import join as j
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from vibe_common.statestore import StateStore
from vibe_core.client import FarmvibesAiClient
from vibe_core.data import ADMAgSeasonalFieldInput
from vibe_core.datamodel import RunDetails
from vibe_server.href_handler import LocalHrefHandler
from vibe_server.orchestrator import WorkflowStateUpdate
from vibe_server.server import TerravibesAPI, TerravibesProvider
from vibe_server.workflow.runner import WorkflowChange
from vibe_server.workflow.spec_parser import WorkflowParser, get_workflow_dir


@pytest.fixture
def rest_client():
    href_handler = LocalHrefHandler("/tmp")
    terravibes_app = TerravibesAPI(href_handler)
    client = TestClient(terravibes_app.versioned_wrapper)
    url_string = str(client.base_url)
    rest_client = FarmvibesAiClient(url_string)
    rest_client.session = client
    rest_client.session.headers.update(rest_client.default_headers)
    yield rest_client


@pytest.fixture
def the_polygon():
    return Polygon(
        [
            [-88.068487, 37.058836],
            [-88.036059, 37.048687],
            [-88.012895, 37.068984],
            [-88.026622, 37.085711],
            [-88.062482, 37.081461],
            [-88.068487, 37.058836],
        ]
    )


@patch("vibe_server.server.list_existing_workflows")
@patch("vibe_server.server.TerravibesProvider.list_workflows")
@pytest.mark.anyio
async def test_list_workflows(
    list_workflows: MagicMock,
    list_existing_workflows: MagicMock,
    rest_client: FarmvibesAiClient,
):
    list_workflows.return_value = list_existing_workflows.return_value = "a/b c".split()
    workflows = rest_client.list_workflows()  # type: ignore
    assert workflows
    assert len(workflows) == len(await list_workflows())


@patch.object(StateStore, "retrieve", side_effect=lambda _: [])
def test_empty_list_runs(_, rest_client: FarmvibesAiClient):
    runs = rest_client.list_runs()
    assert not runs


@pytest.mark.parametrize("workflow", ["helloworld", j(get_workflow_dir(), "helloworld.yaml")])
@pytest.mark.parametrize("params", [None, {}, {"param1": 1}])
@patch.object(TerravibesProvider, "submit_work")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve")
@patch.object(StateStore, "retrieve_bulk")
@patch("vibe_server.server.list_existing_workflows")
@patch("vibe_server.server.build_args_for_workflow")
@patch("vibe_server.server.validate_workflow_input")
def test_submit_run(
    validate: MagicMock,
    build_args: MagicMock,
    list_existing_workflows: MagicMock,
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    transaction: MagicMock,
    _: MagicMock,
    rest_client: FarmvibesAiClient,
    the_polygon: Polygon,
    params: Optional[Dict[str, Any]],
    workflow: str,
    fake_ops_dir: str,
):
    first_retrieve_call = True

    def retrieve_side_effect(_):
        nonlocal first_retrieve_call
        if first_retrieve_call:
            first_retrieve_call = False
            return []
        return asdict(transaction.call_args.args[0][1]["value"])

    def bulk_side_effect(_):
        return [retrieve_side_effect(_)]

    retrieve.side_effect = retrieve_side_effect
    retrieve_bulk.side_effect = bulk_side_effect

    list_existing_workflows.return_value = ["a/b", "c", "helloworld"]
    with patch("vibe_server.workflow.spec_parser.DEFAULT_OPS_DIR", fake_ops_dir):
        run = rest_client.run(
            (workflow if "yaml" not in workflow else WorkflowParser._load_workflow(workflow)),
            "test-run",
            geometry=the_polygon,
            time_range=(datetime(2021, 2, 1), datetime(2021, 2, 2)),
            parameters=params,
        )
    assert run
    assert run.parameters == params
    build_args.assert_called()
    validate.assert_called()


@patch.object(TerravibesProvider, "submit_work")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve")
@patch.object(StateStore, "retrieve_bulk")
def test_submit_base_vibe_run(
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    transaction: MagicMock,
    _: MagicMock,
    rest_client: FarmvibesAiClient,
):
    party_id = "fake-party-id"
    seasonal_field_id = "fake-seasonal-field-id"
    input_data = ADMAgSeasonalFieldInput(
        party_id=party_id,
        seasonal_field_id=seasonal_field_id,
    )

    first_retrieve_call = True

    def retrieve_side_effect(_):
        nonlocal first_retrieve_call
        if first_retrieve_call:
            first_retrieve_call = False
            return []
        return asdict(transaction.call_args.args[0][1]["value"])

    def bulk_side_effect(_):
        return [retrieve_side_effect(_)]

    retrieve.side_effect = retrieve_side_effect
    retrieve_bulk.side_effect = bulk_side_effect

    run = rest_client.run(
        "data_ingestion/admag/admag_seasonal_field",
        "whatever",
        input_data=input_data,
    )
    assert run


@pytest.mark.parametrize("workflow", ["helloworld", j(get_workflow_dir(), "helloworld.yaml")])
@pytest.mark.parametrize("params", [None, {}, {"param1": 1}])
@patch.object(TerravibesProvider, "submit_work")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve")
@patch.object(StateStore, "retrieve_bulk")
@patch("vibe_common.statestore.StateStore.store")
@patch("vibe_server.server.list_existing_workflows")
@patch("vibe_server.server.build_args_for_workflow")
@patch("vibe_server.server.validate_workflow_input")
@pytest.mark.anyio
async def test_monitor_run_with_none_datetime_fields(
    validate: MagicMock,
    build_args: MagicMock,
    list_existing_workflows: MagicMock,
    store: MagicMock,
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    transaction: MagicMock,
    _: MagicMock,
    rest_client: FarmvibesAiClient,
    the_polygon: Polygon,
    params: Optional[Dict[str, Any]],
    workflow: str,
    fake_ops_dir: str,
):
    first_retrieve_call = True
    run_config: Optional[Dict[str, Any]] = None

    def store_side_effect(_: Any, obj: Any):
        nonlocal run_config
        run_config = obj

    def retrieve_side_effect(_):
        nonlocal first_retrieve_call, run_config
        if first_retrieve_call:
            first_retrieve_call = False
            return []

        if run_config is None:
            run_config = asdict(transaction.call_args.args[0][1]["value"])
            if not run_config["task_details"]:
                run_config["task_details"]["hello"] = asdict(RunDetails())
        return run_config

    def bulk_side_effect(_):
        return [retrieve_side_effect(_)]

    store.side_effect = store_side_effect
    retrieve.side_effect = retrieve_side_effect
    retrieve_bulk.side_effect = bulk_side_effect

    list_existing_workflows.return_value = ["a/b", "c", "helloworld"]
    with patch("vibe_server.workflow.spec_parser.DEFAULT_OPS_DIR", fake_ops_dir):
        run = rest_client.run(
            (workflow if "yaml" not in workflow else WorkflowParser._load_workflow(workflow)),
            "test-run",
            geometry=the_polygon,
            time_range=(datetime(2021, 2, 1), datetime(2021, 2, 2)),
            parameters=params,
        )
        assert run
        assert run.parameters == params
        build_args.assert_called()
        validate.assert_called()

        updater = WorkflowStateUpdate(UUID(run.id))
        await updater(WorkflowChange.WORKFLOW_STARTED, tasks=["hello"])

        assert run.task_details

        await updater(WorkflowChange.WORKFLOW_FINISHED)
        run.monitor(1, 0)


def test_system_metrics(rest_client: FarmvibesAiClient):
    metrics = rest_client.get_system_metrics()
    assert metrics
    assert metrics["disk_free"] is not None
