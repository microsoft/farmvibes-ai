# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4 as uuid

import pytest
import requests
from fastapi.testclient import TestClient

from vibe_common.constants import CONTROL_STATUS_PUBSUB, WORKFLOW_REQUEST_PUBSUB_TOPIC
from vibe_common.messaging import WorkflowCancellationMessage
from vibe_common.statestore import StateStore
from vibe_core.data.core_types import InnerIOType
from vibe_core.data.utils import StacConverter, deserialize_stac
from vibe_core.datamodel import RunConfig, RunConfigInput, RunDetails, RunStatus
from vibe_server.href_handler import BlobHrefHandler, LocalHrefHandler
from vibe_server.server import TerravibesAPI, TerravibesProvider
from vibe_server.workflow.input_handler import build_args_for_workflow
from vibe_server.workflow.workflow import load_workflow_by_name


@pytest.fixture
def request_client():
    href_handler = LocalHrefHandler("/tmp")
    terravibes_app = TerravibesAPI(href_handler)
    client = TestClient(terravibes_app.versioned_wrapper)
    yield client


@pytest.fixture
def request_client_with_blob():
    href_handler = BlobHrefHandler()
    terravibes_app = TerravibesAPI(href_handler)
    client = TestClient(terravibes_app.versioned_wrapper)
    yield client


def test_list_workflows(request_client: requests.Session):
    url = "/v0/workflows"
    response = request_client.get(url)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

    for wfname in response.json():
        response = request_client.get(f"{url}/{wfname}")
        assert response.status_code == 200, (wfname, response.text)
        assert isinstance(response.json(), dict)
        fields = "name inputs outputs parameters description"
        for k in response.json():
            assert k in fields


def test_get_workflow_schema(request_client: requests.Session):
    url = "/v0/workflows"
    response = request_client.get(url)
    workflow = response.json()[0]
    url = f"{url}/{workflow}"
    response = request_client.get(url).json()
    assert isinstance(response, dict)
    assert all(k in response for k in ("name", "inputs", "outputs", "parameters", "description"))
    assert isinstance(response["name"], str)
    assert isinstance(response["inputs"], dict)
    assert isinstance(response["outputs"], dict)
    assert isinstance(response["parameters"], dict)
    assert isinstance(response["description"], dict)
    assert sorted(response["parameters"]) == sorted(response["description"]["parameters"])


def test_generate_api_documentation_page(request_client: requests.Session):
    response = request_client.get("/v0/docs")
    assert response.status_code == 200
    openapi_json = request_client.get("/v0/openapi.json")
    assert openapi_json.status_code == 200


@pytest.mark.parametrize("params", [None, {"param1": "new_param"}])
@patch("vibe_server.server.send", return_value="OK")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve", side_effect=lambda _: [])
@patch.object(StateStore, "retrieve_bulk", side_effect=lambda _: [])
def test_workflow_submission(
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    transaction: MagicMock,
    send: MagicMock,
    workflow_run_config: Dict[str, Any],
    params: Dict[str, Any],
    request_client: requests.Session,
):
    workflow_run_config["parameters"] = params
    response = request_client.post("/v0/runs", json=workflow_run_config)
    send.assert_called()
    assert send.call_args[0][0].content.parameters == params

    assert response.status_code == 201
    assert len(transaction.call_args.args[0]) == 2
    id = response.json()["id"]
    assert transaction.call_args.args[0][0]["value"][0] == id
    submitted_config = asdict(transaction.call_args.args[0][1]["value"])
    # Add some tasks here
    tasks = ["task1", "task2", "task3"]
    submitted_config["tasks"] = tasks
    retrieve_bulk.side_effect = [[submitted_config], [asdict(RunDetails()) for _ in tasks]]
    response = request_client.get(f"/v0/runs/{id}")
    assert response.json()["details"]["status"] == RunStatus.pending
    retrieved_task_details = response.json()["task_details"]
    assert len(retrieved_task_details) == len(tasks)
    assert all(retrieved_task_details[t]["status"] == RunStatus.pending for t in tasks)

    retrieve_bulk.side_effect = lambda _: [  # type: ignore
        asdict(transaction.call_args.args[0][1]["value"])
    ]
    response = request_client.get(f"/v0/runs/?ids={id}")
    assert response.status_code == 200
    assert len(response.json()) == 1


@patch.object(StateStore, "retrieve", side_effect=lambda _: [])
def test_no_workflow_runs(_, request_client: requests.Session):
    response = request_client.get("/v0/runs")
    assert response.status_code == 200
    assert len(response.json()) == 0


def test_invalid_workflow_submission(
    workflow_run_config: Dict[str, Any], request_client: requests.Session
):
    workflow_run_config["workflow"] = "invalid workflow"
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 400


def test_missing_field_workflow_submission(
    workflow_run_config: Dict[str, Any], request_client: requests.Session
):
    del workflow_run_config["user_input"]
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "type_error"


@patch.object(TerravibesProvider, "submit_work", side_effect=Exception("sorry"))
@patch.object(TerravibesProvider, "update_run_state")
@patch.object(TerravibesProvider, "list_runs_from_store", return_value=[])
def test_submit_local_workflows_with_broken_work_submission(
    _, __: Any, ___: Any, workflow_run_config: Dict[str, Any], request_client: requests.Session
):
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 500, response


@patch("vibe_server.server.send", return_value="OK")
@patch.object(TerravibesProvider, "submit_work")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve", side_effect=lambda _: [])
@patch.object(StateStore, "retrieve_bulk")
def test_workflow_submission_and_cancellation(
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    transaction: MagicMock,
    _: MagicMock,
    send: MagicMock,
    workflow_run_config: Dict[str, Any],
    request_client: requests.Session,
):
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 201
    assert len(transaction.call_args.args[0]) == 2
    id = response.json()["id"]
    assert transaction.call_args.args[0][0]["value"][0] == id

    response = request_client.post(f"/v0/runs/{id}/cancel")
    assert response.status_code == 202
    assert len(transaction.call_args.args[0]) == 2
    message = send.call_args.args[0]
    assert isinstance(message, WorkflowCancellationMessage)
    assert str(message.run_id) == id

    send.assert_called_with(
        message, "rest-api", CONTROL_STATUS_PUBSUB, WORKFLOW_REQUEST_PUBSUB_TOPIC
    )


@pytest.mark.parametrize("params", [None, {"param1": "new_param"}])
@patch.object(TerravibesProvider, "submit_work")
@patch.object(TerravibesProvider, "update_run_state")
@patch.object(StateStore, "retrieve")
@patch.object(StateStore, "retrieve_bulk", side_effect=lambda _: [])
def test_workflow_resubmission(
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    update_run_state: MagicMock,
    submit_work: MagicMock,
    params: Optional[Dict[str, Any]],
    workflow_run_config: Dict[str, Any],
    request_client: requests.Session,
):
    submitted_runs: List[RunConfig] = []
    first_run = {}

    def submit_work_effect(run: RunConfig):
        nonlocal submitted_runs
        submitted_runs.append(run)

    def update_run_state_effect(run_ids: List[str], new_run: RunConfig):
        nonlocal first_run
        first_run = asdict(new_run)

    submit_work.side_effect = submit_work_effect
    update_run_state.side_effect = update_run_state_effect

    workflow_run_config["parameters"] = params
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 201

    retrieve.side_effect = [first_run, []]
    response = request_client.post(f"/v0/runs/{uuid()}/resubmit")

    assert response.status_code == 201
    r1, r2 = submitted_runs
    for p in ("workflow", "user_input", "parameters", "name"):
        assert getattr(r1, p) == getattr(r2, p)
    assert r1.id != r2.id


@patch.object(StateStore, "retrieve")
def test_resubmission_of_missing_run(retrieve: MagicMock, request_client: requests.Session):
    def retrieve_effect(_):
        raise KeyError()

    retrieve.side_effect = retrieve_effect
    response = request_client.post(f"/v0/runs/{uuid()}/resubmit")
    assert response.status_code == 404


@patch.object(StateStore, "retrieve")
def test_cancelling_missing_run(retrieve: MagicMock, request_client: requests.Session):
    def retrieve_effect(_):
        raise KeyError()

    retrieve.side_effect = retrieve_effect

    response = request_client.post(f"/v0/runs/{uuid()}/cancel")
    assert response.status_code == 404


def test_getting_schema_of_missing_workflow(request_client: requests.Session):
    response = request_client.get("/v0/workflows/i-don't-exist")
    assert response.status_code == 404


def test_build_args_for_workflow_generates_valid_output(workflow_run_config: Dict[str, Any]):
    run_config = RunConfigInput(**workflow_run_config)
    inputs = load_workflow_by_name(cast(str, run_config.workflow)).inputs_spec
    out = build_args_for_workflow(run_config.user_input, list(inputs))

    def genitems(values: Union[InnerIOType, List[InnerIOType]]):
        if isinstance(values, list):
            for e in values:
                yield deserialize_stac(e)
        else:
            yield deserialize_stac(values)

    converter = StacConverter()
    for v in genitems([v for v in out.values()]):
        assert converter.from_stac_item(v) is not None


@pytest.mark.parametrize(
    "fields_exceptions",
    [
        ([], None),
        (["user_input.geojson"], None),
        (["user_input.geojson", "workflow"], None),
        (["user_input.doesnt_exist"], KeyError),
        (["something_else.doesnt_exist"], KeyError),
    ],
)
def test_summarize_runs(
    workflow_run_config: Dict[str, Any], fields_exceptions: Tuple[List[str], Optional[Exception]]
):
    href_handler = LocalHrefHandler("/tmp")
    provider = TerravibesProvider(href_handler)
    fields, exception = fields_exceptions
    run_config = RunConfig(
        **workflow_run_config,
        id=uuid(),
        details=RunDetails(),
        task_details={},
        spatio_temporal_json=None,
    )
    if exception is not None:
        with pytest.raises(exception):  # type: ignore
            provider.summarize_runs([run_config], fields)
    else:
        summary = provider.summarize_runs([run_config], fields)
        print(summary)
        if fields:
            for field in fields:
                if "doesnt" not in field:
                    assert field in summary[0]


@pytest.mark.parametrize("blob_df", [(True, type(None)), (False, int)])
def test_system_metrics(
    request_client: requests.Session,
    request_client_with_blob: requests.Session,
    blob_df: Tuple[bool, Any],
):
    blob, df_type = blob_df
    if blob:
        response = request_client_with_blob.get("/v0/system-metrics")
    else:
        response = request_client.get("/v0/system-metrics")

    assert response.status_code == 200

    metrics = response.json()
    for field in "load_avg cpu_usage free_mem used_mem total_mem disk_free".split():
        assert field in metrics

    assert all(isinstance(v, float) for v in metrics["load_avg"])
    assert isinstance(metrics["cpu_usage"], float)
    assert isinstance(metrics["free_mem"], int)
    assert isinstance(metrics["used_mem"], int)
    assert isinstance(metrics["total_mem"], int)
    assert isinstance(metrics["disk_free"], df_type)
