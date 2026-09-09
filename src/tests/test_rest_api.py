# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4 as uuid

import pytest
import requests
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from vibe_common.constants import CONTROL_STATUS_PUBSUB, RUNS_KEY, WORKFLOW_REQUEST_PUBSUB_TOPIC
from vibe_common.messaging import WorkflowCancellationMessage
from vibe_common.statestore import DEFAULT_BULK_PARALLELISM, StateStore, StateStoreConflictError
from vibe_core.data.core_types import InnerIOType
from vibe_core.data.utils import StacConverter, deserialize_stac
from vibe_core.datamodel import RunConfig, RunConfigInput, RunDetails, RunStatus
from vibe_core.security import API_TOKEN_ENV_VAR, REDACTED_VALUE
from vibe_server.href_handler import BlobHrefHandler, LocalHrefHandler
from vibe_server.orchestrator import Orchestrator
from vibe_server.server import TerravibesAPI, TerravibesProvider, require_api_token
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


@pytest.fixture
def authenticated_api(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv(API_TOKEN_ENV_VAR, "correct-token")
    app = TerravibesAPI(LocalHrefHandler("/tmp"))
    return TestClient(app.versioned_wrapper)


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v0/system-metrics"),
        ("GET", "/v0/workflows"),
        ("GET", "/v0/workflows/helloworld"),
        ("GET", f"/v0/runs/{uuid()}"),
        ("GET", "/v0/runs"),
        ("POST", f"/v0/runs/{uuid()}/cancel"),
        ("DELETE", f"/v0/runs/{uuid()}"),
        ("POST", f"/v0/runs/{uuid()}/resubmit"),
        ("POST", "/v0/runs"),
    ],
)
def test_api_token_protects_all_versioned_routes(
    authenticated_api: TestClient, method: str, path: str
):
    response = authenticated_api.request(method, path)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Basic wrong"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer wrong"},
        {"X-Forwarded-Authorization": "Bearer correct-token"},
        {"Proxy-Authorization": "Bearer correct-token"},
        {"X-API-Key": "correct-token"},
    ],
)
def test_api_token_rejects_malformed_wrong_and_proxy_credentials(
    authenticated_api: TestClient, headers: Dict[str, str]
):
    response = authenticated_api.get("/v0/workflows", headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_api_token_rejects_non_ascii_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_TOKEN_ENV_VAR, "correct-token")

    with pytest.raises(HTTPException) as error:
        require_api_token(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials="tëst")
        )

    assert error.value.status_code == 401


def test_empty_configured_api_token_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(API_TOKEN_ENV_VAR, "")
    app = TerravibesAPI(LocalHrefHandler("/tmp"))

    response = TestClient(app.versioned_wrapper).get("/v0/workflows")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_api_token_allows_bearer_and_leaves_health_and_docs_public(
    authenticated_api: TestClient,
):
    assert authenticated_api.get(
        "/v0/workflows", headers={"Authorization": "Bearer correct-token"}
    ).status_code == 200
    for path in ("/v0/", "/v0/docs", "/v0/openapi.json"):
        assert authenticated_api.get(path).status_code == 200

    openapi = authenticated_api.get("/v0/openapi.json").json()
    assert openapi["components"]["securitySchemes"]["Bearer"]["scheme"] == "bearer"
    assert openapi["paths"]["/workflows"]["get"]["security"] == [{"Bearer": []}]
    assert "security" not in openapi["paths"]["/"]["get"]


@pytest.mark.parametrize("params", [None, {"param1": "new_param"}])
@patch("vibe_server.server.send", return_value="OK")
@patch.object(StateStore, "store_if_absent")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve", side_effect=lambda _: [])
@patch.object(StateStore, "retrieve_bulk", side_effect=lambda _: [])
@patch.object(StateStore, "retrieve_with_etag")
def test_workflow_submission(
    retrieve_with_etag: MagicMock,
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    transaction: MagicMock,
    store_if_absent: MagicMock,
    send: MagicMock,
    workflow_run_config: Dict[str, Any],
    params: Dict[str, Any],
    request_client: requests.Session,
):
    retrieve_with_etag.side_effect = [([], None), ([], "1")]
    workflow_run_config["parameters"] = params
    response = request_client.post("/v0/runs", json=workflow_run_config)
    send.assert_called()
    assert send.call_args[0][0].content.parameters == params

    assert response.status_code == 201
    store_if_absent.assert_awaited_once_with(RUNS_KEY, [])
    assert len(transaction.call_args.args[0]) == 2
    assert transaction.call_args.args[0][0]["etag"] == "1"
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


@patch.object(TerravibesProvider, "submit_work")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve_with_etag")
def test_workflow_submission_retries_run_index_conflict(
    retrieve_with_etag: MagicMock,
    transaction: MagicMock,
    __: MagicMock,
    workflow_run_config: Dict[str, Any],
    request_client: requests.Session,
):
    concurrent_ids = ["concurrent"]
    retrieve_with_etag.side_effect = [(["older"], "1"), (concurrent_ids, "2")]

    async def conflict_after_commit(operations: List[Dict[str, Any]]):
        if transaction.await_count == 1:
            concurrent_ids.append(operations[0]["value"][-1])
            raise StateStoreConflictError("simulated ambiguous conflict")

    transaction.side_effect = conflict_after_commit
    response = request_client.post("/v0/runs", json=workflow_run_config)

    assert response.status_code == 201
    run_id = response.json()["id"]
    assert transaction.await_count == 2
    assert transaction.call_args_list[0].args[0][0]["value"] == ["older", run_id]
    assert transaction.call_args_list[1].args[0][0]["value"] == ["concurrent", run_id]
    assert transaction.call_args_list[1].args[0][0]["etag"] == "2"


@pytest.mark.parametrize("backend", ["redis", "cosmos"])
@pytest.mark.anyio
async def test_independent_providers_create_missing_run_index_without_lost_updates(
    workflow_run_config: Dict[str, Any],
    backend: str,
):
    state: Dict[str, Any] = {}
    version = 0
    missing_reads = 0
    all_read_missing = asyncio.Event()
    state_lock = asyncio.Lock()

    async def retrieve_with_etag(key: str, **_: Any):
        nonlocal missing_reads
        async with state_lock:
            if key in state:
                return deepcopy(state[key]), str(version)
            missing_reads += 1
            if missing_reads == 3:
                all_read_missing.set()
        await all_read_missing.wait()
        raise KeyError(key)

    async def retrieve(key: str):
        value, _ = await retrieve_with_etag(key)
        return value

    async def store_if_absent(key: str, value: Any):
        nonlocal version
        async with state_lock:
            if key in state:
                raise StateStoreConflictError(f"{backend} create conflict")
            state[key] = deepcopy(value)
            version += 1

    async def transaction(operations: List[Dict[str, Any]]):
        nonlocal version
        async with state_lock:
            index_operation = operations[0]
            etag = index_operation.get("etag")
            if RUNS_KEY not in state:
                raise RuntimeError(f"{backend} run index was not initialized")
            if etag != str(version):
                raise StateStoreConflictError(f"{backend} first-write conflict")
            for operation in operations:
                state[operation["key"]] = deepcopy(operation.get("value"))
            version += 1

    providers = [
        TerravibesProvider(LocalHrefHandler(".")),
        TerravibesProvider(LocalHrefHandler(".")),
    ]
    provider_stores = [StateStore(), StateStore()]
    for provider, store in zip(providers, provider_stores):
        store.retrieve_with_etag = AsyncMock(side_effect=retrieve_with_etag)
        store.store_if_absent = AsyncMock(side_effect=store_if_absent)
        store.transaction = AsyncMock(side_effect=transaction)
        provider.state_store = store

    orchestrator = Orchestrator()
    orchestrator.statestore.retrieve = AsyncMock(side_effect=retrieve)
    startup_store = AsyncMock()
    orchestrator.statestore.store = startup_store

    with patch.object(TerravibesProvider, "submit_work"):
        first, second, startup_runs = await asyncio.gather(
            providers[0].create_run(RunConfigInput(**deepcopy(workflow_run_config))),
            providers[1].create_run(RunConfigInput(**deepcopy(workflow_run_config))),
            orchestrator.get_unfinished_workflows(),
        )

    assert isinstance(first, JSONResponse)
    assert isinstance(second, JSONResponse)
    run_ids = {json.loads(response.body)["id"] for response in (first, second)}
    assert first.status_code == second.status_code == 201
    assert set(state[RUNS_KEY]) == run_ids
    assert len(run_ids) == 2
    assert startup_runs == []
    for store in provider_stores:
        store.store_if_absent.assert_awaited_once_with(RUNS_KEY, [])
        first_index_write = store.transaction.call_args_list[0].args[0][0]
        assert first_index_write["etag"]
        assert first_index_write["options"]["concurrency"] == "first-write"
        assert first_index_write["options"]["consistency"] == "strong"
    startup_store.assert_not_awaited()


@patch.object(StateStore, "retrieve", side_effect=lambda _: [])
def test_no_workflow_runs(_, request_client: requests.Session):
    response = request_client.get("/v0/runs")
    assert response.status_code == 200
    assert len(response.json()) == 0


@patch.object(StateStore, "retrieve_bulk", side_effect=KeyError("concurrent deletion"))
@patch.object(StateStore, "retrieve")
def test_compacted_and_missing_runs_do_not_break_bulk_listing(
    retrieve: MagicMock,
    _: MagicMock,
    request_client: requests.Session,
):
    compacted_id = str(uuid())
    missing_id = str(uuid())
    compacted = asdict(
        RunConfig(
            name="compacted",
            workflow="helloworld",
            parameters={"preserved": True},
            user_input={"input": "preserved"},
            id=compacted_id,
            details=RunDetails(status=RunStatus.done),
            task_details={},
            spatio_temporal_json=None,
            output="",
            history_compacted=True,
        )
    )

    def retrieve_effect(key: str):
        if key == compacted_id:
            return compacted
        raise KeyError(key)

    retrieve.side_effect = retrieve_effect
    response = request_client.get(
        "/v0/runs",
        params=[
            ("ids", compacted_id),
            ("ids", missing_id),
            ("fields", "id"),
            ("fields", "history_compacted"),
            ("fields", "task_details"),
            ("fields", "details.status"),
        ],
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": compacted_id,
            "history_compacted": True,
            "task_details": {},
            "details.status": RunStatus.done,
        }
    ]

    detail = request_client.get(f"/v0/runs/{compacted_id}")
    assert detail.status_code == 200
    assert detail.json()["history_compacted"] is True
    assert detail.json()["task_details"] == {}
    assert detail.json()["output"] == {}


@pytest.mark.anyio
async def test_bulk_fallback_bounds_concurrency_and_skips_missing_state():
    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    run_ids = [str(uuid()) for _ in range(DEFAULT_BULK_PARALLELISM * 2 + 3)]
    missing_run_id = run_ids[3]
    missing_task_run_id = run_ids[5]
    state: Dict[str, Any] = {}
    for run_id in run_ids:
        if run_id == missing_run_id:
            continue
        run = asdict(
            RunConfig(
                name=run_id,
                workflow="helloworld",
                parameters={},
                user_input={},
                id=run_id,
                details=RunDetails(status=RunStatus.done),
                task_details={},
                spatio_temporal_json=None,
            )
        )
        run["tasks"] = ["task"]
        state[run_id] = run
        if run_id != missing_task_run_id:
            state[f"{run_id}-task"] = asdict(RunDetails(status=RunStatus.done))

    active = 0
    peak = 0

    async def retrieve(key: str):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            if key not in state:
                raise KeyError(key)
            return deepcopy(state[key])
        finally:
            active -= 1

    provider.state_store.retrieve_bulk = AsyncMock(side_effect=KeyError("partial bulk result"))
    provider.state_store.retrieve = AsyncMock(side_effect=retrieve)

    runs = await provider.get_bulk_runs_by_id(run_ids)

    assert peak == DEFAULT_BULK_PARALLELISM
    assert [str(run.id) for run in runs] == [
        run_id for run_id in run_ids if run_id != missing_run_id
    ]
    runs_by_id = {str(run.id): run for run in runs}
    assert runs_by_id[missing_task_run_id].task_details == {}
    assert all(
        run.task_details["task"].status == RunStatus.done
        for run_id, run in runs_by_id.items()
        if run_id != missing_task_run_id
    )


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
@patch.object(
    TerravibesProvider,
    "list_runs_from_store_with_etag",
    side_effect=lambda: ([], "1"),
)
def test_submit_local_workflows_with_broken_work_submission(
    _, __: Any, ___: Any, workflow_run_config: Dict[str, Any], request_client: requests.Session
):
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 500, response


@patch("vibe_server.server.send", return_value="OK")
@patch.object(TerravibesProvider, "submit_work")
@patch.object(StateStore, "store_if_absent")
@patch.object(StateStore, "transaction")
@patch.object(StateStore, "retrieve", side_effect=lambda _: [])
@patch.object(StateStore, "retrieve_bulk")
@patch.object(StateStore, "retrieve_with_etag")
def test_workflow_submission_and_cancellation(
    retrieve_with_etag: MagicMock,
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    transaction: MagicMock,
    store_if_absent: MagicMock,
    submit_work: MagicMock,
    send: MagicMock,
    workflow_run_config: Dict[str, Any],
    request_client: requests.Session,
):
    retrieve_with_etag.side_effect = [([], None), ([], "1")]
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 201
    store_if_absent.assert_awaited_once_with(RUNS_KEY, [])
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
@patch.object(
    TerravibesProvider,
    "list_runs_from_store_with_etag",
    side_effect=lambda: ([], "1"),
)
@patch.object(StateStore, "retrieve")
@patch.object(StateStore, "retrieve_bulk", side_effect=lambda _: [])
def test_workflow_resubmission(
    retrieve_bulk: MagicMock,
    retrieve: MagicMock,
    _: MagicMock,
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

    def update_run_state_effect(run_ids: List[str], new_run: RunConfig, _: str):
        nonlocal first_run
        first_run = asdict(new_run)

    submit_work.side_effect = submit_work_effect
    update_run_state.side_effect = update_run_state_effect

    workflow_run_config["parameters"] = params
    response = request_client.post("/v0/runs", json=workflow_run_config)
    assert response.status_code == 201

    first_run["history_compacted"] = True
    first_run["output"] = ""
    first_run["task_details"] = {}
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
        (["spatio_temporal_json.doesnt_exist"], KeyError),
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
        with pytest.raises(exception, match="does not have field"):  # type: ignore
            provider.summarize_runs([run_config], fields)
    else:
        summary = provider.summarize_runs([run_config], fields)
        print(summary)
        if fields:
            for field in fields:
                if "doesnt" not in field:
                    assert field in summary[0]


@pytest.mark.anyio
async def test_run_parameter_responses_are_redacted_without_changing_state_or_resubmit():
    raw_parameters = {
        "visible": "plain",
        "Password": "hidden",
        "credentials": {"username": "hidden-user"},
        "nested": {
            "api_key": "also-hidden",
            "headers": {"Authorization": "Bearer workflow-secret"},
        },
    }
    raw_workflow = {"name": "helloworld", "parameters": {"pc_key": "workflow-hidden"}}
    run = RunConfig(
        name="sensitive",
        workflow=deepcopy(raw_workflow),
        parameters=deepcopy(raw_parameters),
        user_input={},
        id=uuid(),
        details=RunDetails(),
        task_details={},
        spatio_temporal_json=None,
    )
    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    provider.get_bulk_runs_by_id = AsyncMock(return_value=[run])  # type: ignore

    summary = provider.summarize_runs(
        [run],
        [
            "parameters",
            "parameters.Password",
            "parameters.credentials.username",
            "parameters.nested.api_key",
            "parameters.nested.headers.Authorization",
            "workflow",
            "workflow.parameters.pc_key",
        ],
    )[0]
    description = await provider.describe_run(run.id)

    assert summary["parameters"]["visible"] == "plain"
    assert summary["parameters"]["Password"] == REDACTED_VALUE
    assert summary["parameters"]["nested"]["api_key"] == REDACTED_VALUE
    assert summary["parameters.Password"] == REDACTED_VALUE
    assert summary["parameters.credentials.username"] == REDACTED_VALUE
    assert summary["parameters.nested.api_key"] == REDACTED_VALUE
    assert summary["parameters.nested.headers.Authorization"] == REDACTED_VALUE
    assert summary["workflow"]["parameters"]["pc_key"] == REDACTED_VALUE
    assert summary["workflow.parameters.pc_key"] == REDACTED_VALUE
    assert description["parameters"] == summary["parameters"]
    assert description["workflow"] == summary["workflow"]
    assert run.parameters == raw_parameters
    assert run.workflow == raw_workflow

    stored = asdict(run)
    provider.state_store.retrieve = AsyncMock(return_value=stored)
    provider.create_run = AsyncMock(return_value={"id": uuid()})  # type: ignore
    await provider.resubmit_run(run.id)
    resubmitted = provider.create_run.call_args.args[0]

    assert resubmitted.parameters == raw_parameters
    assert stored["parameters"] == raw_parameters
    assert stored["workflow"] == raw_workflow


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
