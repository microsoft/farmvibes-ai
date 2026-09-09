# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import hmac
import logging
import os
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from datetime import datetime
from enum import auto
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
    Union,
    _type_repr,  # type: ignore
    cast,
)
from uuid import UUID, uuid4

import debugpy
import psutil
import pydantic
import requests
import uvicorn
import yaml
from dapr.conf import settings
from fastapi import Body, FastAPI, HTTPException, Path, Query, Security, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_versioning import VersionedFastAPI, version
from hydra_zen import instantiate
from opentelemetry import trace
from starlette.middleware.cors import CORSMiddleware
from strenum import StrEnum

from vibe_common.constants import (
    ALLOWED_ORIGINS,
    CONTROL_STATUS_PUBSUB,
    DEFAULT_SECRET_STORE_NAME,
    RUNS_KEY,
    WORKFLOW_REQUEST_PUBSUB_TOPIC,
)
from vibe_common.dapr import dapr_ready
from vibe_common.messaging import WorkMessageBuilder, send
from vibe_common.secret_provider import DaprSecretConfig
from vibe_common.statestore import (
    StateStore,
    StateStoreConflictError,
    TransactionOperation,
)
from vibe_common.telemetry import (
    add_span_attributes,
    add_trace,
    setup_telemetry,
    update_telemetry_context,
)
from vibe_core.datamodel import (
    SUMMARY_DEFAULT_FIELDS,
    Message,
    MetricsDict,
    RunConfig,
    RunConfigInput,
    RunConfigUser,
    RunDetails,
    RunStatus,
    SpatioTemporalJson,
)
from vibe_core.logconfig import LOG_BACKUP_COUNT, MAX_LOG_FILE_BYTES, configure_logging
from vibe_core.security import (
    API_TOKEN_ENV_VAR,
    BEARER_SCHEME,
    REDACTED_VALUE,
    redact_sensitive,
)

from .href_handler import BlobHrefHandler, HrefHandler, LocalHrefHandler
from .workflow import get_workflow_path, workflow_from_input
from .workflow import list_workflows as list_existing_workflows
from .workflow.input_handler import (
    build_args_for_workflow,
    patch_workflow_sources,
    validate_workflow_input,
)
from .workflow.parameter import ParameterResolver
from .workflow.workflow import Workflow

RUN_CONFIG_SUBMISSION_EXAMPLE: Final[Dict[str, Any]] = {
    "name": "example workflow run for sample region",
    "workflow": "helloworld",
    "parameters": {},
    "user_input": {
        "start_date": "2021-02-02T00:00:00Z",
        "end_date": "2021-08-02T00:00:00Z",
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-88.068487, 37.058836],
                                [-88.036059, 37.048687],
                                [-88.012895, 37.068984],
                                [-88.026622, 37.085711],
                                [-88.062482, 37.081461],
                                [-88.068487, 37.058836],
                            ]
                        ],
                    },
                }
            ],
        },
    },
}
MOUNT_DIR: Final[str] = "/mnt"
RunList = Union[List[str], List[Dict[str, Any]], JSONResponse]
WorkflowList = Union[List[str], Dict[str, Any], JSONResponse]
CreateRunResponse = Union[Dict[str, Union[UUID, str]], JSONResponse]
RUN_INDEX_UPDATE_RETRIES = 3
bearer_auth = HTTPBearer(auto_error=False, scheme_name=BEARER_SCHEME)


def require_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
) -> None:
    """Require the configured API token when remote authentication is enabled."""

    expected_token = os.getenv(API_TOKEN_ENV_VAR)
    if expected_token is None:
        return
    if (
        credentials is not None
        and credentials.scheme.casefold() == BEARER_SCHEME.casefold()
        and hmac.compare_digest(
            credentials.credentials.encode(), expected_token.encode()
        )
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing bearer token",
        headers={"WWW-Authenticate": BEARER_SCHEME},
    )


class WorkflowReturnFormat(StrEnum):
    description = auto()
    yaml = auto()


class TerravibesProvider:
    state_store: StateStore
    logger: logging.Logger
    href_handler: HrefHandler

    def __init__(self, href_handler: HrefHandler):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.state_store = StateStore()
        self.href_handler = href_handler
        self.run_index_lock = asyncio.Lock()

    @add_trace
    def summarize_runs(self, runs: List[RunConfig], fields: List[str] = SUMMARY_DEFAULT_FIELDS):
        """Summarizes a list of runs given a list of fields.

        Supports accessing members deeper in the object by using dots to separate levels.
        For example, to extract the "status" member from "details", use "details.status".
        """

        sources = [redact_sensitive(asdict(run)) for run in runs]
        summarized_runs = [{k: v for k, v in source.items() if k in fields} for source in sources]
        for field in fields:
            if "." not in field:
                continue
            for i, src in enumerate(sources):
                prefixes, suffix = field.rsplit(".", maxsplit=1)
                obj = src
                try:
                    for prefix in prefixes.split("."):
                        if obj == REDACTED_VALUE:
                            break
                        obj = obj[prefix]
                    value = REDACTED_VALUE if obj == REDACTED_VALUE else obj[suffix]
                except (KeyError, TypeError) as e:
                    raise KeyError(
                        f"Workflow run with id {runs[i].id} does not have field {field}"
                    ) from e
                summarized_runs[i].update({field: value})
        return summarized_runs

    @add_trace
    def system_metrics(self) -> MetricsDict:
        """Returns a dict of system metrics."""

        load_avg: Tuple[float, float, float] = psutil.getloadavg()
        cpu_usage: float = psutil.cpu_percent()
        mem = psutil.virtual_memory()

        df: Optional[int]
        if isinstance(self.href_handler, BlobHrefHandler):
            df = None
        else:
            df = psutil.disk_usage(MOUNT_DIR).free

        return MetricsDict(
            load_avg=load_avg,
            cpu_usage=cpu_usage,
            free_mem=mem.free,
            used_mem=mem.used,
            total_mem=mem.total,
            disk_free=df,
        )

    async def root(self) -> Message:
        return Message(message="REST API server is running")

    @add_trace
    async def list_workflows(
        self,
        workflow: Optional[str] = None,
        return_format: str = WorkflowReturnFormat.description,
    ) -> WorkflowList:
        if not workflow:
            return [i for i in list_existing_workflows() if "private" not in i]
        try:
            if return_format == WorkflowReturnFormat.description:
                wf = Workflow.build(get_workflow_path(workflow))
                wf_spec = wf.workflow_spec
                param_resolver = ParameterResolver(wf_spec.workflows_dir, wf_spec.ops_dir)
                parameters = param_resolver.resolve(wf_spec)
                param_defaults = {k: v.default for k, v in parameters.items()}
                param_descriptions = {k: v.description for k, v in parameters.items()}
                description = wf.workflow_spec.description
                description.parameters = param_descriptions  # type: ignore
                return {
                    "name": wf.name,
                    "inputs": {k: _type_repr(v) for k, v in wf.inputs_spec.items()},
                    "outputs": {k: _type_repr(v) for k, v in wf.output_spec.items()},
                    "parameters": param_defaults,
                    "description": asdict(wf.workflow_spec.description),
                }
            elif return_format == WorkflowReturnFormat.yaml:
                with open(get_workflow_path(workflow)) as f:
                    yaml_content = yaml.safe_load(f)
                return yaml_content
            else:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=asdict(Message(f"Invalid return format: {return_format}")),
                )
        except FileNotFoundError:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=asdict(Message(f'Workflow "{workflow}" not found')),
            )
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=asdict(Message(f"Internal server error: {str(e)}")),
            )

    @add_trace
    async def list_runs(
        self,
        ids: Optional[List[UUID]],
        page: Optional[int],
        items: Optional[int],
        fields: Optional[List[str]],
    ) -> RunList:
        def paginate(
            things: List[Any], page: Optional[int] = 0, items: Optional[int] = 0
        ) -> List[Any]:
            if items is None or items <= 0:
                return things
            if page is None or page <= 0:
                page = 0
            return things[items * page : items * (page + 1)]

        ret: Union[List[str], List[Dict[str, Any]]] = []
        try:
            if ids is None:
                all_ids = await self.list_runs_from_store()
                if fields is None:
                    return all_ids

                ret = self.summarize_runs(await self.get_bulk_runs_by_id(all_ids), fields)
            else:
                ids = cast(List[Any], ids)
                if not all([isinstance(i, UUID) for i in ids]):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content=asdict(Message("Provided ids must be UUIDs")),
                    )
                if fields is None:
                    ret = self.summarize_runs(await self.get_bulk_runs_by_id(ids))
                else:
                    ret = self.summarize_runs(await self.get_bulk_runs_by_id(ids), fields)

            return paginate(ret, page, items)
        except (KeyError, IndexError):
            reason = f"Failed to get id(s) {ids}"
            self.logger.debug(reason)
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND, content=asdict(Message(reason))
            )

    async def describe_run(
        self,
        run_id: UUID = Path(..., title="The ID of the workflow execution to get."),
    ):
        try:
            run = (await self.get_bulk_runs_by_id([run_id]))[0]
            run_config_user = RunConfigUser.from_runconfig(run)
            response = jsonable_encoder(self.href_handler.handle(run_config_user))
            return redact_sensitive(response)
        except (KeyError, IndexError):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=asdict(Message(f'Workflow execution "{run_id}" not found')),
            )

    @add_trace
    async def cancel_run(
        self,
        run_id: UUID = Path(..., title="The ID of the workflow run to cancel."),
    ) -> JSONResponse:
        try:
            await self.state_store.retrieve(str(run_id))
        except KeyError:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=asdict(Message(f"Workflow execution {run_id} not found")),
            )

        message = WorkMessageBuilder.build_workflow_cancellation(run_id)

        response = send(
            message,
            "rest-api",
            CONTROL_STATUS_PUBSUB,
            WORKFLOW_REQUEST_PUBSUB_TOPIC,
        )

        if not response:
            raise RuntimeError("Failed to submit workflow cancellation request.")
        self.logger.debug(f"Successfully posted workflow cancellation request for run {run_id}")

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=asdict(Message(f"Requested cancellation of workflow run {run_id}")),
        )

    @add_trace
    async def delete_run(
        self,
        run_id: UUID = Path(..., title="The ID of the workflow run to delete."),
    ) -> JSONResponse:
        try:
            run_data = await self.state_store.retrieve(str(run_id))
        except KeyError:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=asdict(Message(f"Workflow execution {run_id} not found")),
            )

        run_config = RunConfig(**run_data)

        if not RunStatus.finished(run_config.details.status):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=asdict(Message("Cannot delete an unfinished workflow run.")),
            )

        message = WorkMessageBuilder.build_workflow_deletion(run_id)

        response = send(
            message,
            "rest-api",
            CONTROL_STATUS_PUBSUB,
            WORKFLOW_REQUEST_PUBSUB_TOPIC,
        )

        if not response:
            raise RuntimeError("Failed to submit workflow deletion request.")
        self.logger.debug(f"Successfully posted workflow deletion request for run {run_id}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=asdict(Message(f"Requested deletion of workflow run {run_id}")),
        )

    async def create_run(self, runConfig: RunConfigInput) -> CreateRunResponse:
        response: JSONResponse
        try:
            if (
                isinstance(runConfig.workflow, str)
                and runConfig.workflow not in list_existing_workflows()
            ):
                raise ValueError(f'Workflow "{runConfig.workflow}" unknown')

            workflow = workflow_from_input(runConfig.workflow)
            inputs_spec = workflow.inputs_spec
            # Build and validate inputs
            user_input = build_args_for_workflow(runConfig.user_input, list(inputs_spec))
            # Validate workflow inputs and potentially patch workflow for input fan-out
            validate_workflow_input(user_input, inputs_spec)
            patch_workflow_sources(user_input, workflow)

            new_id: Optional[str] = None
            new_run: Optional[RunConfig] = None
            async with self.run_index_lock:
                run_ids, runs_etag = await self.list_runs_from_store_with_etag()
                if runs_etag is None:
                    try:
                        await self.state_store.store_if_absent(RUNS_KEY, [])
                    except StateStoreConflictError:
                        pass
                    run_ids, runs_etag = await self.list_runs_from_store_with_etag()
                    if runs_etag is None:
                        raise RuntimeError("Failed to initialize the workflow run index")

                for attempt in range(RUN_INDEX_UPDATE_RETRIES):
                    if new_run is None:
                        new_id, new_run = self.create_new_run(runConfig, run_ids)
                    else:
                        retry_id = cast(str, new_id)
                        if retry_id not in run_ids:
                            run_ids.append(retry_id)
                    try:
                        await self.update_run_state(run_ids, new_run, runs_etag)
                        break
                    except StateStoreConflictError:
                        if attempt + 1 == RUN_INDEX_UPDATE_RETRIES:
                            raise
                        run_ids, runs_etag = await self.list_runs_from_store_with_etag()
                        if runs_etag is None:
                            raise RuntimeError("Workflow run index disappeared during update")

            assert new_run is not None
            add_span_attributes({"run_id": new_id})

            if new_id is None:
                raise RuntimeError("Failed to create new run id")

            # Update run id with parsed workflow and user input
            new_run.workflow = asdict(workflow.workflow_spec)
            new_run.user_input = user_input
            self.submit_work(new_run)

            response = JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content=asdict(
                    Message(
                        id=new_id,
                        location=f"/runs/{new_id}",
                        message="Workflow created and queued for execution",
                    )
                ),
            )
        except (
            ValueError,
            pydantic.ValidationError,
            requests.exceptions.RequestException,
        ) as e:
            self.logger.exception("Failed to submit workflow to worker")
            response = JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=asdict(
                    Message(f"Unable to run workflow with provided parameters. {str(e)}")
                ),
            )
        except FileNotFoundError as e:
            self.logger.exception("Failed to submit workflow")
            response = JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=asdict(Message(f"Unable to find workflow with name {str(e)}.")),
            )
        except Exception as e:
            self.logger.exception("Failed to update workflow state")
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=asdict(
                    Message(f"Unable to run workflow with provided parameters. {str(e)}")
                ),
            )
        return response

    @add_trace
    async def resubmit_run(self, run_id: UUID) -> CreateRunResponse:
        try:
            run = await self.state_store.retrieve(str(run_id))
        except KeyError:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=asdict(Message(f"Workflow run {run_id} not found")),
            )
        run_config = RunConfigInput(
            **{
                k: v
                for k, v in run.items()
                if k in ("name", "workflow", "parameters", "user_input")
            }
        )
        return await self.create_run(run_config)

    def create_new_run(self, workflow: RunConfigInput, run_ids: List[str]):
        new_id = str(uuid4())

        workflow_data = {k: v for k, v in asdict(workflow).items() if k != "user_input"}
        workflow_data["id"] = new_id
        workflow_data["details"] = RunDetails()  # type: ignore
        # Set workflow submission time
        workflow_data["details"].submission_time = datetime.utcnow()
        workflow_data["task_details"] = {}
        workflow_data["user_input"] = workflow.user_input
        if isinstance(workflow.user_input, SpatioTemporalJson):
            workflow_data["spatio_temporal_json"] = workflow.user_input
        else:
            workflow_data["spatio_temporal_json"] = None

        new_run = RunConfig(**workflow_data)
        run_ids.append(new_id)

        return new_id, new_run

    @add_trace
    async def update_run_state(
        self, run_ids: List[str], new_run: RunConfig, runs_etag: str
    ):
        runs_operation = cast(
            TransactionOperation,
            {
                "key": RUNS_KEY,
                "operation": "upsert",
                "value": run_ids,
                "etag": runs_etag,
                "options": {
                    "concurrency": "first-write",
                    "consistency": "strong",
                },
            },
        )
        await self.state_store.transaction(
            [
                runs_operation,
                cast(
                    TransactionOperation,
                    {
                        "key": str(new_run.id),
                        "operation": "upsert",
                        "value": new_run,
                    },
                ),
            ]
        )

    @add_trace
    async def list_runs_from_store(self) -> List[str]:
        try:
            return await self.state_store.retrieve(RUNS_KEY)
        except KeyError:
            # No workflows exist yet, ignore the failure
            return []

    @add_trace
    async def list_runs_from_store_with_etag(self) -> Tuple[List[str], Optional[str]]:
        try:
            return await self.state_store.retrieve_with_etag(RUNS_KEY, consistency="strong")
        except KeyError:
            return [], None

    @add_trace
    async def get_bulk_runs_by_id(self, run_ids: Union[List[str], List[UUID]]) -> List[RunConfig]:
        requested_ids = [str(run_id) for run_id in run_ids]
        run_state = await self.state_store.retrieve_bulk_existing(requested_ids)
        ordered_run_data = [
            run_state[run_id] for run_id in requested_ids if isinstance(run_state.get(run_id), dict)
        ]
        run_task_ids = [
            (str(run_data.get("id", "")), task)
            for run_data in ordered_run_data
            for task in run_data.get("tasks", [])
        ]
        task_keys = [f"{run_id}-{task}" for run_id, task in run_task_ids]
        task_state = await self.state_store.retrieve_bulk_existing(task_keys)
        for run_data in ordered_run_data:
            run_data.setdefault("task_details", {})
            run_id = str(run_data.get("id", ""))
            for task in run_data.get("tasks", []):
                task_key = f"{run_id}-{task}"
                if task_key in task_state:
                    run_data["task_details"][task] = task_state[task_key]

        runs = []
        for run_data in ordered_run_data:
            try:
                runs.append(RunConfig(**cast(Dict[str, Any], run_data)))
            except (TypeError, ValueError, pydantic.ValidationError):
                self.logger.warning(
                    f"Ignoring invalid or concurrently deleted workflow run "
                    f"{run_data.get('id', '<unknown>')}"
                )
        return runs

    def submit_work(self, new_run: RunConfig):
        assert isinstance(new_run.workflow, dict)
        assert isinstance(new_run.user_input, dict)
        message = WorkMessageBuilder.build_workflow_request(
            new_run.id, new_run.workflow, new_run.parameters, new_run.user_input
        )

        tracer = trace.get_tracer(__name__)
        update_telemetry_context(message.id)

        with tracer.start_as_current_span("submit-workflow"):
            response = send(
                message,
                "rest-api",
                CONTROL_STATUS_PUBSUB,
                WORKFLOW_REQUEST_PUBSUB_TOPIC,
            )

            if not response:
                raise RuntimeError("Failed to submit workflow for processing.")
            self.logger.debug(f"Successfully posted workflow message for run {new_run.id}")


class TerravibesAPI(FastAPI):
    uvicorn_config: uvicorn.Config
    terravibes: TerravibesProvider

    def __init__(
        self,
        href_handler: HrefHandler,
        allowed_origins: List[str] = ALLOWED_ORIGINS,
        host: str = "127.0.0.1",
        port: int = 8000,
        reload: bool = False,
        debug: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.terravibes = TerravibesProvider(href_handler)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"TerraVibes server: using {allowed_origins} as allowed origins")
        self.description = """# TerraVibes REST API

        TerraVibes is the execution engine of the FarmVibes platform, a
        containerized, distributed system that can run machine learning models at scale.
        TerraVibes uses Kubernetes for container orchestration and supports a variety of
        machine learning frameworks, as well as various data sources.

        With TerraVibes, farmers can run geospatial ingestion and machine learning models
        in the cloud or on-premises, depending on their needs. The platform is
        designed to be highly scalable and flexible, so userscan start with a
        small deployment and scale up as needed.

        ### Endpoints

        - `GET /`: Root endpoint
        - `GET /system-metrics`: Get system metrics

        ## Workflows

        The base computation unit users interact with is a workflow. A workflow is a
        collection of tasks that are arranged in a computational graph. Each task
        represents a single operation, and the graph represents the dependencies
        between the tasks. For example, a workflow might have a task that downloads
        satellite imagery, a task that runs a machine learning model on the imagery,
        and a task that uploads the results to a cloud storage bucket. The tasks are
        executed in parallel, and the results of each task are passed to the next task
        in the graph.

        ### Endpoints

        - `GET /workflows`: List all workflows
        - `GET /workflows/{workflow_name}`: Get a workflow by name, either as
          JSON description, or YAML graph implementation

        ## Runs

        Every time a workflow is executed, the API creates a new run. A run is a
        specific instance of a workflow, and it is uniquely identified by a run ID.
        The run ID is a UUID, and it is returned to the user when the workflow is
        submitted. The run ID can be used to query the status of the workflow, and it
        can be used to cancel the workflow.

        ### Endpoints

        - `GET /runs`: Lists all the workflow runs currently in the system.
        - `GET /runs/{run_id}`: Get information of a specific run.
        - `POST /runs`: Submit a new workflow run.
        - `POST /runs/{run_id}/cancel`: Cancel a workflow run.
        """

        self.openapi_tags = [
            {
                "name": "workflows",
                "description": (
                    "Operations on workflows, including listing, describing, "
                    "and obtaining workflow definition YAMLs."
                ),
                "externalDocs": {
                    "description": "FarmVibes.AI Workflow Documentation",
                    "url": (
                        "https://github.com/microsoft/farmvibes-ai/blob/main/documentation/"
                        "WORKFLOWS.md"
                    ),
                },
            },
            {
                "name": "runs",
                "description": (
                    "Operations on workflow runs, including submitting, listing, "
                    "describing, and cancelling runs.",
                ),
            },
        ]
        protected = [Security(require_api_token)]

        @self.get("/")
        @version(0)
        async def terravibes_root() -> Message:
            """Root endpoint."""
            return await self.terravibes.root()

        @self.get("/system-metrics", dependencies=protected)
        @version(0)
        async def terravibes_metrics() -> MetricsDict:
            """Get system metrics, including CPU usage, memory usage, and storage disk space."""
            return self.terravibes.system_metrics()

        @self.get(
            "/workflows", tags=["workflows"], response_model=None, dependencies=protected
        )
        @version(0)
        async def terravibes_list_workflows() -> WorkflowList:
            """List all workflows available in FarmVibes.AI."""
            return await self.terravibes.list_workflows()

        @self.get("/workflows/{workflow:path}", tags=["workflows"], dependencies=protected)
        @version(0)
        async def terravibes_describe_workflow(
            workflow: str = Path(
                ..., title="Workflow name", description="The name of the workflow to be described."
            ),
            return_format: str = Query(
                "description",
                title="Return format",
                description="The format to return the workflow in [description, yaml].",
            ),
        ):
            """Get a workflow by name, either as JSON description, or YAML graph implementation."""
            return await self.terravibes.list_workflows(workflow, return_format)

        @self.get("/runs", tags=["runs"], response_model=None, dependencies=protected)
        @version(0)
        async def terravibes_list_runs(
            ids: Optional[List[UUID]] = Query(
                None,
                description=(
                    "The list of run IDs to retrieve. If not provided, all runs are returned."
                ),
            ),
            page: Optional[int] = Query(0, description="The page number to retrieve."),
            items: Optional[int] = Query(0, description="The number of items per page."),
            fields: Optional[List[str]] = Query(
                None,
                description=(
                    "Fields to return alongside each run id. "
                    "If not provided, only run ids are returned."
                ),
            ),
        ) -> RunList:
            """List all the workflow runs currently in the system."""
            return await self.terravibes.list_runs(ids, page, items, fields)

        @self.get("/runs/{run_id}", tags=["runs"], dependencies=protected)
        @version(0)
        async def terravibes_describe_run(
            run_id: UUID = Path(
                ...,
                title="Run ID",
                description="The ID of the workflow execution to get.",
            ),
        ):
            """Get information of a specific run."""
            return await self.terravibes.describe_run(run_id)

        @self.post("/runs/{run_id}/cancel", tags=["runs"], dependencies=protected)
        @version(0)
        async def terravibes_cancel_run(
            run_id: UUID = Path(
                ...,
                title="Run ID",
                description="The ID of the workflow run to cancel.",
            ),
        ) -> JSONResponse:
            """Cancel a workflow run."""
            return await self.terravibes.cancel_run(run_id)

        @self.delete("/runs/{run_id}", tags=["runs"], dependencies=protected)
        @version(0)
        async def terravibes_delete_run(
            run_id: UUID = Path(
                ...,
                title="Run ID",
                description="The ID of the workflow run to delete.",
            ),
        ) -> JSONResponse:
            """Delete data associated with a workflow run (if not shared by other runs).

            For a detailed overview on how data is managed in FarmVibes.AI, please refer to the
            [documentation](https://microsoft.github.io/farmvibes-ai/docfiles/markdown/CACHE.html).
            """
            return await self.terravibes.delete_run(run_id)

        @self.post(
            "/runs/{run_id}/resubmit",
            tags=["runs"],
            response_model=None,
            dependencies=protected,
        )
        @version(0)
        async def terravibes_resubmit_run(
            run_id: UUID = Path(
                ...,
                title="Run ID",
                description="The ID of the workflow run to resubmit.",
            ),
        ) -> CreateRunResponse:
            """Resubmit a workflow run."""
            return await self.terravibes.resubmit_run(run_id)

        @self.post(
            "/runs",
            tags=["workflows", "runs"],
            response_model=None,
            dependencies=protected,
        )
        @version(0)
        async def terravibes_create_run(
            runConfig: RunConfigInput = Body(
                default=None,
                example=RUN_CONFIG_SUBMISSION_EXAMPLE,
                description="The configuration and inputs of the workflow run to submit.",
            ),
        ) -> CreateRunResponse:
            """Submit a new workflow run."""
            return await self.terravibes.create_run(runConfig)

        self.versioned_wrapper = VersionedFastAPI(
            self, version_format="{major}", prefix_format="/v{major}"
        )
        self.versioned_wrapper.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials="*" not in allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.uvicorn_config = uvicorn.Config(
            app=self.versioned_wrapper,
            host=host,
            port=port,
            reload=reload,
            debug=debug,
            log_config=None,
        )

    async def run(self):
        server = uvicorn.Server(self.uvicorn_config)
        await server.serve()


def build_href_handler(options: Namespace) -> HrefHandler:
    logger = logging.getLogger(f"{__name__}.build_href_handler")
    if options.terravibes_host_assets_dir:
        return LocalHrefHandler(options.terravibes_host_assets_dir)
    else:
        try:
            storage_account_connection_string = instantiate(
                DaprSecretConfig(
                    store_name=DEFAULT_SECRET_STORE_NAME,
                    secret_name=os.environ["BLOB_STORAGE_ACCOUNT_CONNECTION_STRING"],
                    key_name=os.environ["BLOB_STORAGE_ACCOUNT_CONNECTION_STRING"],
                )
            )
        except Exception:
            storage_account_connection_string = ""
            logger.exception(
                "Failed to load blob storage account connection string from Dapr secret store. "
                "Expect describing runs to fail due to an inability to resolve asset hrefs."
            )
        return BlobHrefHandler(
            connection_string=storage_account_connection_string,
        )


async def main() -> None:
    parser = ArgumentParser(description="TerraVibes 🌎 REST API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP address to listen on")
    parser.add_argument(
        "--port",
        type=int,
        default=int(settings.HTTP_APP_PORT),
        help="Port to listen on",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to enable debug support",
    )
    parser.add_argument(
        "--reload",
        default=False,
        action="store_true",
        help="Whether to reload the server on file change",
    )
    parser.add_argument(
        "--debugger-port",
        type=int,
        default=5678,
        help="The port on which to listen to the debugger",
    )
    parser.add_argument(
        "--terravibes-host-assets-dir",
        type=str,
        help="The asset directory on the host",
        default="",
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

    appname = "terravibes-rest-api"
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

    terravibes_api = TerravibesAPI(
        href_handler=build_href_handler(options),
        allowed_origins=ALLOWED_ORIGINS,
        host=options.host,
        port=options.port,
        reload=options.reload,
        debug=options.debug,
        title="TerraVibes 🌎 Spatial API",
        description="Low-code planetary analytics with powerful operators",
    )

    await start_service(terravibes_api)


@dapr_ready
async def start_service(terravibes_api: TerravibesAPI) -> None:
    await terravibes_api.run()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
