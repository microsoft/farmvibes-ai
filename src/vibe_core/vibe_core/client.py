"""FarmVibes.AI client.

This module provides a client for the FarmVibes.AI service, which allows users to interact with the
platform, such as inspecting workflow description and managing workflow runs (listing, creating,
monitoring, cancelling, etc.)
"""

import json
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from enum import auto
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast, overload
from urllib.parse import urljoin

import requests
import yaml
from dateutil.parser import ParserError, parse
from dateutil.tz import tzlocal
from dateutil.tz.tz import tzfile
from requests.exceptions import HTTPError
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry
from strenum import StrEnum

from vibe_core.data import BaseVibeDict, StacConverter
from vibe_core.data.core_types import BaseVibe
from vibe_core.data.json_converter import dump_to_json
from vibe_core.data.utils import deserialize_stac, serialize_input
from vibe_core.datamodel import (
    MetricsDict,
    MonitoredWorkflowRun,
    RunConfigInput,
    RunConfigUser,
    RunDetails,
    RunStatus,
    SpatioTemporalJson,
    TaskDescription,
    WorkflowRun,
)
from vibe_core.monitor import VibeWorkflowDocumenter, VibeWorkflowRunMonitor
from vibe_core.utils import ensure_list, format_double_escaped

FALLBACK_SERVICE_URL = "http://127.0.0.1:31108/"
"""Fallback URL for FarmVibes.AI service.

:meta hide-value:
"""

XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
"""Path to configuration file for FarmVibes.AI service.

:meta hide-value:
"""

FARMVIBES_AI_SERVICE_URL_PATH = os.path.join(XDG_CONFIG_HOME, "farmvibes-ai", "service_url")
"""Path to the local service URL file.

:meta hide-value:
"""

FARMVIBES_AI_REMOTE_SERVICE_URL_PATH = os.path.join(
    XDG_CONFIG_HOME, "farmvibes-ai", "remote_service_url"
)
"""Path to the local remote service URL file.

:meta hide-value:
"""

DISK_FREE_THRESHOLD_BYTES = 50 * 1024 * 1024 * 1024  # 50 GiB
"""Threshold for disk space in bytes."""

TASK_SORT_KEY = "submission_time"
"""Key for sorting tasks."""

T = TypeVar("T", bound=BaseVibe, covariant=True)
InputData = Union[Dict[str, Union[T, List[T]]], List[T], T]


class ClusterType(StrEnum):
    """An enumeration of cluster types."""

    remote = auto()
    local = auto()

    def client(self):
        """Create a client based on the cluster type.

        Returns:
            A :class:`FarmvibesAiClient` object based on the cluster type.

        """
        return FarmvibesAiClient(
            get_remote_service_url() if self.value == self.remote else get_local_service_url()
        )


class Client(ABC):
    """An abstract base class for clients."""

    @abstractmethod
    def list_workflows(self) -> List[str]:
        """List all available workflows.

        Returns:
            A list of workflow names.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        workflow: str,
        geometry: BaseGeometry,
        time_range: Tuple[datetime, datetime],
    ) -> WorkflowRun:
        """Run a workflow.

        Args:
            workflow: The name of the workflow to run.
            geometry: The geometry to run the workflow on.
            time_range: The time range to run the workflow on.

        Returns:
            A :class:`WorkflowRun` object.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError


class FarmvibesAiClient(Client):
    """A client for the FarmVibes.AI service.

    Args:
        baseurl: The base URL of the FarmVibes.AI service.

    """

    default_headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    """The default headers to use for requests to the FarmVibes.AI service."""

    def __init__(self, baseurl: str):
        """Instantiate a new FarmVibes.AI client."""
        self.baseurl = baseurl
        self.session = requests.Session()
        self.session.headers.update(self.default_headers)

    def _request(self, method: str, endpoint: str, *args: Any, **kwargs: Any):
        """Send a request to the FarmVibes.AI service and handle errors.

        Args:
            method: The HTTP method to use (e.g., 'GET' or 'POST').
            endpoint: The endpoint to request.
            args: Positional arguments to pass to :meth:`requests.Session.request`.
            kwargs: Keyword arguments to pass to :meth:`requests.Session.request`.

        Returns:
            The response from the FarmVibes.AI service.

        """
        response = self.session.request(method, urljoin(self.baseurl, endpoint), *args, **kwargs)
        try:
            r = json.loads(response.text)
        except json.JSONDecodeError:
            r = response.text
        try:
            response.raise_for_status()
        except HTTPError as e:
            error_message = r.get("message", "") if isinstance(r, dict) else r
            msg = f"{e}. {error_message}"
            raise HTTPError(msg, response=e.response)
        return cast(Any, r)

    def _form_payload(
        self,
        workflow: Union[str, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]],
        geometry: Optional[BaseGeometry],
        time_range: Optional[Tuple[datetime, datetime]],
        input_data: Optional[InputData[T]],
        run_name: str,
    ) -> Dict[str, Any]:
        """Form a payload dictionary for submitting a workflow run.

        Args:
            workflow: The name of the workflow to run or a dict containing the workflow definition.
            parameters: A dict of optional parameters to pass to the workflow.
                The keys and values depend on the specific workflow definition.
            geometry: The geometry to use for the input data.
                It must be a valid shapely geometry object (e.g., Point or Polygon).
                It will be converted to GeoJSON format internally.
                Alternatively it can be None if input_data is provided instead.
            time_range: The time range to use for the input data. It must be
                a tuple of two datetime objects representing the start and end dates.
                Alternatively it can be None if input_data is provided instead.
            input_data: The input data to use for the workflow run.
                It must be an instance of InputData or one of its subclasses
                (e.g., SpatioTemporalJson or SpatioTemporalRaster). Alternatively
                it can be None if geometry and time_range are provided instead.
            run_name: The name to assign to the workflow run.

        Returns:
            A dict containing the payload for submitting a workflow run.
            The keys are 'run_name', 'workflow', 'parameters', and 'user_input'.

        Note:
            Either (`geometry` and `time_range`) or (`input_data`) must be provided. Providing both
            will result in an error.

        """
        if input_data is not None:
            user_input = serialize_input(input_data)
        elif geometry is None or time_range is None:
            raise ValueError("Either `input_data` or `geometry` and `time_range` are required")
        else:
            geojson = {
                "features": [{"geometry": None, "type": "Feature"}],
                "type": "FeatureCollection",
            }
            geojson["features"][0]["geometry"] = shpg.mapping(geometry)
            user_input = SpatioTemporalJson(time_range[0], time_range[1], geojson)
        return asdict(RunConfigInput(run_name, workflow, parameters, user_input))

    def verify_disk_space(self):
        """Verify that there is enough disk space available for the cache.

        This method checks the system metrics returned by the FarmVibes.AI service
        and compares the disk free space with a predefined threshold. If the disk
        free space is below the threshold, a warning message is displayed to the user,
        suggesting to clear the cache.

        Note:
            The disk free space threshold is defined by :const:`DISK_FREE_THRESHOLD_BYTES`.

        Raises:
            :exc:`RuntimeWarning` if the disk space is low.

        """
        metrics = self.get_system_metrics()
        df = cast(Optional[int], metrics.get("disk_free", None))
        if df is not None and df < DISK_FREE_THRESHOLD_BYTES:
            warnings.warn(
                "The FarmVibes.AI cache is running low on disk space "
                f"and only has {df / 1024 / 1024 / 1024} GiB left. "
                "Please consider clearing the cache to free up space and "
                "to avoid potential failures.",
                category=RuntimeWarning,
            )

    def list_workflows(self) -> List[str]:
        """List all available workflows on the FarmVibes.AI service.

        This method returns a list of workflow names that can be used to
        submit workflow runs or to get more details about a specific workflow.

        Returns:
            A list of workflow names.

        """
        return self._request("GET", "v0/workflows")

    def describe_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Describe a workflow.

        This method returns a dictionary containing the description of a
        workflow, such as its inputs, outputs, parameters and tasks.

        Args:
            workflow_name: The name of the workflow to describe.
                It must be one of the names returned by list_workflows().

        Returns:
            A dictionary containing the workflow description.
            The keys are 'name', 'description', 'inputs', 'outputs' and 'parameters'.

        Note:
            The description is returned as a :class:`TaskDescription` object,
            which is a dataclass that represents the structure and
            properties of a workflow.

        """
        desc = self._request("GET", f"v0/workflows/{workflow_name}?return_format=description")

        param_descriptions = desc["description"]["parameters"]
        for p, d in param_descriptions.items():
            if isinstance(d, List):
                param_descriptions[p] = d[0]

        desc["description"] = TaskDescription(**desc["description"])
        return desc

    def get_system_metrics(self) -> MetricsDict:
        """Get system metrics from the FarmVibes.AI service.

        This method returns a dictionary containing various system metrics,
        such as CPU usage, memory usage and disk space.

        Returns:
            A dictionary containing system metrics.

        """
        metrics = self._request("GET", "v0/system-metrics")
        return MetricsDict(**metrics)

    def get_workflow_yaml(self, workflow_name: str) -> str:
        """Get the YAML definition of a workflow.

        This method returns a string containing the YAML definition of a
        workflow. The YAML definition specifies the name and operations of
        the workflow in a human-readable format.

        Args:
            workflow_name: The name of the workflow. It must be one
                of the names returned by list_workflows().

        Returns:
            The YAML definition of the workflow.

        """
        yaml_content = self._request("GET", f"v0/workflows/{workflow_name}?return_format=yaml")
        return yaml.dump(yaml_content, default_flow_style=False, default_style="", sort_keys=False)

    def cancel_run(self, run_id: str) -> str:
        """Cancel a workflow run.

        This method sends a request to the FarmVibes.AI service to cancel
        a workflow run that is in progress or pending. If the cancellation
        is successful, the workflow run status will be set to 'cancelled'.

        Args:
            run_id: The ID of the workflow run to cancel.

        Return:
            The message from the FarmVibes.AI service indicating whether
            the cancellation was successful or not.

        Note:
            The cancellation may take some time to take effect depending on
            the state of the workflow run and the service availability.

        Warnings:
            A workflow run that is already completed or failed cannot be cancelled.

        """
        return self._request("POST", f"v0/runs/{run_id}/cancel")["message"]

    def delete_run(self, run_id: str) -> str:
        """Delete a workflow run.

        This method sends a request to the FarmVibes.AI service to delete a completed workflow run
        (i.e. a run with the a status of 'done', 'failed', or 'cancelled'). If the deletion is
        successful, all cached data the workflow run produced that is not shared with other workflow
        runs will be deleted and status will be set to 'deleted'.

        Args:
            run_id: The ID of the workflow run to delete.

        Returns:
            The message from the FarmVibes.AI service indicating whether the deletion request
            was successful or not.

        Note:
            The deletion may take some time to take effect depending on the state of the workflow
            run and the service availability.

        Warnings:
            A workflow run that is in progress or pending cannot be deleted. A cancelled workflow
            run can be deleted. So, in order to delete workflow run that is in progress or pending,
            it first needs to be cancelled and then it can be deleted.

        """
        return self._request("DELETE", f"v0/runs/{run_id}")["message"]

    def describe_run(self, run_id: str) -> RunConfigUser:
        """Describe a workflow run.

        This method returns a RunConfigUser object containing the description of a
        workflow run, such as its name, status, inputs and outputs.

        Args:
            run_id: The ID of the workflow run to describe.

        Returns:
            A :class:`RunConfigUser` object containing the workflow run description.

        """
        response = self._request("GET", f"v0/runs/{run_id}")
        try:
            run = RunConfigUser(**response)
            for v in run.task_details.values():
                if v.subtasks is not None:
                    v.subtasks = [RunDetails(**i) for i in v.subtasks]
        except Exception as e:
            raise RuntimeError(f"Failed to parse description for run {run_id}: {e}") from e
        return run

    def document_workflow(self, workflow_name: str) -> None:
        """Print the documentation of a workflow.

        This method prints a formatted documentation of a workflow,
        including its name, description, inputs, outputs and parameters.

        Args:
            workflow_name: The name of the workflow to document.

        Note:
            The documentation is printed to stdout and can be redirected to
            other outputs if needed.

        """
        wf_dict = self.describe_workflow(workflow_name)

        documenter = VibeWorkflowDocumenter(
            name=workflow_name,
            sources=wf_dict["inputs"],
            sinks=wf_dict["outputs"],
            parameters=wf_dict["parameters"],
            description=wf_dict["description"],
        )

        documenter.print_documentation()

    def list_runs(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        fields: Optional[Union[str, List[str]]] = None,
    ):
        """List workflow runs on the FarmVibes.AI service.

        Args:
            ids: The IDs of the workflow runs to list.
                If None, all workflow runs will be listed.
            fields: The fields to return for each workflow run.
                If None, all fields will be returned.

        Returns:
            A list of workflow runs. Each run is represented by a dictionary
            with keys corresponding to the requested fields and values containing
            the field values.

        """
        ids = [f"ids={id}" for id in ensure_list(ids)] if ids is not None else []
        fields = [f"fields={field}" for field in ensure_list(fields)] if fields is not None else []
        query_str = "&".join(ids + fields)

        return self._request("GET", f"v0/runs?{query_str}")

    def get_run_by_id(self, id: str) -> "VibeWorkflowRun":
        """Get a workflow run by ID.

        This method returns a :class:`VibeWorkflowRun` object containing
        the details of a workflow run by its ID.

        Args:
            id: The ID of the workflow run to get.

        Returns:
            A :class:`VibeWorkflowRun` object.

        """
        fields = ["id", "name", "workflow", "parameters"]
        run = self.list_runs(id, fields=fields)[0]
        return VibeWorkflowRun(*(run[f] for f in fields), self)  # type: ignore

    def get_last_runs(self, n: int) -> List["VibeWorkflowRun"]:
        """Get the last 'n' workflow runs.

        This method returns a list of :class:`VibeWorkflowRun` objects containing
        the details of the last n workflow runs.

        Args:
            n: The number of workflow runs to get (with n>0).

        Returns:
            A list of :class:`VibeWorkflowRun` objects.

        """
        if n <= 0:
            raise ValueError(f"The number of runs (n) must be greater than 0. Got {n} instead.")

        last_runs = self.list_runs()[-n:]
        if not last_runs:
            raise ValueError("No past runs available.")
        elif len(last_runs) < n:
            logging.warning(
                f"Requested {n} runs, but only {len(last_runs)} are available. "
                "Returning all available runs."
            )
        return [self.get_run_by_id(run_id) for run_id in last_runs]

    def get_api_time_zone(self) -> tzfile:
        """Get the time zone of the FarmVibes.AI REST-API.

        This method returns a tzfile object representing the time zone of
        the FarmVibes.AI REST-API. The time zone is determined by parsing
        the 'date' header from the response of a GET request to the base URL
        of the service. If the 'date' header is missing or invalid, a warning
        is issued and the client time zone is used instead.

        Returns:
            The time zone of the FarmVibes.AI REST-API as a tzfile object.

        Note:
            The tzfile object is a subclass of datetime.tzinfo that represents
            a time zone using an Olson database file.

        """
        tz = tzlocal()
        response = self.session.request("GET", self.baseurl)
        try:
            dt = parse(response.headers["date"])
            tz = dt.tzinfo if dt.tzinfo is not None else tzlocal()
        except KeyError:
            warnings.warn(
                "Could not determine the time zone of the FarmVibes.AI REST-API. "
                "'date' header is missing from the response. "
                "Using the client time zone instead.",
                category=RuntimeWarning,
            )
        except ParserError:
            warnings.warn(
                "Could not determine the time zone of the FarmVibes.AI REST-API. "
                "Unable to parse the 'date' header from the response. "
                "Using the client time zone instead.",
                category=RuntimeWarning,
            )
        return cast(tzfile, tz)

    @overload
    def run(
        self,
        workflow: Union[str, Dict[str, Any]],
        name: str,
        *,
        geometry: BaseGeometry,
        time_range: Tuple[datetime, datetime],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "VibeWorkflowRun": ...

    @overload
    def run(
        self,
        workflow: Union[str, Dict[str, Any]],
        name: str,
        *,
        input_data: InputData[T],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "VibeWorkflowRun": ...

    def run(
        self,
        workflow: Union[str, Dict[str, Any]],
        name: str,
        *,
        geometry: Optional[BaseGeometry] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        input_data: Optional[InputData[T]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "VibeWorkflowRun":
        """Run a workflow.

        This method instantiates a workflow run using the provided data and parameters.

        Args:
            workflow: The name of the workflow to run or a dict containing
                the workflow definition.
            name: The name to assign to the workflow run.
            geometry: The geometry to use for the input data.
                It must be a valid shapely geometry object (e.g., Point or Polygon).
                It will be converted to GeoJSON format internally.
                Alternatively it can be None if input_data is provided instead.
            time_range: The time range to use for the input data. It must be
                a tuple of two datetime objects representing the start and end dates.
                Alternatively it can be None if input_data is provided instead.
            input_data: The input data to use for the workflow run.
                It must be an instance of InputData or one of its subclasses
                (e.g., SpatioTemporalJson or SpatioTemporalRaster). Alternatively
                it can be None if geometry and time_range are provided instead.
            parameters: A dict of optional parameters to pass to the workflow.
                The keys and values depend on the specific workflow definition.

        Returns:
            A :class:`VibeWorkflowRun` object.

        Note:
            Either (`geometry` and `time_range`) or (`input_data`) must be provided. Providing both
            will result in an error.

        """
        self.verify_disk_space()
        payload = dump_to_json(
            self._form_payload(workflow, parameters, geometry, time_range, input_data, name),
        )
        response = self._request("POST", "v0/runs", data=payload)
        return self.get_run_by_id(response["id"])

    def resubmit_run(self, run_id: str) -> "VibeWorkflowRun":
        """Resubmit a workflow run with the given run ID.

        Args:
            run_id: The ID of the workflow run to resubmit.

        Returns:
            The resubmitted workflow run.

        """
        self.verify_disk_space()
        response = self._request("POST", f"v0/runs/{run_id}/resubmit")
        return self.get_run_by_id(response["id"])

    def _loop_update_monitor_table(
        self,
        runs: List["VibeWorkflowRun"],
        monitor: VibeWorkflowRunMonitor,
        refresh_time_s: int,
        refresh_warnings_time_min: int,
        timeout_min: Optional[int],
    ):
        stop_monitoring = False
        time_start = last_warning_refresh = time.monotonic()

        with warnings.catch_warnings(record=True) as monitored_warnings:
            with monitor.live_context:
                while not stop_monitoring:
                    monitor.update_run_status(
                        [cast(MonitoredWorkflowRun, run) for run in runs],
                        [w.message for w in monitored_warnings],
                    )

                    time.sleep(refresh_time_s)
                    current_time = time.monotonic()

                    # Check for warnings every refresh_warnings_time_min minutes
                    if (current_time - last_warning_refresh) / 60.0 > refresh_warnings_time_min:
                        self.verify_disk_space()
                        last_warning_refresh = current_time

                    # Check for timeout
                    did_timeout = (
                        timeout_min is not None and (current_time - time_start) / 60.0 > timeout_min
                    )
                    stop_monitoring = (
                        all(
                            [
                                RunStatus.finished(r.status) or r.status == RunStatus.deleted
                                for r in runs
                            ]
                        )
                        or did_timeout
                    )

                # Update one last time to make sure we have the latest state
                monitor.update_run_status(
                    [cast(MonitoredWorkflowRun, run) for run in runs],
                    [w.message for w in monitored_warnings],
                )

    def monitor(
        self,
        runs: Union[List["VibeWorkflowRun"], "VibeWorkflowRun", int] = 1,
        refresh_time_s: int = 1,
        refresh_warnings_time_min: int = 5,
        timeout_min: Optional[int] = None,
        detailed_task_info: bool = False,
    ) -> None:
        """Monitor workflow runs.

        This method will block and print the status of the runs each refresh_time_s seconds,
        until the workflow runs finish or it reaches timeout_min minutes. It will also
        print warnings every refresh_warnings_time_min minutes.

        Args:
            runs: A list of workflow runs, a single run object, or an integer. The method will
                monitor the provided workflow runs. If a list of runs is provided, the method will
                provide a summarized table with the status of each run. If only one run is provided,
                the method will monitor that run directly. If an integer > 0 is provided, the method
                will fetch the respective last runs and provide the summarized monitor table.
            refresh_time_s: Refresh interval in seconds (defaults to 1 second).
            refresh_warnings_time_min: Refresh interval in minutes for updating
                the warning messages (defaults to 5 minutes).
            timeout_min: The maximum time to monitor the workflow run, in minutes.
                If not provided, the method will monitor indefinitely.
            detailed_task_info: If True, detailed information about task progress
                will be included in the output (defaults to False).

        Raises:
            ValueError: If no workflow runs are provided (empty list).

        """
        if isinstance(runs, int):
            runs = self.get_last_runs(runs)

        if isinstance(runs, VibeWorkflowRun):
            runs = [runs]

        runs = cast(List[VibeWorkflowRun], runs)

        if len(runs) == 0:
            raise ValueError("At least one workflow run must be provided.")

        monitor = VibeWorkflowRunMonitor(
            api_time_zone=self.get_api_time_zone(),
            detailed_task_info=detailed_task_info,
            multi_run=len(runs) > 1,
        )

        self._loop_update_monitor_table(
            runs, monitor, refresh_time_s, refresh_warnings_time_min, timeout_min
        )


class VibeWorkflowRun(WorkflowRun, MonitoredWorkflowRun):
    """Represent a workflow run in FarmVibes.AI.

    Args:
        id: The ID of the workflow run.
        name: The name of the workflow run.
        workflow: The name of the workflow associated to the run.
        parameters: The parameters associated to the workflow run, as a dict.
            The keys and values depend on the specific workflow definition.
        client: An instance of the :class:`FarmVibesAiClient` class.
    """

    wait_s = 10

    def __init__(
        self,
        id: str,
        name: str,
        workflow: str,
        parameters: Dict[str, Any],
        client: FarmvibesAiClient,
    ):
        """Instantiate a new VibeWorkflowRun."""
        self.id = id
        self.name = name
        self.workflow = workflow
        self.parameters = parameters
        self.client = client
        self._status = RunStatus.pending
        self._reason = ""
        self._output = None
        self._task_details = None

    def _convert_output(self, output: Dict[str, Any]) -> BaseVibeDict:
        """Convert the output of the workflow run to a :class:`BaseVibeDict`.

        This method takes the output of the workflow run as a dictionary and
        converts each value to a DataVibe object using a StacConverter object.
        It returns a new dictionary with the same keys and converted values.

        Args:
            output: The output of the workflow run. It is a dictionary
                with key-value pairs where each value is a STAC item in JSON format.

        Returns:
            The converted output of the workflow run. It is a dictionary
            with key-value pairs where each value is a BaseVibe object that
            represents a geospatial data asset.

        """
        converter = StacConverter()
        return {k: converter.from_stac_item(deserialize_stac(v)) for k, v in output.items()}

    def _convert_task_details(self, details: Dict[str, Any]) -> Dict[str, RunDetails]:
        """Convert the task details of the workflow run to a :class:`RunDetails` dictionary.

        This method takes the task details of the workflow run as a dictionary and converts
        each value to a RunDetails object using keyword arguments. It returns a new dictionary
        with the same keys and converted values. The keys are sorted by their corresponding
        start time or end time if available.

        Args:
            details: The task details of the workflow run.

        Returns:
            The converted task details of the workflow run.

        """
        return {
            k: RunDetails(**v)
            for k, v in sorted(
                details.items(),
                key=lambda x: cast(
                    datetime, parse(x[1][TASK_SORT_KEY]) if x[1][TASK_SORT_KEY] else datetime.now()
                ),
            )
        }

    def _block_until_status(
        self,
        block_until_statuses: List[RunStatus],
        timeout_s: Optional[int] = None,
    ) -> "VibeWorkflowRun":
        """Block until the workflow run has a status that has been specified.

        Also allows defining an optional timeout in seconds.

        Args:
            block_until_statuses: List of :class:`RunStatus` to wait for.
            timeout_s (optional): Timeout in seconds to wait for the workflow to reach one of the
                desired statuses. If not provided, the method will wait indefinitely.

        Returns:
            The workflow run object.

        Raises:
            RuntimeError: If the run does not complete before timeout_s.

        """
        time_start = time.monotonic()
        while self.status not in block_until_statuses:
            time.sleep(self.wait_s)
            if timeout_s is not None and (time.monotonic() - time_start) > timeout_s:
                status_options = " or ".join(block_until_statuses)
                raise RuntimeError(
                    f"Timeout of {timeout_s}s reached while waiting for the workflow to have a "
                    f"status of {status_options}."
                )
        return self

    @property
    def status(self) -> RunStatus:
        """Get the status of the workflow run."""
        if self._status is not RunStatus.deleted:
            self._status = cast(
                RunStatus, RunStatus(self.client.list_runs(self.id)[0]["details.status"])
            )
        return self._status

    @property
    def task_details(self) -> Dict[str, RunDetails]:
        """Get the task details of the workflow run."""
        if self._task_details is not None:
            return self._task_details
        status = self.status
        task_details = self._convert_task_details(
            self.client.list_runs(ids=self.id, fields="task_details")[0]["task_details"]
        )
        if RunStatus.finished(status):
            self._task_details = task_details
        return task_details

    @property
    def task_status(self) -> Dict[str, str]:
        """Get the task status of the workflow run."""
        details = self.task_details
        return {k: v.status for k, v in details.items()}

    @property
    def output(self) -> Optional[BaseVibeDict]:
        """Get the output of the workflow run."""
        if self._output is not None:
            return self._output
        run = self.client.describe_run(self.id)
        self._status = run.details.status
        if self._status != RunStatus.done:
            return None
        self._output = self._convert_output(run.output)
        return self._output

    @property
    def reason(self) -> Optional[str]:
        """Get the reason of the workflow run.

        The reason is a string that describes the status of the workflow run.
        In case of failure, it also contains the reason of the failure.

        """
        status = self.status
        if status == RunStatus.failed:
            run = self.client.list_runs(self.id, "details.reason")[0]
            self._reason = format_double_escaped(run["details.reason"])
        elif status == RunStatus.done:
            self._reason = "Workflow run was successful."
        elif status == RunStatus.cancelled:
            self._reason = f"Workflow run {status}."
        else:
            self._reason = (
                f"Workflow run is {status}. "
                f"Check {self.__class__.__name__}.monitor() for task updates."
            )
        return self._reason

    def cancel(self) -> "VibeWorkflowRun":
        """Cancel the workflow run.

        Returns:
            The workflow run.

        """
        self.client.cancel_run(self.id)
        self.status
        return self

    def delete(self) -> "VibeWorkflowRun":
        """Delete the workflow run.

        Returns:
            The workflow run.

        """
        self.client.delete_run(self.id)
        self.status
        return self

    def resubmit(self) -> "VibeWorkflowRun":
        """Resubmit the current workflow run.

        Returns:
            The resubmitted workflow run instance.

        """
        return self.client.resubmit_run(self.id)

    def block_until_cancelled(
        self,
        timeout_s: Optional[int] = None,
    ) -> "VibeWorkflowRun":
        """Block until the workflow run is cancelled, with an optional timeout in seconds.

        Args:
            timeout_s (optional): Timeout in seconds to wait for the workflow to be cancelled.
                If not provided, the method will wait indefinitely.

        Returns:
            The workflow run object.

        Raises:
            RuntimeError: If the run is not cancelled before timeout_s.

        """
        self._block_until_status([RunStatus.cancelled], timeout_s)
        return self

    def block_until_complete(
        self,
        timeout_s: Optional[int] = None,
    ) -> "VibeWorkflowRun":
        """Block until the workflow run execution completes or fails.

        Also allows defining a timeout in seconds.

        Args:
            timeout_s (optional): Timeout in seconds to wait for the workflow to complete.
                If not provided, the method will wait indefinitely.

        Returns:
            The workflow run object.

        Raises:
            RuntimeError: If the run does not complete before timeout_s.

        """
        self._block_until_status([RunStatus.done, RunStatus.failed], timeout_s)
        return self

    def block_until_deleted(
        self,
        timeout_s: Optional[int] = None,
    ) -> "VibeWorkflowRun":
        """Block until the workflow run is deleted, with an optional timeout in seconds.

        Args:
            timeout_s (optional): Timeout in seconds to wait for the workflow to be deleted.
                If not provided, the method will wait indefinitely.

        Returns:
            The workflow run object.

        Raises:
            RuntimeError: If the run does not complete before timeout_s.

        """
        self._block_until_status([RunStatus.deleted], timeout_s)
        return self

    def monitor(
        self,
        refresh_time_s: int = 1,
        refresh_warnings_time_min: int = 5,
        timeout_min: Optional[int] = None,
        detailed_task_info: bool = False,
    ):
        """Monitor the workflow run.

        This method will call :meth:`vibe_core.client.FarmvibesAiClient.monitor` to monitor
        the workflow run.

        Args:
            refresh_time_s: Refresh interval in seconds (defaults to 1 second).
            refresh_warnings_time_min: Refresh interval in minutes for updating
                the warning messages (defaults to 5 minutes).
            timeout_min: The maximum time to monitor the workflow run, in minutes.
                If not provided, the method will monitor indefinitely.
            detailed_task_info: If True, detailed information about task progress
                will be included in the output (defaults to False).

        """
        self.client.monitor(
            runs=[self],
            refresh_time_s=refresh_time_s,
            refresh_warnings_time_min=refresh_warnings_time_min,
            timeout_min=timeout_min,
            detailed_task_info=detailed_task_info,
        )

    def __repr__(self):
        """Get the string representation of the workflow run.

        Returns:
            The string representation of the workflow run.

        """
        return (
            f"'{self.__class__.__name__}'(id='{self.id}', name='{self.name}',"
            f" workflow='{self.workflow}', status='{self.status}')"
        )


def get_local_service_url() -> str:
    """Retrieve the local service URL used to submit workflow runs to the FarmVibes.AI service.

    This function attempts to read the service URL from a file, and if that fails,
    it will return a fallback URL.

    Returns:
        The local service URL.

    """
    try:
        with open(FARMVIBES_AI_SERVICE_URL_PATH, "r") as fp:
            return fp.read().strip()
    except FileNotFoundError:
        return FALLBACK_SERVICE_URL


def get_remote_service_url() -> str:
    """Get the remote service URL.

    Returns:
        The remote service URL.

    """
    with open(FARMVIBES_AI_REMOTE_SERVICE_URL_PATH, "r") as fp:
        return fp.read().strip()


def get_vibe_client(url: str) -> FarmvibesAiClient:
    """Get a vibe client given an API base URL.

    Args:
        url: The URL.

    Returns:
        The vibe client.

    """
    if not url:
        raise ValueError("URL for vibe client must be provided")
    return FarmvibesAiClient(url)


def get_default_vibe_client(type: Union[str, ClusterType] = "") -> FarmvibesAiClient:
    """Get the default vibe client.

    If given a cluster type, it will try to connect to a cluster of that type assuming the data
    files are present. Otherwise, it will try to connect to a remote cluster first, and if that
    fails, try to connect to the local one (if it exists).

    Args:
        type: The type of the cluster (from :class:`ClusterType`) to connect to.

    Returns:
        The vibe client.

    """
    if not type:
        try:
            return FarmvibesAiClient(get_remote_service_url())
        except Exception:
            return FarmvibesAiClient(get_local_service_url())

    if isinstance(type, str):
        type = ClusterType(type)

    return cast(ClusterType, type).client()
