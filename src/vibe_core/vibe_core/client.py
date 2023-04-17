import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
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

from vibe_core.data import BaseVibeDict, StacConverter
from vibe_core.data.core_types import BaseVibe
from vibe_core.data.json_converter import dump_to_json
from vibe_core.data.utils import deserialize_stac, serialize_input
from vibe_core.datamodel import (
    RunConfigInput,
    RunConfigUser,
    RunDetails,
    RunStatus,
    SpatioTemporalJson,
    TaskDescription,
)
from vibe_core.monitor import VibeWorkflowDocumenter, VibeWorkflowRunMonitor
from vibe_core.utils import ensure_list, format_double_escaped

FALLBACK_SERVICE_URL = "http://192.168.49.2:30000/"
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


class WorkflowRun(ABC):
    """An abstract base class for workflow runs."""

    @property
    @abstractmethod
    def status(self) -> str:
        """Gets the status of the workflow run.

        :return: The status of the workflow run as a string.

        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output(self) -> BaseVibeDict:
        """Gets the output of the workflow run.

        :return: The output of the workflow run as a :class:`vibe_core.data.BaseVibeDict`.

        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class Client(ABC):
    """An abstract base class for clients."""

    @abstractmethod
    def list_workflows(self) -> List[str]:
        """Lists all available workflows.

        :return: A list of workflow names.

        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        workflow: str,
        geometry: BaseGeometry,
        time_range: Tuple[datetime, datetime],
    ) -> WorkflowRun:
        """Runs a workflow.

        :param workflow: The name of the workflow to run.
        :param geometry: The geometry to run the workflow on.
        :param time_range: The time range to run the workflow on.

        :return: A :class:`WorkflowRun` object.

        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class FarmvibesAiClient(Client):
    """A client for the FarmVibes.AI service.

    :param baseurl: The base URL of the FarmVibes.AI service.
    """

    default_headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    """The default headers to use for requests to the FarmVibes.AI service."""

    def __init__(self, baseurl: str):
        self.baseurl = baseurl
        self.session = requests.Session()
        self.session.headers.update(self.default_headers)

    def _request(self, method: str, endpoint: str, *args: Any, **kwargs: Any):
        """Sends a request to the FarmVibes.AI service and handle errors.

        :param method: The HTTP method to use (e.g., 'GET' or 'POST').

        :param endpoint: The endpoint to request.

        :param args: Positional arguments to pass to :meth:`requests.Session.request`.

        :param kwargs: Keyword arguments to pass to :meth:`requests.Session.request`.

        :return: The response from the FarmVibes.AI service.
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
        """Forms a payload dictionary for submitting a workflow run.

        :param workflow: The name of the workflow to run or a dict containing
            the workflow definition.

        :param parameters: A dict of optional parameters to pass to the workflow.
            The keys and values depend on the specific workflow definition.

        :param geometry: The geometry to use for the input data.
            It must be a valid shapely geometry object (e.g., Point or Polygon).
            It will be converted to GeoJSON format internally.
            Alternatively it can be None if input_data is provided instead.
        .. note::
            Either `geometry` and `time_range` or `input_data` must be provided.
        .. warning::
            Providing both `geometry` and `input_data` will result in an error.

        :param time_range: The time range to use for the input data. It must be
            a tuple of two datetime objects representing the start and end dates.
            Alternatively it can be None if input_data is provided instead.
        .. note::
            Either `geometry` and `time_range` or `input_data` must be provided.
        .. warning::
            Providing both `time_range` and `input_data` will result in an error.

        :param input_data: The input data to use for the workflow run.
            It must be an instance of InputData or one of its subclasses
            (e.g., SpatioTemporalJson or SpatioTemporalRaster). Alternatively
            it can be None if geometry and time_range are provided instead.
        .. note::
            Either `geometry` and `time_range` or `input_data` must be provided.
        .. warning::
            Providing both `input_data` and either `geometry` or `time_range`
        will result in an error.

        :param run_name: The name to assign to the workflow run.

        :return: A dict containing the payload for submitting a workflow run.
            The keys are 'run_name', 'workflow', 'parameters', and 'user_input'.
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
        """Verifies that there is enough disk space available for the cache.

        This method checks the system metrics returned by the FarmVibes.AI service
        and compares the disk free space with a predefined threshold. If the disk
        free space is below the threshold, a warning message is displayed to the user,
        suggesting to clear the cache.

        .. note::
            The disk free space threshold is defined by :const:`DISK_FREE_THRESHOLD_BYTES`.

        :raises: :exc:`RuntimeWarning` if the disk space is low.
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
        """Lists all available workflows on the FarmVibes.AI service.

        This method returns a list of workflow names that can be used to
        submit workflow runs or to get more details about a specific workflow.

        :return: A list of workflow names.
        """
        return self._request("GET", "v0/workflows")

    def describe_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Describes a workflow.

        This method returns a dictionary containing the description of a
        workflow, such as its inputs, outputs, parameters and tasks.

        .. note::
            The description is returned as a :class:`TaskDescription` object,
            which is a dataclass that represents the structure and
            properties of a workflow.

        :param workflow_name: The name of the workflow to describe.
            It must be one of the names returned by list_workflows().

        :return: A dictionary containing the workflow description.
            The keys are 'name', 'description', 'inputs', 'outputs' and 'parameters'.
        """
        desc = self._request("GET", f"v0/workflows/{workflow_name}?return_format=description")
        desc["description"] = TaskDescription(**desc["description"])
        return desc

    def get_system_metrics(self) -> Dict[str, Union[int, float, Tuple[float, ...]]]:
        """Gets system metrics from the FarmVibes.AI service.

        This method returns a dictionary containing various system metrics,
        such as CPU usage, memory usage and disk space.

        :return: A dictionary containing system metrics.
        """
        return self._request("GET", "v0/system-metrics")

    def get_workflow_yaml(self, workflow_name: str) -> str:
        """Gets the YAML definition of a workflow.

        This method returns a string containing the YAML definition of a
        workflow. The YAML definition specifies the name and operations of
        the workflow in a human-readable format.

        :param workflow_name: The name of the workflow. It must be one
            of the names returned by list_workflows().

        :return: The YAML definition of the workflow.
        """
        yaml_content = self._request("GET", f"v0/workflows/{workflow_name}?return_format=yaml")
        return yaml.dump(yaml_content, default_flow_style=False, default_style="", sort_keys=False)

    def cancel_run(self, run_id: str) -> str:
        """Cancels a workflow run.

        This method sends a request to the FarmVibes.AI service to cancel
        a workflow run that is in progress or pending. If the cancellation
        is successful, the workflow run status will be set to 'cancelled'.

        .. note::
            The cancellation may take some time to take effect depending on
            the state of the workflow run and the service availability.

        .. warning::
            A workflow run that is already completed or failed cannot be cancelled.

        :param run_id: The ID of the workflow run to cancel.

        :return: The message from the FarmVibes.AI service indicating whether
            the cancellation was successful or not.
        """
        return self._request("POST", f"v0/runs/{run_id}/cancel")["message"]

    def describe_run(self, run_id: str) -> RunConfigUser:
        """Describes a workflow run.

        This method returns a RunConfigUser object containing the description of a
        workflow run, such as its name, status, inputs and outputs.

        :param run_id: The ID of the workflow run to describe.

        :return: A :class:`RunConfigUser` object containing the workflow run description.
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
        """Prints the documentation of a workflow.

        This method prints a formatted documentation of a workflow,
        including its name, description, inputs, outputs and parameters.

        .. note::
            The documentation is printed to stdout and can be redirected to
            other outputs if needed.

        :param workflow_name: The name of the workflow to document.
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
        """Lists workflow runs on the FarmVibes.AI service.

        :param ids: The IDs of the workflow runs to list.
            If None, all workflow runs will be listed.

        :param fields: The fields to return for each workflow run.
            If None, all fields will be returned.

        :return: A list of workflow runs. Each run is represented by a dictionary
            with keys corresponding to the requested fields and values containing
            the field values.
        """

        ids = [f"ids={id}" for id in ensure_list(ids)] if ids is not None else []
        fields = [f"fields={field}" for field in ensure_list(fields)] if fields is not None else []
        query_str = "&".join(ids + fields)

        return self._request("GET", f"v0/runs?{query_str}")

    def get_run_by_id(self, id: str) -> "VibeWorkflowRun":
        """Gets a workflow run by ID.

        This method returns a :class:`VibeWorkflowRun` object containing
        the details of a workflow run by its ID.

        :param id: The ID of the workflow run to get.

        :return: A :class:`VibeWorkflowRun` object.
        """

        fields = ["id", "name", "workflow", "parameters"]
        run = self.list_runs(id, fields=fields)[0]
        return VibeWorkflowRun(*(run[f] for f in fields), self)  # type: ignore

    def get_api_time_zone(self) -> tzfile:
        """Gets the time zone of the FarmVibes.AI REST-API.

        This method returns a tzfile object representing the time zone of
        the FarmVibes.AI REST-API. The time zone is determined by parsing
        the 'date' header from the response of a GET request to the base URL
        of the service. If the 'date' header is missing or invalid, a warning
        is issued and the client time zone is used instead.

        .. note::
            The tzfile object is a subclass of datetime.tzinfo that represents
            a time zone using an Olson database file.

        :return: The time zone of the FarmVibes.AI REST-API as a tzfile object.
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
    ) -> "VibeWorkflowRun":
        ...

    @overload
    def run(
        self,
        workflow: Union[str, Dict[str, Any]],
        name: str,
        *,
        input_data: InputData[T],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "VibeWorkflowRun":
        ...

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
        """Runs a workflow.

        This method instantiates a workflow run using the provided data and parameters.

        :param workflow: The name of the workflow to run or a dict containing
                the workflow definition.

        :param name: The name to assign to the workflow run.

        :param geometry: The geometry to use for the input data.
            It must be a valid shapely geometry object (e.g., Point or Polygon).
            It will be converted to GeoJSON format internally.
            Alternatively it can be None if input_data is provided instead.
        .. note::
            Either `geometry` and `time_range` or `input_data` must be provided.
        .. warning::
            Providing both `geometry` and `input_data` will result in an error.

        :param time_range: The time range to use for the input data. It must be
            a tuple of two datetime objects representing the start and end dates.
            Alternatively it can be None if input_data is provided instead.
        .. note::
            Either `geometry` and `time_range` or `input_data` must be provided.
        .. warning::
            Providing both `time_range` and `input_data` will result in an error.

        :param input_data: The input data to use for the workflow run.
            It must be an instance of InputData or one of its subclasses
            (e.g., SpatioTemporalJson or SpatioTemporalRaster). Alternatively
            it can be None if geometry and time_range are provided instead.
        .. note::
            Either `geometry` and `time_range` or `input_data` must be provided.
        .. warning::
            Providing both `input_data` and either `geometry` or `time_range`
            will result in an error.

        :param parameters: A dict of optional parameters to pass to the workflow.
            The keys and values depend on the specific workflow definition.

        :return: A :class:`VibeWorkflowRun` object.
        """
        self.verify_disk_space()
        payload = dump_to_json(
            self._form_payload(workflow, parameters, geometry, time_range, input_data, name),
        )
        response = self._request("POST", "v0/runs", data=payload)
        return self.get_run_by_id(response["id"])

    def resubmit_run(self, run_id: str) -> "VibeWorkflowRun":
        """
        Resubmits a workflow run with the given run ID.

        :param run_id: The ID of the workflow run to resubmit.

        :return: The resubmitted workflow run.
        """

        self.verify_disk_space()
        response = self._request("POST", f"v0/runs/{run_id}/resubmit")
        return self.get_run_by_id(response["id"])


class VibeWorkflowRun(WorkflowRun):
    """Represents a workflow run in FarmVibes.AI.

    :param id: The ID of the workflow run.

    :param name: The name of the workflow run.

    :param workflow: The name of the workflow associated to the run.

    :param parameters: The parameters associated to the workflow run, as a dict.
        The keys and values depend on the specific workflow definition.

    :param client: An instance of the :class:`FarmVibesAiClient` class.
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
        """Converts the output of the workflow run to a :class:`BaseVibeDict`.

        This method takes the output of the workflow run as a dictionary and
        converts each value to a DataVibe object using a StacConverter object.
        It returns a new dictionary with the same keys and converted values.

        :param output: The output of the workflow run. It is a dictionary
            with key-value pairs where each value is a STAC item in JSON format.

        :return: The converted output of the workflow run. It is a dictionary
            with key-value pairs where each value is a BaseVibe object that
            represents a geospatial data asset.
        """
        converter = StacConverter()
        return {k: converter.from_stac_item(deserialize_stac(v)) for k, v in output.items()}

    def _convert_task_details(self, details: Dict[str, Any]) -> Dict[str, RunDetails]:
        """Converts the task details of the workflow run to a :class:`RunDetails` dictionary.

        This method takes the task details of the workflow run as a dictionary and converts
        each value to a RunDetails object using keyword arguments. It returns a new dictionary
        with the same keys and converted values. The keys are sorted by their corresponding
        start time or end time if available.

        :param details: The task details of the workflow run.

        :return: The converted task details of the workflow run.
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

    @property
    def status(self) -> RunStatus:
        """Gets the status of the workflow run."""
        if not RunStatus.finished(self._status):
            self._status = RunStatus(self.client.list_runs(self.id)[0]["details.status"])
        return self._status

    @property
    def task_details(self) -> Dict[str, RunDetails]:
        """Gets the task details of the workflow run."""

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
        """Gets the task status of the workflow run."""

        details = self.task_details
        return {k: v.status for k, v in details.items()}

    @property
    def output(self) -> Optional[BaseVibeDict]:
        """Gets the output of the workflow run."""

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
        """Gets the reason of the workflow run.

        The reason is a string that describes the status of the workflow run.
        In case of failure, it also contains the reason of the failure.

        """
        status = self.status
        if status == RunStatus.done:
            self._reason = "Workflow run was successful."
        elif status in [RunStatus.cancelled, RunStatus.cancelling]:
            self._reason = f"Workflow run {status}."
        elif status in [RunStatus.running, RunStatus.pending]:
            self._reason = (
                f"Workflow run is {status}. "
                f"Check {self.__class__.__name__}.monitor() for task updates."
            )
        else:  # RunStatus.failed
            run = self.client.list_runs(self.id, "details.reason")[0]
            self._reason = format_double_escaped(run["details.reason"])
        return self._reason

    def cancel(self) -> "VibeWorkflowRun":
        """Cancels the workflow run.

        :return: The workflow run.
        """
        self.client.cancel_run(self.id)
        self.status
        return self

    def resubmit(self) -> "VibeWorkflowRun":
        """
        Resubmits the current workflow run.

        :return: The resubmitted workflow run instance.
        """
        return self.client.resubmit_run(self.id)

    def block_until_complete(self, timeout_s: Optional[int] = None) -> "VibeWorkflowRun":
        """Blocks until the workflow run execution completes or fails, with an optional
        timeout in seconds.

        :param timeout_s:  Optional timeout in seconds to wait for the workflow to complete.
            If not provided, the method will wait indefinitely.

        :raises RuntimeError: If the run does not complete before timeout_s.

        :return: The workflow run object.
        """
        time_start = time.time()
        while self.status not in (RunStatus.done, RunStatus.failed):
            time.sleep(self.wait_s)
            if timeout_s is not None and (time.time() - time_start) > timeout_s:
                raise RuntimeError(
                    f"Timeout of {timeout_s}s reached while waiting for workflow completion"
                )
        return self

    def monitor(
        self,
        refresh_time_s: int = 1,
        refresh_warnings_time_min: int = 5,
        timeout_min: Optional[int] = None,
        detailed_task_info: bool = False,
    ):
        """Monitors the workflow run.

        This method will block and print the status of the run each refresh_time_s seconds,
        until the workflow run finishes or it reaches timeout_min minutes. It will also
        print warnings every refresh_warnings_time_min minutes.

        :param refresh_time_s: Refresh interval in seconds (defaults to 1 second).

        :param refresh_warnings_time_min: Refresh interval in minutes for updating
            the warning messages (defaults to 5 minutes).

        :param timeout_min: The maximum time to monitor the workflow run, in minutes.
            If not provided, the method will monitor indefinitely.

        :param detailed_task_info: If True, detailed information about task progress
            will be included in the output (defaults to False).
        """
        with warnings.catch_warnings(record=True) as monitored_warnings:
            monitor = VibeWorkflowRunMonitor(
                api_time_zone=self.client.get_api_time_zone(),
                detailed_task_info=detailed_task_info,
            )

            stop_monitoring = False
            time_start = last_warning_refresh = time.time()

            with monitor.live_context:
                while not stop_monitoring:
                    monitor.update_run_status(
                        self.workflow,
                        self.name,
                        self.id,
                        self.status,
                        self.task_details,
                        [w.message for w in monitored_warnings],
                    )

                    time.sleep(refresh_time_s)
                    curent_time = time.time()

                    # Check for warnings every refresh_warnings_time_min minutes
                    if (curent_time - last_warning_refresh) / 60.0 > refresh_warnings_time_min:
                        self.client.verify_disk_space()
                        last_warning_refresh = curent_time

                    # Check for timeout
                    did_timeout = (
                        timeout_min is not None and (curent_time - time_start) / 60.0 > timeout_min
                    )
                    stop_monitoring = RunStatus.finished(self.status) or did_timeout

                # Update one last time to make sure we have the latest state
                monitor.update_run_status(
                    self.workflow,
                    self.name,
                    self.id,
                    self.status,
                    self.task_details,
                    [w.message for w in monitored_warnings],
                )

    def __repr__(self):
        """Gets the string representation of the workflow run.

        :return: The string representation of the workflow run.
        """
        return (
            f"'{self.__class__.__name__}'(id='{self.id}', name='{self.name}',"
            f" workflow='{self.workflow}', status='{self.status}')"
        )


def get_local_service_url() -> str:
    """Retrieves the local service URL used to submit workflow runs to the FarmVibes.AI service.

    This function attempts to read the service URL from a file, and if that fails,
    it will return a fallback URL.

    :return: The local service URL.
    """
    try:
        with open(FARMVIBES_AI_SERVICE_URL_PATH, "r") as fp:
            return fp.read().strip()
    except FileNotFoundError:
        return FALLBACK_SERVICE_URL


def get_remote_service_url() -> str:
    """Gets the remote service URL.

    :return: The remote service URL.
    """
    try:
        with open(FARMVIBES_AI_REMOTE_SERVICE_URL_PATH, "r") as fp:
            return fp.read().strip()
    except FileNotFoundError as e:
        print(e)
        raise


def get_default_vibe_client(url: str = "", connect_remote: bool = False):
    """Gets the default vibe client.

    :param url: The URL.
    :param connect_remote: Whether to connect remotely.

    :return: The vibe client.
    """
    if not url:
        url = get_remote_service_url() if connect_remote else get_local_service_url()

    return FarmvibesAiClient(url)
