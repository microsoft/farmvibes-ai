# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data model classes definition used throughout FarmVibes.AI."""

import codecs
import json
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict, field
from datetime import datetime
from enum import auto
from typing import Any, Dict, Final, List, Optional, Tuple, Union
from uuid import UUID

from pydantic.dataclasses import dataclass
from strenum import StrEnum
from typing_extensions import TypedDict

from .data import BaseVibeDict
from .data.core_types import OpIOType
from .data.json_converter import dump_to_json

SUMMARY_DEFAULT_FIELDS: Final[List[str]] = ["id", "workflow", "name", "details.status"]


class MetricsDict(TypedDict):
    """Type definition for metrics dictionary."""

    load_avg: Tuple[float, ...]
    """Average system load.

    The number of processes in the system run queue averaged over the last 1, 5, and 15 minutes
    respectively as a tuple.
    """

    cpu_usage: float
    """The current system-wide CPU utilization as a percentage."""

    free_mem: int
    """The amount of free memory in bytes."""

    used_mem: int
    """The amount of used memory in bytes."""

    total_mem: int
    """The total amount of memory in bytes."""

    disk_free: Optional[int]
    """The amount of free disk space in bytes."""


@dataclass
class Message:
    """Dataclass that represents an API message."""

    message: str
    """The message."""

    id: Optional[str] = None
    """The id of the message."""

    location: Optional[str] = None
    """The location of the message."""


@dataclass
class Region:
    """Dataclass that represents a region."""

    name: str
    """The name of the region."""

    geojson: Dict[str, Any] = field(default_factory=dict)
    """The geojson of the region."""


@dataclass
class SpatioTemporalJson:
    """Dataclass that represents a spatio temporal json."""

    start_date: datetime
    """The start date of the spatio temporal json."""

    end_date: datetime
    """The end date of the spatio temporal json."""

    geojson: Dict[str, Any]
    """The geojson of the spatio temporal json."""


@dataclass
class RunBase:
    """Base dataclass for a run."""

    name: str
    """The name of the run."""
    workflow: Union[str, Dict[str, Any]]
    """The workflow of the run."""
    parameters: Optional[Dict[str, Any]]
    """The parameters of the run."""

    def __post_init__(self):
        if isinstance(self.workflow, str):
            try:
                self.workflow = json.loads(self.workflow)
            except json.decoder.JSONDecodeError:
                pass


@dataclass
class RunConfigInput(RunBase):
    """Dataclass that represents a run config input."""

    user_input: Union[SpatioTemporalJson, Dict[str, Any], List[Any]]
    """The user input of the run config (usually a region/geometry and time range)."""

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.user_input, dict):
            try:
                self.user_input = SpatioTemporalJson(**self.user_input)
            except TypeError:
                # We need this because of BaseVibe.
                pass


class RunStatus(StrEnum):
    """Enum that represents the status of a run."""

    pending = auto()
    """The run is pending"""
    queued = auto()
    """The run is queued."""
    running = auto()
    """The run is running."""
    failed = auto()
    """The run has failed."""
    done = auto()
    """The run is done."""
    cancelled = auto()
    """The run is cancelled."""
    deleting = auto()
    """The run is being deleted."""
    deleted = auto()
    """The run has been deleted."""

    @staticmethod
    def finished(status: "RunStatus"):
        """Check if a run has finished.

        This method checks if a run status is either
        :attr:`vibe_core.datamodel.RunStatus.done`,
        :attr:`vibe_core.datamodel.RunStatus.cancelled`, or
        :attr:`vibe_core.datamodel.RunStatus.failed`.

        Args:
            status: The status to check.

        Returns:
            Whether the run has finished.

        """
        return status in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


@dataclass
class RunDetails:
    """Dataclass that encapsulates the details of a run."""

    start_time: Optional[datetime] = None
    """The start time of the run."""
    submission_time: Optional[datetime] = None
    """The submission time of the run."""
    end_time: Optional[datetime] = None
    """The end time of the run."""
    reason: Optional[str] = None
    """A description of the reason for the status of the run."""
    status: RunStatus = RunStatus.pending  # type: ignore
    """The status of the run."""
    subtasks: Optional[List[Any]] = None
    """Details about the subtasks of the run."""


@dataclass
class RunConfig(RunConfigInput):
    """Dataclass that represents a run config."""

    id: UUID
    """The id of the run config."""
    details: RunDetails
    """The details of the run config."""
    task_details: Dict[str, RunDetails]
    """The details of the tasks of the run config."""
    spatio_temporal_json: Optional[SpatioTemporalJson]
    """The spatio temporal json of the run config."""
    output: str = ""
    """The output of the run."""

    def set_output(self, value: OpIOType):  # pydantic won't let us use a property setter
        """Set the output of the run config.

        Args:
            value: The value to set the output to.
        """
        self.output = encode(dump_to_json(value))

    def __post_init__(self):
        if self.spatio_temporal_json is not None and isinstance(self.spatio_temporal_json, dict):
            try:
                self.spatio_temporal_json = SpatioTemporalJson(**self.spatio_temporal_json)
            except TypeError:
                pass

        super().__post_init__()


@dataclass
class RunConfigUser(RunConfig):
    """Dataclass that represents a run config for the user."""

    output: OpIOType
    """The output of the run."""

    @classmethod
    def from_runconfig(cls, run_config: RunConfig):
        """Create a :class:`RunConfigUser` from a :class:`RunConfig`.

        Args:
            run_config: The run config to create the user run config from.

        Returns:
            The user run config.

        """
        rundict = asdict(run_config)
        output = rundict.pop("output")
        rcu = cls(**rundict)
        rcu.output = json.loads(decode(output)) if output else {}
        return rcu

    @staticmethod
    def finished(status: "RunStatus"):
        """Check if a run has finished.

        This method checks if a given status is either
        :attr:`vibe_core.datamodel.RunStatus.done`,
        :attr:`vibe_core.datamodel.RunStatus.cancelled`, or
        :attr:`vibe_core.datamodel.RunStatus.failed`.

        Args:
            status: The status to check.

        Returns:
            Whether the run has finished.

        """
        return status in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


@dataclass
class TaskDescription:
    """Dataclass that represents a task description."""

    inputs: Dict[str, str] = field(default_factory=dict)
    """The inputs of the task."""
    outputs: Dict[str, str] = field(default_factory=dict)
    """The outputs of the task."""
    parameters: Dict[str, Union[str, Dict[str, str]]] = field(default_factory=dict)
    """The task parameters."""
    task_descriptions: Dict[str, str] = field(default_factory=dict)
    """The descriptions of subtasks."""
    short_description: str = ""
    """The short description of the task."""
    long_description: str = ""
    """The long description of the task."""


class WorkflowRun(ABC):
    """An abstract base class for workflow runs."""

    @property
    @abstractmethod
    def status(self) -> str:
        """Get the status of the workflow run.

        Returns:
            The status of the workflow run as a string.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output(self) -> BaseVibeDict:
        """Get the output of the workflow run.

        Returns:
            The output of the workflow run as a :class:`vibe_core.data.BaseVibeDict`.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


@dataclass
class MonitoredWorkflowRun:
    """Dataclass that represents the monitored workflow run information."""

    workflow: Union[str, Dict[str, Any]]
    """The workflow name or workflow dictionary definition of the run."""

    name: str
    """The name of the run."""

    id: str
    """The id of the run."""

    status: RunStatus
    """The status of the run."""

    task_details: Dict[str, RunDetails]
    """The details of the tasks of the run."""


def encode(data: str) -> str:
    """Encode a string using zlib and base64 encoding.

    This function compresses the data string with zlib and then encodes it into a base64 string.

    Args:
        data: The string to be encoded.

    Returns:
        The encoded string.
    """
    return codecs.encode(zlib.compress(data.encode("utf-8")), "base64").decode("utf-8")  # JSON 😞


def decode(data: str) -> str:
    """Decode the given data using zlib and base64 encodings.

    Args:
        data: The string to decode.

    Returns:
        The decoded string.
    """
    return zlib.decompress(codecs.decode(data.encode("utf-8"), "base64")).decode("utf-8")  # JSON 😞
