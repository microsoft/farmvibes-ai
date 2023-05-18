from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

from dateutil.tz import tzlocal
from dateutil.tz.tz import tzfile
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.progress_bar import ProgressBar

from vibe_core.datamodel import RunDetails, RunStatus, TaskDescription

LEFT_BORDER_PADDING = (0, 0, 0, 4)
CONSOLE_WIDTH = 100

STATUS_STR_MAP = {
    RunStatus.pending: "[yellow]pending[/]",
    RunStatus.running: "[cyan]running[/]",
    RunStatus.failed: "[red]failed[/]",
    RunStatus.done: "[green]done[/]",
    RunStatus.queued: "[yellow]queued[/]",
    RunStatus.cancelled: "[yellow]cancelled[/]",
    RunStatus.cancelling: "[yellow]cancelling[/]",
}

FETCHING_INFO_STR = ":hourglass_not_done: [yellow]Fetching information...[/]"


def strftimedelta(start: datetime, end: datetime) -> str:
    """Returns the time delta between two datetimes as a string in the format 'HH:MM:SS'.

    :param start: Start datetime object.

    :param end: End datetime object.

    :return: The timedelta formatted as a 'HH:MM:SS' string.
    """
    tdelta = end - start
    hours, rem = divmod(int(tdelta.total_seconds()), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_typing(type_dict: Dict[str, str]) -> Dict[str, str]:
    """Formats the types in a type dictionary.

    This function takes a dictionary with type strings and formats it,
    replacing the "typing." prefix in the values with an empty string.

    :param type_dict: The type dictionary to format.

    :return: The formatted dictionary.
    """
    return {k: v.replace("typing.", "") for k, v in type_dict.items()}


class VibeWorkflowDocumenter:
    """Documenter class for :class:`VibeWorkflow` objects.

    This class implements the logic for printing/formatting information about workflows,
    including formatting the text elements and adding styling tags. It contains the methods
    to print out a description of the workflow with its sources, sinks and parameters.

    :param name: The name of the workflow.

    :param sources: Dictionary with source names and types.

    :param sinks: Dictionary with sink names and types.

    :param parameters: Dictionary with parameter names and default values.

    :param description: A :class:`TaskDescription` object containing the short and
        long description of the workflow.
    """

    TITLE_STR = "[bold green]Workflow:[/] [bold underline dodger_blue2]{}[/]"
    DESCRIPTION_SECTION_STR = "\n[bold green]Description:[/]"
    DESCRIPTION_STR = "{short_description} {long_description}"
    ITEM_SECTION_STR = "\n[bold green]{}:[/]"
    ITEM_STR = "- [bold]{name}[/]{additional_info}{description}"

    def __init__(
        self,
        name: str,
        sources: Dict[str, str],
        sinks: Dict[str, str],
        parameters: Dict[str, Any],
        description: TaskDescription,
    ):
        self.wf_name = name
        self.parameters = parameters
        self.description = description

        self.sources = format_typing(sources)
        self.sinks = format_typing(sinks)

        self.console = Console(width=CONSOLE_WIDTH, highlighter=NullHighlighter())
        self.console.clear()

    @property
    def formatted_parameters(self) -> Dict[str, str]:
        """Returns a dictionary of workflow's parameters with their default values.

        :return: A dictionary containing the formatted parameters and default values.
        """
        return {
            param_name: "default: task defined"
            if isinstance(param_value, list)
            else f"default: {param_value}"
            for param_name, param_value in self.parameters.items()
        }

    def _print_header(self):
        self.console.print(self.TITLE_STR.format(self.wf_name))
        self.console.print(self.DESCRIPTION_SECTION_STR)

        desc = escape(
            self.DESCRIPTION_STR.format(
                short_description=self.description.short_description,
                long_description=self.description.long_description,
            )
        )
        desc = Padding(desc, LEFT_BORDER_PADDING)
        self.console.print(desc)

    def _print_sources(self, section_name: str = "Sources"):
        self._print_items_description(self.description.inputs, section_name, self.sources)

    def _print_sinks(self, section_name: str = "Sinks"):
        self._print_items_description(self.description.outputs, section_name, self.sinks)

    def _print_parameters(self, section_name: str = "Parameters"):
        if self.parameters:
            desc = {
                k: str(v) if not isinstance(v, list) else ""
                for k, v in self.description.parameters.items()
            }
            self._print_items_description(desc, section_name, self.formatted_parameters)

    def _print_tasks(self, section_name: str = "Tasks"):
        task_dict = {task_name: "" for task_name in self.description.task_descriptions.keys()}
        self._print_items_description(self.description.task_descriptions, section_name, task_dict)

    def _print_items_description(
        self,
        description_dict: Dict[str, str],
        section_name: str,
        additional_info: Dict[str, str] = {},
    ):
        self.console.print(self.ITEM_SECTION_STR.format(section_name))

        for item_name, item_info in additional_info.items():
            item_description = description_dict.get(item_name, "")
            item_description = f": {item_description}" if item_description else ""

            item_info = f" ([blue]{escape(item_info)}[/])" if item_info else ""

            item_doc = self.ITEM_STR.format(
                name=item_name, additional_info=item_info, description=escape(item_description)
            ).strip(":")
            item_doc = Padding(item_doc, LEFT_BORDER_PADDING)
            self.console.print(item_doc)

    def print_documentation(self):
        """Prints the full documentation of the workflow.

        This method prints the header of the documentation, the sources, the sinks,
        the parameters and the tasks provided in the parsed workflow yaml file.

        """
        self._print_header()
        self._print_sources()
        self._print_sinks()
        self._print_parameters()
        self._print_tasks()


class VibeWorkflowRunMonitor:
    """Class that abstracts the formatting of workflow run status

    :param api_time_zone: The time zone of the API server.

    :param detailed_task_info: If True, detailed information about task progress will be
        included in the output (defaults to False).
    """

    TITLE_STR = (
        "[not italic]:earth_americas: "
        "FarmVibes.AI :earth_africa: "
        "[dodger_blue3]{}[/] :earth_asia: \n"
        "Run name: [dodger_blue3]{}[/]\n"
        "Run id: [dark_green]{}[/]\n"
        "Run status: {}\n"
        "Run duration: [dodger_blue3]{}[/][/]"
    )
    WARNING_HEADER_STR = "\n[yellow]:warning:  Warnings :warning:[/]"
    WARNING_STR = "\n{}\n[yellow]:warning:  :warning:  :warning:[/]"
    TABLE_FIELDS = [
        "Task Name",
        "Status",
        "Start Time",
        "End Time",
        "Duration",
    ]
    SIMPLE_COMLUMN_NAME = "Progress"
    DETAILED_COLUMN_NAME = "Subtasks\n([green]D[/]/[blue]R[/]/[yellow]Q[/]/[yellow]P[/]/[red]F[/])"
    TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
    TIME_FORMAT_WITH_TZ = "%Y/%m/%d %H:%M:%S %Z"
    PBAR_WIDTH = 20

    def __init__(self, api_time_zone: tzfile, detailed_task_info: bool = False):
        self.api_tz = api_time_zone
        self.detailed_task_info = detailed_task_info
        self.column_names = self.TABLE_FIELDS + [
            self.DETAILED_COLUMN_NAME if self.detailed_task_info else self.SIMPLE_COMLUMN_NAME
        ]
        self.client_tz = tzlocal()
        self._populate_table()

        console = Console()
        console.clear()
        self.live_context = Live(self.table, console=console, screen=False, auto_refresh=False)

    def _get_time_str(self, time: Optional[datetime]) -> str:
        if time is None:
            return "N/A".center(len(self.TIME_FORMAT), " ")
        return (
            time.replace(tzinfo=self.api_tz)
            .astimezone(tz=self.client_tz)
            .strftime(self.TIME_FORMAT)
        )

    def _render_subtask_info(self, task_info: RunDetails) -> Union[Table, str]:
        if task_info.subtasks is None:
            return "-"
        counts = Counter([RunStatus(r["status"]) for r in task_info.subtasks])
        if self.detailed_task_info:
            # Let's just print out informative text
            return (
                f"[green]{counts[RunStatus.done]}[/]/[blue]{counts[RunStatus.running]}[/]/"
                f"[yellow]{counts[RunStatus.queued]}[/]/"
                f"[yellow]{counts[RunStatus.pending]}[/]/[red]{counts[RunStatus.failed]}[/]"
            )
        # Let's render a nice looking progress bar
        total = sum(counts.values())
        subtasks = Table(
            "bar",
            "text",
            show_edge=False,
            show_footer=False,
            show_header=False,
            show_lines=False,
            box=None,  # Remove line between columns
        )
        subtasks.add_row(
            ProgressBar(total=total, completed=counts[RunStatus.done], width=self.PBAR_WIDTH),
            f"{counts[RunStatus.done]}/{total}",
        )
        return subtasks

    def _add_row(self, task_name: str, task_info: RunDetails):
        start_time_str = self._get_time_str(task_info.start_time)
        end_time_str = self._get_time_str(task_info.end_time)
        duration = strftimedelta(
            self.time_or_now(task_info.start_time), self.time_or_now(task_info.end_time)
        )

        subtasks = self._render_subtask_info(task_info)

        self.table.add_row(
            task_name,
            STATUS_STR_MAP[task_info.status],
            start_time_str,
            end_time_str,
            duration,
            subtasks,
        )

    def _init_table(self, monitored_warnings: List[Union[str, Warning]] = []):
        """Creates a new table and populate with wf-agnostic info"""
        current_time_caption = warnings_caption = ""

        self.table = Table(show_footer=False)
        for col_name in self.column_names:
            self.table.add_column(col_name)

        # Build current time caption
        current_time_caption = (
            f"Last update: {datetime.now(tz=self.client_tz).strftime(self.TIME_FORMAT_WITH_TZ)}"
        )

        # Build monitored warnings caption
        if monitored_warnings:
            warnings_caption = "".join(
                [self.WARNING_HEADER_STR] + [self.WARNING_STR.format(w) for w in monitored_warnings]
            )

        self.table.caption = current_time_caption + warnings_caption

    def time_or_now(self, time: Optional[datetime]) -> datetime:
        """
        Converts a given datetime object to the client's timezone.
        If no datetime object is provided, the current time is used.

        :param time: Datetime object to convert to the client's timezone.

        :return: The datetime object converted to the client's timezone.
        """
        return (
            time.replace(tzinfo=self.api_tz).astimezone(tz=self.client_tz)
            if time is not None
            else datetime.now(tz=self.client_tz)
        )

    def _get_run_duration(
        self, sorted_tasks: List[Tuple[str, RunDetails]], run_status: RunStatus
    ) -> str:
        run_duration: str = ":hourglass_not_done:"

        if sorted_tasks:
            # Get the start time from the first submitted task
            run_start_time = self.time_or_now(sorted_tasks[-1][1].submission_time)

            # Get the end time of the last task (if finished) or current time otherwise
            run_end_time = (
                self.time_or_now(sorted_tasks[0][1].end_time)
                if RunStatus.finished(run_status)
                else datetime.now(tz=self.client_tz)
            )
            run_duration = strftimedelta(start=run_start_time, end=run_end_time)

        return run_duration

    def _populate_table(
        self,
        wf_name: Union[str, Dict[str, Any]] = ":hourglass_not_done:",
        run_name: str = ":hourglass_not_done:",
        run_id: str = ":hourglass_not_done:",
        run_status: RunStatus = RunStatus.pending,
        wf_tasks: Optional[Dict[str, RunDetails]] = None,
        monitored_warnings: List[Union[str, Warning]] = [],
    ):
        """Method that creates a new table with updated task info"""
        run_duration: str = ":hourglass_not_done:"

        # Create new table
        self._init_table(monitored_warnings)

        # Populate Rows
        if wf_tasks is None:
            self.table.add_row(FETCHING_INFO_STR)
        else:
            # Sort tasks by reversed submission/start/end time (running tasks will be on top)
            sorted_tasks = sorted(
                wf_tasks.items(),
                key=lambda t: (
                    self.time_or_now(t[1].submission_time),
                    self.time_or_now(t[1].start_time),
                    self.time_or_now(t[1].end_time),
                ),
                reverse=True,
            )

            # Add each task to the table
            for task_name, task_info in sorted_tasks:
                self._add_row(task_name, task_info)

            # Compute run duration
            run_duration = self._get_run_duration(sorted_tasks, run_status)

        # Populate Header
        # Do not print the whole dict definition if it is a custom workflow
        wf_name = f"Custom: '{wf_name['name']}'" if isinstance(wf_name, dict) else wf_name
        self.table.title = self.TITLE_STR.format(
            wf_name, run_name, run_id, STATUS_STR_MAP[run_status], run_duration
        )

    def update_run_status(
        self,
        wf_name: Union[str, Dict[str, Any]],
        run_name: str,
        run_id: str,
        run_status: RunStatus,
        wf_tasks: Dict[str, RunDetails],
        monitored_warnings: List[Union[str, Warning]],
    ):
        """Updates the monitor table.

        This method will update the monitor table with the latest information about the workflow
        run, individual task status and monitored warnings.

        :param wf_name: Name of the workflow being executed.
            It can be a string or a custom workflow definition (as a dict).

        :param run_name: Name of the workflow run.

        :param run_id: Id of the workflow run.

        :param run_status: Status of the run.

        :param wf_tasks: Dictionary containing the details of each task in the workflow.

        :param monitored_warnings: List of monitored warnings.
        """
        self._populate_table(wf_name, run_name, run_id, run_status, wf_tasks, monitored_warnings)
        self.live_context.update(self.table, refresh=True)
