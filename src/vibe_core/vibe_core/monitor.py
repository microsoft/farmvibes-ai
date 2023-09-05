from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from dateutil.tz import tzlocal
from dateutil.tz.tz import tzfile
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.progress_bar import ProgressBar
from rich.table import Table

from vibe_core.datamodel import MonitoredWorkflowRun, RunDetails, RunStatus, TaskDescription

LEFT_BORDER_PADDING = (0, 0, 0, 4)
CONSOLE_WIDTH = 100

STATUS_STR_MAP = {
    RunStatus.pending: "[yellow]pending[/]",
    RunStatus.running: "[cyan]running[/]",
    RunStatus.failed: "[red]failed[/]",
    RunStatus.done: "[green]done[/]",
    RunStatus.queued: "[yellow]queued[/]",
    RunStatus.cancelled: "[yellow]cancelled[/]",
}

FETCHING_ICON_STR = ":hourglass_not_done:"
FETCHING_INFO_STR = f"{FETCHING_ICON_STR} [yellow]Fetching information...[/]"


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
    """Class that abstracts the formatting of workflow run status.

    :param api_time_zone: The time zone of the API server.

    :param detailed_task_info: If True, detailed information about task progress will be
        included in the output (defaults to False).
    """

    SINGLE_RUN_TITLE_STR = (
        "[not italic]:earth_americas: "
        "FarmVibes.AI :earth_africa: "
        "[dodger_blue3]{}[/] :earth_asia: \n"
        "Run name: [dodger_blue3]{}[/]\n"
        "Run id: [dark_green]{}[/]\n"
        "Run status: {}\n"
        "Run duration: [dodger_blue3]{}[/][/]"
    )

    MULTI_RUN_TITLE_STR = (
        "[not italic]:earth_americas: "
        "FarmVibes.AI :earth_africa: "
        "Multi-Run Monitoring :earth_asia: \n"
        "Total duration: [dodger_blue3]{}[/][/]"
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

    def __init__(
        self, api_time_zone: tzfile, detailed_task_info: bool = False, multi_run: bool = False
    ):
        self.api_tz = api_time_zone
        self.detailed_task_info = detailed_task_info
        self.client_tz = tzlocal()
        self.multi_run = multi_run

        self._init_table()

    def _init_table(self):
        prefix_column = ["Run Name"] if self.multi_run else []
        self.column_names = (
            prefix_column
            + self.TABLE_FIELDS
            + [self.DETAILED_COLUMN_NAME if self.detailed_task_info else self.SIMPLE_COMLUMN_NAME]
        )

        if self.multi_run:
            self._populate_multi_run_table([])
        else:
            self._populate_single_run_table()

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

    def _render_progress(
        self, task_info: Union[List[Tuple[str, RunDetails]], RunDetails]
    ) -> Union[Table, str]:
        if isinstance(task_info, RunDetails):
            if task_info.subtasks is None:
                return ""
            counts = Counter([RunStatus(r["status"]) for r in task_info.subtasks])
        else:
            if not task_info:
                return ""
            counts = Counter([r[1].status for r in task_info])
        if self.detailed_task_info:
            # Let's just print out informative text
            return (
                f"[green]{counts[RunStatus.done]}[/]/[blue]{counts[RunStatus.running]}[/]/"
                f"[yellow]{counts[RunStatus.queued]}[/]/"
                f"[yellow]{counts[RunStatus.pending]}[/]/[red]{counts[RunStatus.failed]}[/]"
            )
        # Let's render a nice looking progress bar
        total = sum(counts.values())
        progress_table = Table(
            "bar",
            "text",
            show_edge=False,
            show_footer=False,
            show_header=False,
            show_lines=False,
            box=None,  # Remove line between columns
        )
        progress_table.add_row(
            ProgressBar(total=total, completed=counts[RunStatus.done], width=self.PBAR_WIDTH),
            f"{counts[RunStatus.done]}/{total}",
        )
        return progress_table

    def _add_task_row(self, task_name: str, task_info: RunDetails):
        start_time_str = self._get_time_str(task_info.start_time)
        end_time_str = self._get_time_str(task_info.end_time)
        duration = strftimedelta(
            self.time_or_now(task_info.start_time), self.time_or_now(task_info.end_time)
        )

        subtasks = self._render_progress(task_info)

        if self.multi_run:
            self.table.add_row(
                ":left_arrow_curving_right:",
                task_name,
                STATUS_STR_MAP[task_info.status],
                start_time_str,
                end_time_str,
                duration,
                subtasks,
            )
        else:
            self.table.add_row(
                task_name,
                STATUS_STR_MAP[task_info.status],
                start_time_str,
                end_time_str,
                duration,
                subtasks,
            )

    def _add_workflow_row(
        self, run: MonitoredWorkflowRun, sorted_tasks: List[Tuple[str, RunDetails]]
    ):
        start_time_str = self._get_time_str(sorted_tasks[-1][1].submission_time)
        end_time_str = self._get_time_str(sorted_tasks[0][1].end_time)

        run_progress = self._render_progress(sorted_tasks)

        # Compute run duration
        run_duration = self._get_run_duration(sorted_tasks, run.status)

        # TODO: Add missing fields
        self.table.add_row(
            run.name,
            "",
            STATUS_STR_MAP[run.status],
            start_time_str,
            end_time_str,
            run_duration,
            run_progress,
        )

    def _build_clean_table(self, monitored_warnings: List[Union[str, Warning]] = []):
        """Creates a new table and populate with wf-agnostic info."""
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
        run_duration: str = FETCHING_ICON_STR

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

    def _populate_single_run_table(
        self,
        run: Optional[MonitoredWorkflowRun] = None,
        monitored_warnings: List[Union[str, Warning]] = [],
    ):
        """Method that creates a new table with updated task info for a single run."""
        run_duration: str = FETCHING_ICON_STR

        # Create new table
        self._build_clean_table(monitored_warnings)

        if run:
            # Populate Rows
            if run.task_details is None:
                self.table.add_row(FETCHING_INFO_STR)
            else:
                # Sort tasks by reversed submission/start/end time (running tasks will be on top)
                sorted_tasks = sorted(
                    run.task_details.items(),
                    key=lambda t: (
                        self.time_or_now(t[1].submission_time),
                        self.time_or_now(t[1].start_time),
                        self.time_or_now(t[1].end_time),
                    ),
                    reverse=True,
                )

                # Add each task to the table
                for task_name, task_info in sorted_tasks:
                    self._add_task_row(task_name, task_info)

                # Compute run duration
                run_duration = self._get_run_duration(sorted_tasks, run.status)

            # Populate Header
            # Do not print the whole dict definition if it is a custom workflow
            wf_name = (
                f"Custom: '{run.workflow['name']}'"
                if isinstance(run.workflow, dict)
                else run.workflow
            )
            self.table.title = self.SINGLE_RUN_TITLE_STR.format(
                wf_name, run.name, run.id, STATUS_STR_MAP[run.status], run_duration
            )
        else:
            self.table.title = self.SINGLE_RUN_TITLE_STR.format(
                FETCHING_ICON_STR,
                FETCHING_ICON_STR,
                FETCHING_ICON_STR,
                FETCHING_ICON_STR,
                run_duration,
            )

    def _populate_multi_run_table(
        self,
        runs: List[MonitoredWorkflowRun],
        monitored_warnings: List[Union[str, Warning]] = [],
    ):
        """Method that creates a new table with updated task info for multiple runs."""
        total_duration: str = FETCHING_ICON_STR
        multi_run_task_list = []

        # Create new table
        self._build_clean_table(monitored_warnings)

        # Populate Rows
        for run in runs:
            # Sort tasks by reversed submission/start/end time
            sorted_tasks = sorted(
                run.task_details.items(),
                key=lambda t: (
                    self.time_or_now(t[1].submission_time),
                    self.time_or_now(t[1].start_time),
                    self.time_or_now(t[1].end_time),
                ),
                reverse=True,
            )

            # Add first and last task to the list
            multi_run_task_list += sorted_tasks[:1] + sorted_tasks[-1:]

            # Add workflow run info to the table
            self._add_workflow_row(run, sorted_tasks)

            # Displaying all running/queued tasks, or the last finished task if they all completed
            active_tasks = [
                t for t in sorted_tasks if t[1].status in (RunStatus.running, RunStatus.queued)
            ]
            finished_tasks = [t for t in sorted_tasks if RunStatus.finished(t[1].status)]
            displaying_tasks = active_tasks if active_tasks else finished_tasks[:1]

            # Add tasks to the table
            for task_name, task_info in displaying_tasks:
                self._add_task_row(task_name, task_info)

            # Add a separator row between workflows
            self.table.add_section()

        # Compute total duration
        if runs:
            sorted_tasks = sorted(
                multi_run_task_list,
                key=lambda t: (
                    self.time_or_now(t[1].submission_time),
                    self.time_or_now(t[1].start_time),
                    self.time_or_now(t[1].end_time),
                ),
                reverse=True,
            )
            multi_run_status = (
                sorted_tasks[-1][1].status
                if RunStatus.finished(sorted_tasks[-1][1].status)
                else RunStatus.running
            )
            total_duration = self._get_run_duration(sorted_tasks, multi_run_status)

        # Populate Header
        self.table.title = self.MULTI_RUN_TITLE_STR.format(total_duration)

    def update_run_status(
        self,
        monitored_runs: List[MonitoredWorkflowRun],
        monitored_warnings: List[Union[str, Warning]],
    ) -> None:
        """Updates the monitor table with the latest runs status.

        :param runs: List of workflow runs to monitor. If only one run is provided,
            the method will monitor that run directly.
        :param monitored_warnings: List of warnings to display in the table.
        """
        if len(monitored_runs) == 1:
            self._populate_single_run_table(monitored_runs[0], monitored_warnings)
        else:
            self._populate_multi_run_table(monitored_runs, monitored_warnings)
        self.live_context.update(self.table, refresh=True)
