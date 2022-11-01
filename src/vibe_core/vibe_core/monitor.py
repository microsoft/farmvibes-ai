from datetime import datetime
from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table

from vibe_core.datamodel import RunDetails, RunStatus, TaskDescription

LEFT_BORDER_PADDING = (0, 0, 0, 4)
CONSOLE_WIDTH = 100

STATUS_STR_MAP = {
    RunStatus.pending: "[yellow]pending[/]",
    RunStatus.running: "[cyan]running[/]",
    RunStatus.failed: "[red]failed[/]",
    RunStatus.done: "[green]done[/]",
    RunStatus.cancelled: "[yellow]cancelled[/]",
    RunStatus.cancelling: "[yellow]cancelling[/]",
}

FETCHING_INFO_STR = ":hourglass_not_done: [yellow]Fetching information...[/]"


def strftimedelta(start: datetime, end: datetime) -> str:
    """Method that formats the string of a timedelta (end - time)"""
    tdelta = end - start
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_typing(type_dict: Dict[str, str]) -> Dict[str, str]:
    return {k: v.replace("typing.", "") for k, v in type_dict.items()}


class VibeWorkflowDocumenter:
    """Class that implements the printing/formatting of workflow descriptions"""

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
        parameters: Dict[str, str],
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
        return {
            param_name: "default: task defined"
            if param_value is None
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
            self._print_items_description(
                self.description.parameters, section_name, self.formatted_parameters
            )

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

    def print_documentation(self) -> None:
        self._print_header()
        self._print_sources()
        self._print_sinks()
        self._print_parameters()
        self._print_tasks()


class VibeWorkflowRunMonitor:
    """Class that abstracts the printing/formatting of workflow run status"""

    TITLE_STR = (
        "[not italic]:earth_americas: "
        "FarmVibes.AI :earth_africa: "
        "[dodger_blue3]{}[/] :earth_asia: \n"
        "Run name: [dodger_blue3]{}[/]\n"
        "Run id: [dark_green]{}[/][/]"
    )
    TABLE_FIELDS = ["Task Name", "Status", "Start Time", "End Time", "Duration"]

    def __init__(self):
        self._populate_table()

        console = Console()
        console.clear()
        self.live_context = Live(self.table, console=console, screen=False, auto_refresh=False)

    def _add_row(self, task_name: str, task_info: RunDetails) -> None:
        start_time = datetime.now() if task_info.start_time is None else task_info.start_time
        start_time_str = start_time.strftime("%Y/%m/%d %H:%M:%S")

        if task_info.status == RunStatus.running or task_info.end_time is None:
            end_time_str = ""
            duration = strftimedelta(start_time, datetime.now())
        else:
            end_time_str = task_info.end_time.strftime("%Y/%m/%d %H:%M:%S")
            duration = strftimedelta(start_time, task_info.end_time)

        self.table.add_row(
            task_name,
            STATUS_STR_MAP[task_info.status],
            start_time_str,
            end_time_str,
            duration,
        )

    def _init_table(self):
        """Create a new table and populate with wf-agnostic info"""
        self.table = Table(show_footer=False)
        for col_name in self.TABLE_FIELDS:
            self.table.add_column(col_name)

        # Set current time as caption
        self.table.caption = f"Last update: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}"

    def _populate_table(
        self,
        wf_name: Union[str, Dict[str, Any]] = ":hourglass_not_done:",
        run_name: str = ":hourglass_not_done:",
        run_id: str = ":hourglass_not_done:",
        wf_tasks: Optional[Dict[str, RunDetails]] = None,
    ) -> None:
        """Method that creates a new table with updated task info"""

        # Create new table
        self._init_table()

        # Populate Header
        # Do not print the whole dict definition if it is a custom workflow
        wf_name = f"Custom: '{wf_name['name']}'" if isinstance(wf_name, dict) else wf_name
        self.table.title = self.TITLE_STR.format(wf_name, run_name, run_id)

        # Populate Rows
        if wf_tasks is None:
            self.table.add_row(FETCHING_INFO_STR)
        else:
            # Sort tasks by reversed start time (running tasks will be on top)
            sorted_tasks = sorted(
                wf_tasks.items(),
                key=lambda t: (
                    t[1].start_time,
                    datetime.now() if t[1].end_time is None else t[1].end_time,  # type: ignore
                ),
                reverse=True,
            )

            # Add each task to the table
            for task_name, task_info in sorted_tasks:
                self._add_row(task_name, task_info)

    def update_task_status(
        self,
        wf_name: Union[str, Dict[str, Any]],
        run_name: str,
        run_id: str,
        wf_tasks: Dict[str, RunDetails],
    ):
        """Recreate the table and update context"""
        self._populate_table(wf_name, run_name, run_id, wf_tasks)
        self.live_context.update(self.table, refresh=True)
