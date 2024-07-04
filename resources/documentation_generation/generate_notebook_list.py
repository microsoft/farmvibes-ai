import json
import os
from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Tuple

from jinja2 import Template

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
NOTEBOOK_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "notebooks"))
LIST_TEMPLATE_PATH = os.path.abspath(os.path.join(HERE, "templates", "list_notebook_template.md"))
DOC_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "docs", "source", "docfiles", "markdown"))
OUTPUT_PATH = os.path.abspath(os.path.join(DOC_DIR, "NOTEBOOK_LIST.md"))
GITHUB_URL = "https://github.com/microsoft/farmvibes-ai/blob/main"
PRIVATE_TAG = "private"


@dataclass
class NotebookDataSource:
    name: str
    description: str
    tags: List[Tuple[int, str]]
    repo_path: str
    disk_time_req: str


def render_template(
    data_source: List[NotebookDataSource],
    tag_data_source: List[Tuple[Tuple[int, str], List[NotebookDataSource]]],
    output_path: str,
):
    """Load and render template given a data source"""

    with open(LIST_TEMPLATE_PATH) as f:
        t = Template(f.read())

    rendered_template = t.render(
        data_source=data_source,
        tag_data_source=tag_data_source,
    )

    with open(output_path, "w") as f:
        f.write(rendered_template)


def format_disk_time_req(disk_space: str, running_time: str) -> str:
    """Format the disk space and running time requirements"""
    output_str = "({}{}{})" if disk_space or running_time else "{}{}{}"
    sep = ", " if disk_space and running_time else ""
    return output_str.format(disk_space, sep, running_time)


def parse_nb_metadata(nb_path: str) -> Optional[NotebookDataSource]:
    """Parse the ipynb to extract its metadata"""
    with open(nb_path) as f:
        nb_json = json.load(f)

    try:
        nb_metadata = nb_json["metadata"]
    except KeyError:
        raise KeyError(f"Notebook {nb_path} has no metadata")

    # Parse tag order
    nb_tags = []
    try:
        tags = nb_metadata["tags"]
    except KeyError:
        raise KeyError(f"Notebook {nb_path} with metadata {nb_metadata} has no tags")

    for tag in tags:
        tag_components = tag.split("_")
        if len(tag_components) == 2:
            tag_order = int(tag_components[0])
            tag_name = tag_components[-1]
        else:
            tag_order = inf
            tag_name = tag_components[-1]

        if tag_name == PRIVATE_TAG:
            return None
        nb_tags.append((tag_order, tag_name))

    nb_name = nb_metadata["name"]
    nb_description = nb_metadata["description"]
    nb_repo_path = f"{GITHUB_URL}{nb_path.split(PROJECT_DIR)[-1]}"
    nb_disk_time_req = format_disk_time_req(nb_metadata["disk_space"], nb_metadata["running_time"])

    return NotebookDataSource(
        name=nb_name,
        description=nb_description,
        tags=nb_tags,
        repo_path=nb_repo_path,
        disk_time_req=nb_disk_time_req,
    )


def list_notebooks() -> List[str]:
    """Iterate over NOTEBOOK_DIR and retrieve all ipynb paths"""
    notebook_list: List[str] = []

    for folder, _, nb_files in os.walk(NOTEBOOK_DIR):
        for nb_file in nb_files:
            if nb_file.endswith(".ipynb"):
                nb_path = os.path.abspath(os.path.join(folder, nb_file))
                notebook_list.append(nb_path)

    return notebook_list


def sort_tags(
    tag_data_source: Dict[Tuple[int, str], List[NotebookDataSource]]
) -> List[Tuple[Tuple[int, str], List[NotebookDataSource]]]:
    """Sort tags by tag order and then by name"""
    sorted_tags_ds = []
    for tag_tuple, nb_data_source_list in tag_data_source.items():
        sorted_nb_data_source_list = sorted(nb_data_source_list, key=lambda x: x.name)
        sorted_tags_ds.append((tag_tuple, sorted_nb_data_source_list))
    sorted_tags_ds = sorted(sorted_tags_ds, key=lambda x: x[0])
    return sorted_tags_ds


def build_notebook_list():
    """Build the notebook list page"""
    data_source: List[NotebookDataSource] = []
    tag_data_source: Dict[Tuple[str, int], List[NotebookDataSource]] = {}

    # List notebooks in NOTEBOOK_DIR
    notebook_list = list_notebooks()

    # For each notebook, parse the json metadata and get attributes
    for notebook_path in notebook_list:
        notebook_data_source = parse_nb_metadata(notebook_path)

        if notebook_data_source:
            # Add notebook to data source
            data_source.append(notebook_data_source)

            # Add notebook to tag list
            for tag_tuple in notebook_data_source.tags:
                if tag_tuple not in tag_data_source:
                    tag_data_source[tag_tuple] = []
                tag_data_source[tag_tuple].append(notebook_data_source)

    # Sort data source by name
    data_source = sorted(data_source, key=lambda x: x.name)

    # Sort tag data source by tag order and name
    sorted_tags_ds = sort_tags(tag_data_source)

    # Render template
    render_template(data_source, sorted_tags_ds, OUTPUT_PATH)


def main():
    build_notebook_list()


if __name__ == "__main__":
    main()
