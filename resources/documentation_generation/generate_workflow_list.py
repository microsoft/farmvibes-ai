import os
from dataclasses import dataclass
from typing import Dict, List, Union

import yaml
from jinja2 import Template

from vibe_core.client import FarmvibesAiClient
from vibe_core.datamodel import TaskDescription
from vibe_server.workflow.spec_parser import WorkflowParser

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
DOC_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "docs", "source", "docfiles", "markdown"))
WORKFLOW_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "workflows"))

WF_LIST_TEMPLATE_PATH = os.path.abspath(
    os.path.join(HERE, "templates", "list_workflow_template.md")
)
WF_LIST_OUTPUT_PATH = os.path.abspath(os.path.join(DOC_DIR, "WORKFLOW_LIST.md"))

WF_YAML_TEMPLATE_PATH = os.path.abspath(
    os.path.join(HERE, "templates", "workflow_yaml_template.md")
)
WF_YAML_OUTPUT_DIR = os.path.abspath(os.path.join(DOC_DIR, "workflow_yaml"))

WF_CATEGORY_LIST = ["data_ingestion", "data_processing", "farm_ai", "forest_ai", "ml"]


@dataclass
class WorkflowInformation:
    name: str
    description: Union[str, TaskDescription]
    markdown_link: str
    yaml: str
    mermaid_diagram: str


@dataclass
class TemplateDataSource:
    category: str
    wf_list: List[WorkflowInformation]


def format_wf_name(full_wf_name: str, category: str):
    return full_wf_name.split(f"{category}/")[-1]


def render_template(
    data_source: Union[List[TemplateDataSource], WorkflowInformation],
    output_path: str,
    template_path: str,
):
    """Load and render template given a data source"""

    with open(template_path) as f:
        t = Template(f.read())

    rendered_template = t.render(data_source=data_source)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write(rendered_template)


def list_exposed_workflows() -> Dict[str, List[str]]:
    """Call the REST API to list the workflows"""

    workflow_list = FarmvibesAiClient("http://localhost:1108/").list_workflows()

    category_dict = {
        cat: [wf_name for wf_name in workflow_list if wf_name.startswith(cat)]
        for cat in WF_CATEGORY_LIST
    }
    return category_dict


def parse_wf_yamls(category: str, wf_list: List[str]) -> List[WorkflowInformation]:
    """Parse the wf yaml files to extract short description"""
    parsedList = []

    client = FarmvibesAiClient("http://localhost:1108/")

    for wf_name in wf_list:
        wf_yaml = client.get_workflow_yaml(wf_name)
        yaml_dict = yaml.safe_load(wf_yaml)
        wf_spec = WorkflowParser.parse_dict(yaml_dict)

        wf_md_link = os.path.relpath(
            path=os.path.join(WF_YAML_OUTPUT_DIR, f"{wf_name}.md"), start=DOC_DIR
        )

        wf_name = format_wf_name(wf_name, category)

        parsedList.append(
            WorkflowInformation(
                name=wf_name,
                description=wf_spec.description.short_description,
                markdown_link=wf_md_link,
                yaml=wf_yaml,
                mermaid_diagram="",
            )
        )

    return sorted(parsedList, key=lambda x: x.name)


def build_workflow_list():
    """Build the worflow list page from the client"""
    data_source: List[TemplateDataSource] = []

    # List workflows in the REST API
    wf_per_category = list_exposed_workflows()

    # For each workflow, parse the yaml and get description
    for category, wf_list in wf_per_category.items():
        data_source.append(
            TemplateDataSource(category=category, wf_list=parse_wf_yamls(category, wf_list))
        )

    render_template(data_source, WF_LIST_OUTPUT_PATH, WF_LIST_TEMPLATE_PATH)


def build_workflow_yamls():
    """Build the workflow yaml pages from the client"""
    client = FarmvibesAiClient("http://localhost:1108/")

    for wf_name in client.list_workflows():
        wf_yaml = client.get_workflow_yaml(wf_name)
        yaml_dict = yaml.safe_load(wf_yaml)
        wf_spec = WorkflowParser.parse_dict(yaml_dict)

        description = client.describe_workflow(wf_name)["description"]

        wf_yaml_output_path = os.path.join(WF_YAML_OUTPUT_DIR, f"{wf_name}.md")
        if not os.path.exists(os.path.dirname(wf_yaml_output_path)):
            os.makedirs(os.path.dirname(wf_yaml_output_path))

        data_source = WorkflowInformation(
            name=wf_name,
            description=description,
            markdown_link="",
            yaml=wf_yaml,
            mermaid_diagram=wf_spec.to_mermaid(),
        )

        render_template(data_source, wf_yaml_output_path, WF_YAML_TEMPLATE_PATH)


def main():
    build_workflow_list()
    build_workflow_yamls()


if __name__ == "__main__":
    main()
